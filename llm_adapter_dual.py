# llm_adapter_dual.py — отдельный адаптер под ДВА сервера: OCR и Перевод
# Требует: llama.cpp server с vision (OCR) и text (перевод).
# Порты/модели раздельные.
import os
import re
import time, threading
from dataclasses import dataclass
import json

import requests

_SESSION = requests.Session()

# === ЗАГРУЗКА СПИСКОВ ГАЛЛЮЦИНАЦИЙ ===
_BAD_PHRASES = {"ocr": [], "translation": []}
_BAD_PHRASES_PATH = os.path.join(os.path.dirname(__file__), "assets", "bad_phrases.json")

def _load_bad_phrases():
    global _BAD_PHRASES
    try:
        if os.path.exists(_BAD_PHRASES_PATH):
            with open(_BAD_PHRASES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Приводим к нижнему регистру сразу при загрузке
                if "ocr" in data:
                    _BAD_PHRASES["ocr"] = [x.lower() for x in data["ocr"]]
                if "translation" in data:
                    _BAD_PHRASES["translation"] = [x.lower() for x in data["translation"]]
            print(f"[DUAL] Loaded {len(_BAD_PHRASES['ocr'])} OCR filters and {len(_BAD_PHRASES['translation'])} TR filters.")
    except Exception as e:
        print(f"[DUAL] Error loading bad_phrases.json: {e}")

# Загружаем сразу при старте модуля
_load_bad_phrases()

# ============== Мини-история диалога (EN) для MT-модели ================
_HISTORY_LOCK = threading.Lock()
_HISTORY_MAX = 10  # Уменьшаем до 10 для стабильности (было 25)
_HISTORY_EN: list[str] = []
_HISTORY_LOG_PATH = os.path.join(os.path.dirname(__file__), "dialog_history.txt")
_MEMORY_ENABLED = True
_LAST_EN_TEXT = ""
_LAST_RU_TEXT = ""

# ============== РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ ================
_RE_MYSTERY = re.compile(r"[?？]+")
_RE_LEADING_DOTS = re.compile(r"^[.…\s]+")
_RE_PAREN_STD = re.compile(r"\(.*?\)")
_RE_PAREN_CJK = re.compile(r"（.*?）")
_RE_NON_WORD = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_RE_NUMERIC_LIKE = re.compile(r"^(?:Room|Level|Floor|Timer|Stage|Chapter|Part)\s*[\d\s\:\-\.]+$|^[\d\s\:\-\.]+$", re.IGNORECASE)
_RE_SPECIAL_TOKENS = re.compile(r"(?:<\|.*?\|>|&lt;\|.*?\|?&gt;|<s>|</s>|&lt;/s&gt;|<end_of_turn>|<start_of_turn>)", flags=re.IGNORECASE)

# ============== LORE DB (персонажи/имена) ================
_LORE_DB_LOCK = threading.Lock()
_LORE_DB_INIT = False
_CHAR_DB: dict[str, dict] = {}  # canon_en_name -> {en, ru, gender, raw}

from difflib import SequenceMatcher

def _add_dialog_history(name: str | None, body: str, ru_body: str = "") -> None:
    """Сохраняем реплику вида 'Name: text >>> translation'."""
    if not body:
        return
    body = " ".join((body or "").split())
    if not body:
        return
    
    # Очистка от мусора для сравнения
    clean_body = re.sub(r"[\s.…?！!]+", "", body).lower()
    if not clean_body:
        return # Не сохраняем в историю пустые вздохи/троеточия

    line = body
    if name:
        name = " ".join((name or "").split())
        if name:
            line = f"{name}: {body}"
            
    # Добавляем русский перевод в историю, если есть
    if ru_body:
        ru_clean = " ".join((ru_body or "").split())
        if ru_clean:
            line += f"  >>>  {ru_clean}"

    with _HISTORY_LOCK:
        # 1. Проверка на идентичность последней строки
        if _HISTORY_EN and _HISTORY_EN[-1] == line:
            return
        
        # 2. Проверка на высокую схожесть (анти-спам при шуме OCR)
        if _HISTORY_EN:
            last_line = _HISTORY_EN[-1]
            # Сравниваем только английскую часть (до >>>)
            last_en = last_line.split("  >>>  ")[0]
            curr_en = line.split("  >>>  ")[0]
            if SequenceMatcher(None, last_en, curr_en).ratio() > 0.85:
                # Если фразы почти одинаковые (разница в пару символов), обновляем последнюю
                _HISTORY_EN[-1] = line
                return

        _HISTORY_EN.append(line)
        if len(_HISTORY_EN) > _HISTORY_MAX:
            del _HISTORY_EN[0 : len(_HISTORY_EN) - _HISTORY_MAX]
            
        try:
            with open(_HISTORY_LOG_PATH, "w", encoding="utf-8") as f:
                for ln in _HISTORY_EN:
                    f.write(ln + "\n")
        except Exception as e:
            print("[CTX] history log write error:", e)


def _build_dialog_context_block(max_chars: int = 4000) -> str:
    """Формируем текстовый блок с последними репликами для промпта."""
    with _HISTORY_LOCK:
        if not _HISTORY_EN:
            return ""
        collected: list[str] = []
        total = 0
        for ln in reversed(_HISTORY_EN):
            extra = len(ln) + 2
            if collected and total + extra > max_chars:
                break
            collected.append(ln)
            total += extra
        collected.reverse()

    # Используем формат чата без нумерации, чтобы не провоцировать списки
    history_text = "\n".join(collected)
    return (
        "### Recent Dialogue History:\n"
        f"{history_text}\n"
        "### End of History\n\n"
    )

def set_memory_enabled(enabled: bool) -> None:
    """Включение/выключение диалоговой памяти из UI."""
    global _MEMORY_ENABLED
    _MEMORY_ENABLED = bool(enabled)
    # по желанию можно чистить историю при отключении:
    if not _MEMORY_ENABLED:
        with _HISTORY_LOCK:
            _HISTORY_EN.clear()
        try:
            with open(_HISTORY_LOG_PATH, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass

def _canon_name(s: str) -> str:
    """Канонизация имени: режем мусор, но оставляем иероглифы."""
    s = (s or "").strip()
    
    # Разрешаем имя, состоящее только из знаков вопроса (???)
    if _RE_MYSTERY.fullmatch(s):
        return s

    # убираем ведущие точки/многоточия
    s = re.sub(r"^[.…\s]+", "", s)
    s = _RE_LEADING_DOTS.sub("", s)
    # убираем содержимое скобок (любых видов)
    s = _RE_PAREN_STD.sub("", s)      # обычные
    s = _RE_PAREN_CJK.sub("", s)      # китайские/японские
    
    # Очистка: оставляем буквы, цифры и иероглифы.
    # \w в Python 3 включает иероглифы, но мы исключим подчеркивание.
    # Проще удалить явные знаки препинания и символы.
    
    # Удаляем знаки препинания (ASCII + CJK)
    # [^\w] удалит пробелы, поэтому меняем на пробелы.
    s = _RE_NON_WORD.sub(" ", s) 
    
    return " ".join(s.split()).lower()


def _load_lore_db_from_text(lore_text: str) -> None:
    """Разбираем game_bible_exilium.txt в маленькую БД персонажей."""
    global _LORE_DB_INIT, _CHAR_DB
    if _LORE_DB_INIT:
        return

    with _LORE_DB_LOCK:
        if _LORE_DB_INIT:
            return

        char_db: dict[str, dict] = {}

        for line in lore_text.splitlines():
            t = line.strip()
            if not t or t.startswith("#") or t.startswith("//") or "===" in t:
                continue

            # Формат: NameEN -> NameRU | пол: женский/мужской
            # Поддерживаем разделители: ->, →, - (с пробелами)
            m = re.match(
                r"^(?P<en>[^#=→:|]+?)\s*(?:->|→| - )\s*(?P<ru>[^|#]+?)\s*(?:\|\s*пол\s*:\s*(?P<gender>[^|#]+))?\s*$",
                t,
                re.IGNORECASE,
            )
            if not m:
                continue

            en = m.group("en").strip()
            ru = m.group("ru").strip()
            gender_raw = (m.group("gender") or "").strip().lower()

            gender = None
            if "жен" in gender_raw:
                gender = "F"
            elif "муж" in gender_raw:
                gender = "M"

            cname = _canon_name(en)
            if not cname:
                continue

            char_db[cname] = {
                "en": en,
                "ru": ru,
                "gender": gender,
                "raw": t,
            }

        _CHAR_DB = char_db
        _LORE_DB_INIT = True
        print(f"[DUAL][LORE] char_db size={len(_CHAR_DB)}")


def _get_char_info_from_name(name_line: str | None) -> dict | None:
    """Находим персонажа по строке имени (из EN)."""
    if not name_line:
        return None
    cname = _canon_name(name_line)
    if not cname:
        return None
    with _LORE_DB_LOCK:
        return _CHAR_DB.get(cname)


def _build_lore_snippet_for_text(
    name_line: str | None, body: str, max_items: int = 5
) -> tuple[str, str | None]:
    """
    Для текущей реплики собираем краткий LORE-сниппет.
    Адаптировано для CJK: ищем вхождения имен из базы в тексте, а не бьем текст на слова.
    """
    snippet_lines: list[str] = []
    speaker_gender: Optional[str] = None
    seen_cnames: set[str] = set()

    # 1) Сам спикер
    speaker_info = _get_char_info_from_name(name_line)
    if speaker_info:
        cname = _canon_name(speaker_info['en'])
        seen_cnames.add(cname)
        
        g = speaker_info.get("gender")
        g_desc = "female" if g == "F" else "male" if g == "M" else None

        line = f"- {speaker_info['en']} -> {speaker_info['ru']}"
        if g_desc: line += f" ({g_desc})"
        snippet_lines.append(line)

        speaker_gender = g

    # 2) Другие имена/термины в тексте (НОВАЯ ЛОГИКА)
    # Перебираем базу имен и ищем их вхождение в body
    
    found_infos = []
    
    with _LORE_DB_LOCK:
        # Берем список всех записей
        all_chars = list(_CHAR_DB.values())

    # Сортируем по длине имени (сначала длинные), чтобы найти "Mosin-Nagant" раньше "Mosin"
    all_chars.sort(key=lambda x: len(x["en"]), reverse=True)
    
    body_lower = body.lower() if body else ""
    
    for info in all_chars:
        # Если слотов под подсказки уже нет — выходим
        if len(snippet_lines) + len(found_infos) >= max_items:
            break

        en_key = info["en"] # Оригинальное написание (например "グローザ" или "Dobermann")
        cname = _canon_name(en_key)
        
        if cname in seen_cnames:
            continue
            
        is_found = False
        
        # Проверяем, есть ли в имени CJK иероглифы
        is_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', en_key))
        
        if is_cjk:
            # Для японского/китайского: простое вхождение подстроки
            if en_key in body:
                is_found = True
        else:
            # Для английского/русского: ищем слово целиком (чтобы "Al" не находился внутри "Always")
            # Используем regex с границами слов \b (или аналогом)
            # Экранируем имя, чтобы спецсимволы не ломали regex
            esc_name = re.escape(en_key.lower())
            # (?<!\w) - слева нет буквы, (?!\w) - справа нет буквы
            pattern = r"(?<!\w)" + esc_name + r"(?!\w)"
            if re.search(pattern, body_lower):
                is_found = True
        
        if is_found:
            found_infos.append(info)
            seen_cnames.add(cname)

    # Добавляем найденных в список
    for info in found_infos:
        g = info.get("gender")
        g_desc = "female" if g == "F" else "male" if g == "M" else None
        
        line = f"- {info['en']} -> {info['ru']}"
        if g_desc: line += f" ({g_desc})"
        snippet_lines.append(line)

    if not snippet_lines:
        return "", speaker_gender

    snippet = (
        "<lore>\n"
        + "\n".join(snippet_lines)
        + "\n</lore>\n"
    )
    return snippet, speaker_gender

# ============== Общая конфигурация слотов ================
@dataclass
class LLMConfig:
    server: str = "http://127.0.0.1:8080"
    model: str = ""
    timeout_s: float = 30.0
    max_tokens: int = 1024
    temp: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repeat_penalty: float = 1.0
    seed: int = 0
    slot_id: int = 0
    source_lang: str = "en"  # en, ja, ko, ch_sim
    mode: str = "game"       # "game" | "wiki"
    # system:
    system_override: str | None = None
    # контекстные файлы:
    lore_path: str | None = None
    phrasebook_path: str | None = None
    max_ctx_chars: int = 30000
    # prompt-cache:
    use_prompt_cache: bool = True
    disable_name_split: bool = False

# ======= OCR: минимальный system для считывания текста =====
_OCR_SYSTEM_DEFAULT = (
    "You are a professional OCR tool. Your ONLY task is to transcribe visible text from the image.\n"
    "- Return ONLY the exact text you see. NO summaries, NO explanations, NO continuations.\n"
    "- If a character is silent (e.g., just '...'), output ONLY the name and '...'. DO NOT invent thoughts.\n"
    "- CRITICAL: DO NOT guess, hallucinate, or 'complete' sentences that are not there.\n"
    "- CRITICAL: If no readable text is found, output EXACTLY: <NO_TEXT_FOUND>\n"
    "- IMPORTANT: Pay absolute attention to punctuation like ellipses (...) or multiple dots.\n"
    "- Preserve layout: keep speaker names on a separate line or format as 'Name: Dialogue'.\n"
    "- Do NOT add any text that is not physically present on the image."
)

# ======= Translator: system с LORE/PHRASEBOOK вшитыми ======
def _read_text(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _build_tr_system(lore_text: str, phrasebook_text: str, lang_code: str = "en") -> str:
    """
    Генерация 'вшитого' промпта. 
    Адаптируется под выбранный в меню язык.
    """
    
    # 1. Карта языков
    lang_map = {
        "en": "English",
        "ja": "Japanese",
        "ch_sim": "Chinese (Simplified)",
        "ko": "Korean",
        "ru": "Russian"
    }
    src_lang = lang_map.get(lang_code, "English")
    
    # 2. Подбираем стиль в зависимости от языка
    style_block = (
        "STYLE & TONE\n"
        "- Style: Literary, natural Russian translation.\n"
        "- Do NOT invent or add any text not present in the User message.\n"
        "- If the input text consists only of ellipses, sighs, or punctuation (e.g., '...', '!!!', '—'), return it UNCHANGED.\n"
        "- CRITICAL: Do NOT complete the sentence using the provided history. Translate ONLY the current message.\n"
    )

    # 3. Собираем итоговый промпт
    return (
        f"You are a professional video game translator from {src_lang} to Russian.\n"
        "Your goal is a natural, high-quality localization that preserves the original tone, context, and atmosphere.\n\n"
        
        "CONTEXT & TONE:\n"
        "- Adapt to the speaker's identity, role, and the current game setting.\n"
        "- Pay attention to the relationship context to choose between formal 'вы' and informal 'ты'.\n"
        "- Pay close attention to grammatical gender in Russian. Use dialogue history and provided lore tags (e.g., [M] or [F]) to correctly determine male or female verb endings.\n\n"
        
        "TRANSLATION GUIDELINES:\n"
        "1. Output ONLY the Russian translation. Do not add notes, comments, or XML tags.\n"
        "2. The 'Dialogue History' is provided ONLY for context. Translate ONLY the final User message.\n"
        "3. Match the exact intensity of the original text, including slang, humor, and profanity.\n"
        "4. INCOMPLETE SENTENCES: If a sentence is cut off (e.g., ends with '-' or '...'), translate it exactly as is. Do NOT attempt to finish the thought or predict the next words.\n"
        "5. If there is no text to translate, return the original punctuation or the word '(затуп)'.\n\n"
        
        f"{style_block}\n\n"
        
        "LORE & PHRASEBOOK (Use as absolute truth):\n"
        f"{phrasebook_text[:15000]}\n"
    )

_LORE_INIT = False
_LORE_LOCK = threading.Lock()
_TR_SYSTEM_CACHED = ""
_TR_SYSTEM_KEY = "" # Ключ для проверки актуальности кэша

def _ensure_tr_system(cfg: LLMConfig) -> str:
    global _LORE_INIT, _TR_SYSTEM_CACHED, _TR_SYSTEM_KEY
    
    # Формируем уникальный ключ для текущих настроек
    current_key = f"{cfg.source_lang}|{cfg.lore_path}|{cfg.phrasebook_path}|{cfg.system_override}"
    
    if _LORE_INIT and _TR_SYSTEM_CACHED and _TR_SYSTEM_KEY == current_key:
        return _TR_SYSTEM_CACHED
        
    with _LORE_LOCK:
        # Если ключ изменился, нужно переинициализировать лор
        if _TR_SYSTEM_KEY != current_key:
            global _LORE_DB_INIT
            _LORE_DB_INIT = False 
            
        lore = _read_text(cfg.lore_path)[:cfg.max_ctx_chars] if cfg.lore_path else ""
        pb   = _read_text(cfg.phrasebook_path)[:cfg.max_ctx_chars] if cfg.phrasebook_path else ""

        if lore:
            _load_lore_db_from_text(lore)

        if cfg.system_override:
            sys_txt = (cfg.system_override
                       .replace("{{LORE}}", lore)
                       .replace("{{PHRASEBOOK}}", pb))
        else:
            sys_txt = _build_tr_system(lore, pb, cfg.source_lang)
            
        _TR_SYSTEM_CACHED = sys_txt
        _TR_SYSTEM_KEY = current_key
        _LORE_INIT = True
        print(f"[DUAL] translator system ready (cache updated): lore={len(lore)} chars, pb={len(pb)} chars")
        return sys_txt

def reset_tr_system_cache():
    global _LORE_INIT, _TR_SYSTEM_CACHED
    with _LORE_LOCK:
        _LORE_INIT = False
        _TR_SYSTEM_CACHED = ""

def reset_translation_cache():
    """Сброс кэша последнего перевода (анти-эхо)."""
    global _LAST_EN_TEXT, _LAST_RU_TEXT
    _LAST_EN_TEXT = ""
    _LAST_RU_TEXT = ""

def _split_name_and_body_cjk(text: str) -> tuple[str | None, str]:
    """Спец-логика для CJK (Япония/Китай)."""
    text = text.strip()
    
    # 1. Явные японские скобки-маркеры имени: 【Name】 Text
    m = re.match(r"^【(.*?)】\s*(.*)", text, re.DOTALL)
    if m: return m.group(1), m.group(2)

    # 2. Имя и кавычка: Name 「Text」 или Name「Text」
    # Ищем паттерн: (Любые символы, кроме кавычек) (Открывающая кавычка)
    m = re.match(r"^([^「『\n]+)\s*([「『].*)", text, re.DOTALL)
    if m:
        name_cand = m.group(1).strip()
        body = m.group(2)
        # Имя не должно быть слишком длинным (обычно 2-6 иероглифов)
        if len(name_cand) < 15:
            return name_cand, body

    # 3. Широкое двоеточие (Name：Text)
    if "：" in text:
        parts = text.split("：", 1)
        name_cand = parts[0].strip()
        if len(name_cand) < 15 and "\n" not in name_cand:
            return name_cand, parts[1].strip()

    # 4. Если ничего не нашли, пробуем стандартный Lore-поиск
    # (он сработает внутри основной функции, так что тут просто возвращаем None)
    return None, text

def _split_name_and_body(en_text: str, lang: str = "en") -> tuple[str | None, str]:
    """
    Делим сырой OCR-текст на Name и Body.
    Поддерживает:
    1. Вертикальный формат (Имя \n Текст)
    2. Горизонтальный формат (Имя Текст), если Имя есть в ЛОРЕ.
    """
    if not en_text:
        return None, ""

    if lang in ("ja", "ch_sim", "ko"):
        nm, bd = _split_name_and_body_cjk(en_text)
        if nm: 
            return nm, bd
    
    # --- ЧАСТЬ 1: Стандартная проверка (Вертикальная) ---
    lines = [ln.rstrip("\r") for ln in en_text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return None, ""

    # FIX: Защита цифр от фильтра имён (Баг "Комната 456")
    if _RE_NUMERIC_LIKE.fullmatch(en_text.strip()):
        return None, en_text # Вся строка - это не имя, а номер/таймер

    raw_first = lines[0].strip()
    first = re.sub(r"^[.…\s]+", "", raw_first)
    
    # Проверяем, есть ли имя в базе (LORE), игнорируя точки
    is_known_name = False
    try:
        cn = _canon_name(first)
        with _LORE_DB_LOCK:
            if cn in _CHAR_DB:
                is_known_name = True
            else:
                # Умный фильтр мусора: если OCR приклеил анимацию/иконку спереди (например "I... Chiloveig")
                for k_cn, v_info in _CHAR_DB.items():
                    if cn.endswith(k_cn):
                        diff_len = len(cn) - len(k_cn)
                        # Разрешаем до 4 символов каноничного мусора (например "i " или "a ")
                        if 0 < diff_len <= 4 and cn.endswith(" " + k_cn):
                            is_known_name = True
                            first = v_info["en"] # Подменяем грязное имя на чистое из базы
                            break
    except Exception: pass

    # Если строк несколько — проверяем первую как имя
    # FIX: Защита цифр от фильтра имён (Баг "Комната 456")
    if _RE_NUMERIC_LIKE.fullmatch(first):
        return None, en_text # Первая строка - это не имя, а номер/таймер

    if len(lines) > 1:
        # Разрешаем ??? как имя, даже если там есть знак вопроса
        is_mystery = bool(re.fullmatch(r"[?？]+", first))
        
        if is_known_name or is_mystery or (len(first) <= 32 and len(first.split()) <= 3 and not any(ch in first for ch in ".!?;:")):
            found_name = first
            body_lines = lines[1:]

            # --- FIX: Проверка на второе имя (для сцен с двумя персонажами) ---
            if body_lines:
                raw_sec = body_lines[0].strip()
                sec = re.sub(r"^[.…\s]+", "", raw_sec)
                
                is_sec_known = False
                try:
                    cn_sec = _canon_name(sec)
                    with _LORE_DB_LOCK:
                        if cn_sec in _CHAR_DB: is_sec_known = True
                except: pass
                
                # Если вторая строка тоже похожа на имя (есть в базе ИЛИ короткая/без знаков/с большой буквы)
                if is_sec_known or (len(sec) <= 25 and len(sec.split()) <= 3 and not re.search(r"[.,!?;:]", sec) and sec and sec[0].isupper() and len(sec) > 1):
                    body_lines = body_lines[1:] # Пропускаем вторую строку (имя собеседника)

            return found_name, "\n".join(body_lines).lstrip("\n")
    
    # --- ЧАСТЬ 2: Горизонтальная проверка (УМНАЯ) ---
    full_line = lines[0].strip()
    
    # А. Явный разделитель (двоеточие)
    colon_match = re.match(r"^(.{2,20}?):\s+(.*)", full_line)
    if colon_match:
        candidate = colon_match.group(1).strip()
        if len(candidate.split()) <= 3:
            # --- ИСПРАВЛЕНИЕ: Добавляем остальные строки, если они есть ---
            body_part = colon_match.group(2)
            if len(lines) > 1:
                body_part += "\n" + "\n".join(lines[1:])
            # --------------------------------------------------------------
            return candidate, body_part

    # Б. Поиск по базе имен + Эвристики
    # Сортируем имена по длине, чтобы сначала найти "Mosin-Nagant", а не "Mosin"
    with _LORE_DB_LOCK:
        known_names = sorted(_CHAR_DB.keys(), key=len, reverse=True)
    
    canon_line = _canon_name(full_line)
    
    for cname in known_names:
        idx = canon_line.find(cname)
        # Разрешаем до 4 символов мусора перед именем (спасает горизонтальный формат)
        if idx == 0 or (0 < idx <= 4 and canon_line[:idx].endswith(" ")):
            orig_name_en = _CHAR_DB[cname]["en"]
            
            # Ищем имя в оригинальной строке (регистронезависимо), допуская мусор
            esc_name = re.escape(orig_name_en)
            pattern = r"^.{0,6}?" + esc_name if idx > 0 else r"^" + esc_name
            match = re.search(pattern, full_line, re.IGNORECASE)
            
            if match:
                end_pos = match.end()
                remainder = full_line[end_pos:] # Хвост строки
                
                if not remainder.strip():
                    return None, full_line

                is_separator = False
                
                # 1. Явные знаки (: - –)
                if remainder.startswith(":") or remainder.startswith(" -") or remainder.startswith(" –"):
                    is_separator = True
                
                # 2. Большой пробел (2+ пробела) или таб
                elif re.match(r"^\s{2,}", remainder) or remainder.startswith("\t"):
                    is_separator = True

                # 3. ЭВРИСТИКА ЗАГЛАВНОЙ БУКВЫ (Для случая, когда OCR съел пробелы)
                # Если после имени идет 1 пробел, а затем ЗАГЛАВНАЯ буква или спецсимвол (" ' . …)
                # "Dobermann I really..." -> ' I' -> Match! -> Режем.
                # "Dobermann realized..." -> ' r' -> No match -> Не режем (нарратив).
                elif re.match(r"^\s+[A-Z0-9\"'“‘\(\[\.…—]", remainder):
                    is_separator = True
                    
                if is_separator:
                    found_name = orig_name_en # Используем чистое имя вместо захваченного мусора
                    # Очищаем остаток от разделителей/пробелов в начале
                    rest_body = re.sub(r"^[:\-\–\s]+", "", remainder)
                    
                    if len(lines) > 1:
                        rest_body += "\n" + "\n".join(lines[1:])
                        
                    return found_name, rest_body

    # Г. Эвристика для неизвестных имен в горизонтальном тексте.
    # Логика переработана, чтобы избежать ложных срабатываний на обычных предложениях.
    words = full_line.split()
    if len(words) > 1:
        first_word = words[0]
        second_word = words[1]

        # 1. Проверяем кандидата в имена (первое слово)
        # Имя может заканчиваться на знаки препинания, которые нужно отсечь для проверки
        name_candidate = first_word.rstrip(',.…!?:')

        # Условие: Слово с большой буквы, без знаков препинания в середине.
        # Разрешаем дефисы в именах (Mosin-Nagant).
        PUNCTUATION_IN_NAME = ".!?;:"
        if (
            first_word[0].isupper() and
            len(name_candidate) > 1 and
            not any(c in name_candidate for c in PUNCTUATION_IN_NAME)
        ):
            # 2. Проверяем, не является ли это слово обычным началом предложения.
            # Расширенный список для фильтрации.
            common_starters = (
                # pronouns
                "i", "a", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                # possessive
                "my", "your", "his", "her", "its", "our", "their",
                # articles/determiners
                "the", "an", "this", "that", "these", "those", "all", "any", "some", "every", "no", "such",
                # conjunctions
                "and", "but", "or", "so", "for", "nor", "yet",
                # prepositions
                "as", "at", "by", "in", "into", "of", "on", "to", "with", "from", "over", "under", "about", "after",
                # question words
                "what", "when", "where", "why", "how", "who", "which",
                # common verbs / aux
                "is", "are", "was", "were", "be", "been", "am", "has", "have", "had", "do", "does", "did",
                "can", "could", "will", "would", "shall", "should", "may", "might", "must",
                "let", "let's", "said", "get", "go", "make", "know", "think", "see", "come", "take", "want",
                # common adverbs/etc
                "good", "well", "just", "only", "not", "then", "there", "here", "now", "very", "too", "also", "up",
                # contractions
                "i'm", "i've", "i'll", "i'd", "you're", "you've", "he's", "she's", "it's", "we're", "they're",
                "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "can't", "won't", "wouldn't"
            )
            if name_candidate.lower() in common_starters:
                pass # Это обычное слово, пропускаем эвристику
            else:
                # 3. Ключевая эвристика: смотрим на второе слово.
                # Если оно тоже с заглавной, или это кавычка/скобка, то скорее всего первое - имя.
                if second_word and (second_word[0].isupper() or second_word[0] in "\"'(“‘["):
                     print(f"[_split_name_and_body DBG] Horizontal heuristic matched unknown name: '{first_word}'")
                     body_part = " ".join(words[1:])
                     if len(lines) > 1:
                         body_part += "\n" + "\n".join(lines[1:])
                     return first_word, body_part

    # В. Эвристика для "Имя ... " (Тишина/Мысли)
    # Любое имя (без пробелов и знаков) + 2 или более знаков препинания (включая пробелы между ними)
    silence_match = re.match(r"^([^\s.…?？!！:]{1,25})\s*([\s.…?？!！]{2,})$", full_line)
    if silence_match:
        return silence_match.group(1), silence_match.group(2).strip()

    return None, "\n".join(lines)

def _strip_leading_gender_tag(s: str) -> str:
    """
    Убираем в начале строки служебные метки вида [M]/[F]/[М]/[Ф],
    которые иногда галлюцинирует модель.
    """
    if not s:
        return s
    # убираем ведущие пробелы/многоточия
    s = s.lstrip()
    # вырезаем один тег в квадратных скобках в самом начале
    s = re.sub(r'^(?:[.…\s]*)\[(?:m|f|м|ф)\]\s*', "", s, flags=re.IGNORECASE)
    return s

def _normalize_homoglyphs(s: str) -> str:
    """
    Приводим кириллицу к похожей латинице для проверки 'ленивого' перевода.
    Помогает отловить случаи, когда 'O-123' (Lat) превращается в 'О-123' (Cyr).
    """
    s = s.lower()
    table = str.maketrans({
        'а': 'a', 'в': 'b', 'е': 'e', 'к': 'k', 'м': 'm', 'н': 'h', 
        'о': 'o', 'р': 'p', 'с': 'c', 'т': 't', 'у': 'y', 'х': 'x',
        'ё': 'e'
    })
    return s.translate(table)

# ===================== Прогрев cache_prompt ======================
def preload_prompt_cache_ocr(cfg: LLMConfig) -> bool:
    if not cfg.use_prompt_cache:
        return False
    payload = {
        "model": cfg.model,
        "messages": [{"role":"system","content":_OCR_SYSTEM_DEFAULT}],
        "cache_prompt": True,
        "add_generation_prompt": False,
        "max_tokens": 8,
        "slot_id": cfg.slot_id,
    }
    try:
        r = requests.post(cfg.server.rstrip("/") + "/v1/chat/completions",
                          json=payload, timeout=cfg.timeout_s)
        if r.status_code == 200:
            print(f"[DUAL] OCR cache warmed (slot={cfg.slot_id})")
            return True
        print("[DUAL] OCR warmup err:", r.status_code, (r.text or "")[:200])
    except Exception as e:
        print("[DUAL] OCR warmup exc:", e)
    return False

def preload_prompt_cache_tr(cfg: LLMConfig) -> bool:
    if not cfg.use_prompt_cache:
        return False
    system = _ensure_tr_system(cfg)
    payload = {
        "model": cfg.model,
        "messages": [{"role":"system","content":system}],
        "cache_prompt": True,
        "add_generation_prompt": False,
        "max_tokens": 8,
        "slot_id": cfg.slot_id,
    }
    try:
        r = requests.post(cfg.server.rstrip("/") + "/v1/chat/completions",
                          json=payload, timeout=cfg.timeout_s)
        if r.status_code == 200:
            print(f"[DUAL] TR cache warmed (slot={cfg.slot_id})")
            return True
        print("[DUAL] TR warmup err:", r.status_code, (r.text or "")[:200])
    except Exception as e:
        print("[DUAL] TR warmup exc:", e)
    return False

# =========================== Вызовы ==============================
def extract_en_from_image(region_png_b64: str, cfg: LLMConfig) -> str:
    """Вызов на OCR-сервер: картинка -> английский текст (с системой Retry)."""
    url = cfg.server.rstrip("/") + "/v1/chat/completions"
    system_txt = cfg.system_override or _OCR_SYSTEM_DEFAULT
    
    # Стартовые параметры
    max_retries = 2
    current_temp = 0.1       # Начинаем с почти нуля для точности
    current_seed = int(cfg.seed)
    current_penalty = 1.2    # Базовый штраф за повторы
    
    limit_tokens = int(cfg.max_tokens)

    for attempt in range(max_retries + 1):
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_txt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Transcribe all text from this image. Output ONLY the text found, or <NO_TEXT_FOUND> if empty. Be extremely accurate, do not imagine anything."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{region_png_b64}", "detail": "high"}}
                ]},
            ],
            "temperature": current_temp,
            "top_p": 1.0,
            "top_k": 40,
            "repeat_penalty": current_penalty,
            "max_tokens": limit_tokens,
            "add_generation_prompt": True,
            "slot_id": cfg.slot_id,
            "seed": current_seed,
            "stop": ["<|im_end|>", "<|im_start|>", "</s>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "[/INST]"],
        }
        
        t0 = time.perf_counter()
        try:
            r = _SESSION.post(url, json=payload, timeout=cfg.timeout_s)
        except Exception as e:
            print(f"[DUAL][OCR] connection error: {e}")
            return ""

        dt = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            print("[DUAL][OCR] http", r.status_code, "in", f"{dt:.0f} ms", (r.text or "")[:180])
            return ""
            
        js = r.json()
        out = (js.get("choices", [{}])[0]
                  .get("message", {})
                  .get("content") or "").strip()

        # Clean special tokens (like <|im_end|>, </s>, etc.)
        out = _RE_SPECIAL_TOKENS.sub("", out).strip()
        
        # === УМНЫЙ ФИЛЬТР ПУСТОТЫ / ГАЛЛЮЦИНАЦИЙ ===
        # 1. Проверка на "Тишину" с галлюцинацией.
        # Если модель выдает "Имя\n... [куча текста]", а в оригинале скорее всего только "..."
        # Мы можем попробовать детектировать это по паттерну: "... " (многоточие с пробелом) в начале тела.
        if "\n..." in out or out.startswith("..."):
            parts = out.split("\n", 1)
            body = parts[1] if len(parts) > 1 else parts[0]
            if body.startswith("...") and len(body) > 15:
                # Если после многоточия идет ОЧЕНЬ много текста БЕЗ знаков препинания и пробелов — это галлюцинация.
                # Если же там нормальное предложение (есть пробелы, разные буквы) — доверяем OCR.
                letters_only = re.findall(r'[a-zA-Zа-яА-Я\u4e00-\u9fff\u3040-\u30ff]', body)
                words_count = len(body.split())
                if len(letters_only) > 60 and words_count < 2:
                    print(f"[DUAL][OCR] Potential hallucination after ellipsis detected (no spaces). Truncating. Original: {repr(out[:60])}")
                    if len(parts) > 1: out = parts[0] + "\n..."
                    else: out = "..."

        # 2. Проверка на петлю повторений
        if len(out) > 30 and (out.count("?") + out.count("？")) / len(out) > 0.4 or re.search(r"(.{2,})\1{5,}", out):
            print(f"[DUAL][OCR] Repetition loop detected in OCR output: {out[:50]}... Returning empty.")
            return ""
        if "NO_TEXT_FOUND" in out.upper():
            print(f"[DUAL][OCR] VLM explicitly reported no text. Clearing output.")
            return ""

        # --- ПРОВЕРКА НА ПЕТЛЮ (Repetition Check) ---
        # Если строка длинная и состоит в основном из '?' или '？' или повторяющегося мусора
        is_broken = False
        
        if len(out) > 30:
            # Считаем процент вопросительных знаков
            q_count = out.count("?") + out.count("？")
            if q_count / len(out) > 0.4:
                is_broken = True
            
            # Проверяем на повторяющиеся паттерны (например, " ! ! ! ! ")
            # Если 5 раз подряд повторяется один и тот же чанк
            if re.search(r"(.{2,})\1{5,}", out):
                is_broken = True

        if is_broken:
            print(f"[DUAL][OCR] Loop detected (attempt {attempt+1}/{max_retries+1}): {out[:50]}...")
            
            if attempt < max_retries:
                # МЕНЯЕМ ПАРАМЕТРЫ ДЛЯ СЛЕДУЮЩЕЙ ПОПЫТКИ
                current_temp += 0.3      # Повышаем температуру (креативность), чтобы выйти из цикла
                current_penalty += 0.3   # Сильно повышаем штраф за повторы
                current_seed += 777      # Меняем зерно
                continue                 # Пробуем еще раз
            else:
                # Если попытки кончились — возвращаем пустую строку, чтобы не засорять экран мусором
                print("[DUAL][OCR] All retries failed. Ignoring garbage.")
                return ""

        lower = out.lower()
        # --- Фильтр галлюцинаций OCR (описание вместо текста) ---
        hallucination_markers = _BAD_PHRASES.get("ocr", [])
        if any(m in lower for m in hallucination_markers):
            print(f"[DUAL][OCR] hallucination detected (json filter), ignoring")
            return ""

        # === ЧИСТКА МУСОРА ОТ QWEN-2B ===
        # 1. Убираем строки, где только точка, запятая или палка (.\n, ,\n)
        # Qwen-2B любит ставить точку на новой строке вместо пробела.
        out = re.sub(r'\n\s*[\.\,\-\_]\s*\n', '\n', out)

        # 2. Схлопываем гигантские отступы (3 и более энтеров -> 2 энтера)
        # Чтобы текст не улетал за экран.
        out = re.sub(r'\n{2,}', '\n', out)
        # ================================

        # Успех
        print(f"[DUAL][OCR] {len(out)} chars in {dt:.0f} ms, text = {repr(out[:200])}")
        return out
        
    return ""

def translate_en_to_ru_text(en_text: str, cfg: LLMConfig) -> str:
    """Вызов на сервер перевода: EN -> RU (текст -> текст)."""
    if not en_text or not en_text.strip():
        return ""

    # --- ФИЛЬТР ПУНКТУАЦИИ (Если букв нет, возвращаем как есть) ---
    if not re.search(r'[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', en_text):
        # Если в строке есть хоть какие-то знаки, возвращаем их вместо пустой строки
        if en_text.strip():
            return en_text.strip()
        return ""
    
    # === WIKI MODE: Простой перевод без контекста и лора ===
    if getattr(cfg, "mode", "game") == "wiki":
        system = "You are a professional translator. Translate the text to Russian. Output ONLY the translation."
        user_content = (
            f"Translate the following text to Russian:\n\n{en_text}\n\n"
            "Output ONLY the Russian translation.\n"
            "Do NOT add notes, explanations, or advice."
        )
        # Используем базовые настройки, но без сложной логики
        # (код отправки запроса ниже общий, просто подменили system/user_content)
        # Но нужно пропустить логику split_name_and_body
        name_line = None
        target_ru_name_for_cleaning = None
    else:
        # === GAME MODE: Стандартная логика ===
        system = _ensure_tr_system(cfg)
        # 1. Пробуем разделить Имя и Тело красиво
        if not cfg.disable_name_split:
            name_line, body = _split_name_and_body(en_text or "", cfg.source_lang)
        else:
            name_line, body = None, en_text
        if not body:
            body = en_text or ""

        # --- FIX: Если тело состоит только из тишины/пунктуации — не переводим ---
        if not re.sub(r"[\s.…?？!！]+", "", body) and body.strip():
             return body

        # 2. АВАРИЙНАЯ ПРОВЕРКА ПОЛА (Если разделение не сработало)
        # Если имя не выделилось (name_line is None), проверяем, не начинается ли текст с имени персонажа
        forced_gender = None
        target_ru_name_for_cleaning = None

        if not name_line:
            # Ищем первое слово-кандидат
            first_word_match = re.match(r"^([a-zA-Z0-9_\-]+)", body)
            if first_word_match:
                fw = first_word_match.group(1)
                cn = _canon_name(fw)
                with _LORE_DB_LOCK:
                    # Проверяем в базе
                    char_info = None
                    if cn in _CHAR_DB:
                        char_info = _CHAR_DB[cn]
                    # Если не нашли, пробуем найти вхождение любой длинной фразы из базы в начале строки
                    # (для имен с пробелами, типа "Mosin Nagant")
                    else:
                        for k_cn, v_info in _CHAR_DB.items():
                            if _canon_name(body).startswith(k_cn):
                                char_info = v_info
                                break
                    
                    if char_info:
                        forced_gender = char_info.get("gender")
                        # target_ru_name_for_cleaning = char_info.get("ru") # ОТКЛЮЧАЕМ ЧИСТКУ ДЛЯ НАРРАТИВА
                        # print(f"[DUAL] Gender inferred from merged text: {forced_gender}")

        # Если имя выделилось штатно — берем данные оттуда
        if name_line:
            cn = _canon_name(name_line)
            with _LORE_DB_LOCK:
                if cn in _CHAR_DB:
                    target_ru_name_for_cleaning = _CHAR_DB[cn]["ru"]

        # 3. Собираем контекст и определяем собеседника
        if _MEMORY_ENABLED:
            ctx_block = _build_dialog_context_block(max_chars=min(2048, cfg.max_ctx_chars))
        else:
            ctx_block = ""

        # --- ОПРЕДЕЛЕНИЕ СОБЕСЕДНИКА (Addressee) ---
        addressee_gender = None
        addressee_name = None
        
        if _MEMORY_ENABLED:
            with _HISTORY_LOCK:
                if _HISTORY_EN:
                    last_line = _HISTORY_EN[-1]
                    # Извлекаем имя из последней строки истории "Name: Body"
                    h_match = re.match(r"^(.*?):\s", last_line)
                    if h_match:
                        last_speaker = h_match.group(1)
                        # Если текущий спикер - это НЕ последний спикер, то скорее всего он говорит последнему
                        if not name_line or _canon_name(name_line) != _canon_name(last_speaker):
                            addressee_name = last_speaker
                            a_info = _get_char_info_from_name(addressee_name)
                            if a_info:
                                addressee_gender = a_info.get("gender")

        lore_snippet, gender_tag = _build_lore_snippet_for_text(name_line, body)
        
        # Если штатный тег пола пуст, но мы нашли его "аварийно" — используем аварийный
        final_gender = gender_tag or forced_gender

        # --- ИНСТРУКЦИЯ ДЛЯ ГРАММАТИЧЕСКОГО РОДА ---
        special_instruction = ""
        
        # Если текст в скобках
        if body.strip().startswith("(") and body.strip().endswith(")"):
            special_instruction += "The text is a narration/action in parentheses. Translate it as description without adding a speaker name.\n"
        
        # ПРИНУДИТЕЛЬНОЕ УКАЗАНИЕ ПОЛА ГОВОРЯЩЕГО
        if final_gender == "F":
            special_instruction += "SPEAKER is FEMALE. Use feminine grammatical gender for 'I' (e.g., 'я поняла', 'я сама').\n"
        elif final_gender == "M":
            special_instruction += "SPEAKER is MALE. Use masculine grammatical gender for 'I' (e.g., 'я понял', 'я сам').\n"

        # ПРИНУДИТЕЛЬНОЕ УКАЗАНИЕ ПОЛА СОБЕСЕДНИКА (Для обращений на 'Ты')
        if addressee_gender == "F":
            special_instruction += f"ADDRESSEE (the person being spoken to) is FEMALE. Use feminine grammatical gender for 'you/thou' (e.g., 'ты пришла', 'ты была').\n"
        elif addressee_gender == "M":
            special_instruction += f"ADDRESSEE (the person being spoken to) is MALE. Use masculine grammatical gender for 'you/thou' (e.g., 'ты пришел', 'ты был').\n"
        elif addressee_name:
             # Если пол не знаем, но знаем имя - просто намекаем
             special_instruction += f"The speaker is talking to {addressee_name}. Ensure grammatical agreement.\n"

        # Определяем название языка для промпта
        lang_map = {
            "en": "English",
            "ja": "Japanese",
            "ch_sim": "Chinese",
            "ko": "Korean"
        }
        src_lang_name = lang_map.get(cfg.source_lang, "English")

        dynamic_system_parts = []
        if ctx_block:
            dynamic_system_parts.append(ctx_block.strip())
        if lore_snippet:
            dynamic_system_parts.append(lore_snippet.strip())

        if name_line:
            tag = ""
            if final_gender in ("F", "M"):
                tag = f"[{final_gender}] "
            name_for_prompt = f"{tag}{name_line}"
            dynamic_system_parts.append(f"<speaker>\n{name_for_prompt}\n</speaker>")
            
        if special_instruction:
            dynamic_system_parts.append(f"<instructions>\n{special_instruction.strip()}\n</instructions>")

        dynamic_system_content = "\n\n".join(dynamic_system_parts).strip()

        if dynamic_system_content:
            system += "\n\n" + dynamic_system_content
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": body}
        ]

    url = cfg.server.rstrip("/") + "/v1/chat/completions"

    # Dynamic max_tokens to prevent runaway hallucinations if the model's template is broken
    # For translation, we rarely need more than 4-5x the source length in tokens.
    estimated_max = max(150, len(body) * 5)
    max_out = min(int(cfg.max_tokens), estimated_max)

    # === ЦИКЛ ПОПЫТОК (RETRY LOOP) ===
    current_temp = cfg.temp
    current_seed = int(cfg.seed)
    max_retries = 1 

    for attempt in range(max_retries + 1):
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": current_temp,
            "top_p": cfg.top_p,
            "top_k": int(cfg.top_k),
            "repeat_penalty": float(cfg.repeat_penalty),
            "max_tokens": max_out,
            "add_generation_prompt": True,
            "slot_id": cfg.slot_id,
            "seed": current_seed,
            "cache_prompt": cfg.use_prompt_cache,
            "stop": [
                "<|im_end|>", "<|im_start|>", "</s>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "[/INST]",
                "\n<instructions", "\n<context", "\n<lore", "\n<speaker"
            ],
        }
        
        t0 = time.perf_counter()
        try:
            r = _SESSION.post(url, json=payload, timeout=cfg.timeout_s)
        except Exception as e:
            print(f"[DUAL][TR] connection error: {e}")
            return ""

        dt = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            print("[DUAL][TR] http", r.status_code, "in", f"{dt:.0f} ms", (r.text or "")[:180])
            return ""
            
        js = r.json()
        out = (js.get("choices", [{}])[0]
                  .get("message", {})
                  .get("content") or "").strip()
                  
        print(f"[DUAL][TR-RAW] Attempt {attempt} output: {repr(out)}")

        # Clean special tokens (like <|im_end|>, </s>, etc.)
        out = _RE_SPECIAL_TOKENS.sub("", out).strip()

        out = _strip_leading_gender_tag(out)
        # print(f"[DUAL][TR-CLEANED] After gender tag strip: {repr(out)}")
        
        # Intro cleaner
        intro_match = re.match(r"^.*?(?:перевод|перевести|translation|translate).*?:\s*(.*)", out, re.IGNORECASE | re.DOTALL)
        if intro_match:
            clean_part = intro_match.group(1).strip()
            if clean_part:
                out = clean_part
                
        # Clean possible XML tags if the model hallucinates them
        out = re.sub(r"</?(?:text|translation|ru|output)>", "", out, flags=re.IGNORECASE).strip()

        # Wrapper cleaner
        def _strip_wrapper(text_ru: str, text_en: str) -> str:
            s_ru = text_ru.strip(); s_en = text_en.strip()
            wrappers = [("**", "**"), ("*", "*"), ('"', '"'), ("'", "'"), ("“", "”"), ("«", "»")]
            for start, end in wrappers:
                if s_ru.startswith(start) and s_ru.endswith(end) and len(s_ru) >= len(start)+len(end):
                    if not (s_en.startswith(start) and s_en.endswith(end)):
                        return s_ru[len(start):-len(end)].strip()
            return s_ru

        for _ in range(2):
            new_out = _strip_wrapper(out, en_text)
            if new_out == out: break
            out = new_out
        
        # --- Проверка на "(затуп)" ---
        # Если модель вернула сигнал о том, что не может перевести, пробуем еще раз с другими параметрами
        if "(затуп)" in out:
            print(f"[DUAL][TR] '(затуп)' detected on attempt {attempt+1}, retrying...")
            if attempt < max_retries:
                current_temp = min(1.0, current_temp + 0.25) # Повышаем "креативность"
                current_seed += 222 # Меняем зерно
                continue # Переходим к следующей попытке
            else:
                print("[DUAL][TR] All retries for '(затуп)' failed. Returning empty.")
                return "" # Все попытки провалены

        # --- Name Echo Cleaner (УЛУЧШЕННЫЙ) ---
        # 1. Сначала чистим Русское имя (если оно есть в базе лора)
        if target_ru_name_for_cleaning:
            esc_name = re.escape(target_ru_name_for_cleaning)
            # Режем "Доберман," "Доберман:" "Доберман - " и даже "Доберман " (просто пробел)
            pattern = r"^[\s\W]*" + esc_name + r"[\s:,\.\-\n]+"
            match_echo = re.match(pattern, out, re.IGNORECASE)
            if match_echo:
                print(f"[DUAL][TR] Name echo detected ('{match_echo.group(0).strip()}'). Stripping.")
                out = out[match_echo.end():].strip()
        
        # 2. Затем чистим Английское имя (на случай, если модель не перевела его или его нет в базе)
        if name_line:
            esc_en_name = re.escape(name_line)
            pattern_en = r"^[\s\W]*" + esc_en_name + r"[\s:,\.\-\n]+"
            match_echo_en = re.match(pattern_en, out, re.IGNORECASE)
            if match_echo_en:
                print(f"[DUAL][TR] English Name echo detected ('{match_echo_en.group(0).strip()}'). Stripping.")
                out = out[match_echo_en.end():].strip()
        
        # print(f"[DUAL][TR-CLEANED] After name echo strip: {repr(out)}")
        # Дополнительная чистка для "???", если модель повторила их в начале перевода
        if name_line and re.fullmatch(r"[?？]+", name_line.strip()):
            match_mystery = re.match(r"^[\s]*[?？]+[\s:,\.\-\n]+", out)
            if match_mystery:
                out = out[match_mystery.end():].strip()
        
        # 3. Защита от утечки истории (если модель начала повторять формат "EN >>> RU")
        if "  >>>  " in out:
            print(f"[DUAL][TR] History format leak detected (>>>). Taking only the last part.")
            parts = out.split("  >>>  ")
            # Берем последнюю часть, которая скорее всего и есть нужный перевод
            out = parts[-1].strip()

        lower = out.lower()
        # print(f"[DUAL][TR-CLEANED] After mystery strip: {repr(out)}")
        # 1. Фильтр галлюцинаций JSON
        prompt_echo_snippets = _BAD_PHRASES.get("translation", [])
        if any(s in lower for s in prompt_echo_snippets):
            print(f"[DUAL][TR] prompt echo detected (json), ignoring")
            return ""

        if _normalize_homoglyphs(out.strip()) == _normalize_homoglyphs(en_text.strip()):
             print(f"[DUAL][TR] Output equals Input (homoglyph echo), retrying...")
             if attempt < max_retries:
                 current_temp = min(1.0, current_temp + 0.2) # Делаем модель "креативнее"
                 current_seed += 333
                 continue
             else:
                 return "" # Лучше ничего не показать, чем английский поверх английского

        # Б. Нет кириллицы (English only)
        # FIX: Улучшенный анти-эхо фильтр: если есть буквы, но нет кириллицы, или смешанный язык.
        # Если в тексте есть буквы, но нет НИ ОДНОЙ русской буквы
        has_letters = re.search(r'[a-zA-Zа-яА-Я]', out)
        has_cyrillic = re.search(r'[а-яА-ЯёЁ]', out)
        
        # Проверка на "чистый английский" (модель не перевела)
        if has_letters and not has_cyrillic and not re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', out): # И не CJK
            print(f"[DUAL][TR] No Cyrillic chars found (lazy english). Cleaned output was: {repr(out)} -> retrying...")
            if attempt < max_retries:
                current_temp = min(0.9, current_temp + 0.15)
                current_seed += 444
                continue
            else:
                return ""
        
        # Проверка на CJK (модель перевела не на тот язык)
        has_cjk = re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', out)
        
        if has_cjk and not has_cyrillic:
             print(f"[DUAL][TR] CJK chars detected in output (wrong language), retrying...")
             if attempt < max_retries:
                 current_temp = min(0.9, current_temp + 0.15)
                 current_seed += 888
                 continue
             else:
                 return ""
        # print(f"[DUAL][TR-CLEANED] After language check: {repr(out)}")
        
        if re.match(r"^\d+\)", out.strip()):
             print(f"[DUAL][TR] Context leak detected (starts with 'N)'), retrying...")
             if attempt < max_retries:
                 current_temp = min(1.0, current_temp + 0.15) # Повышаем креативность, чтобы сбить шаблон
                 current_seed += 666
                 continue
             else:
                 return ""
        
        # FIX: Улучшенный анти-эхо фильтр: Фильтр "Полу-перевода"
        # Если есть кириллица, но также есть 3+ английских слова подряд (признак смешанного языка)
        if re.search(r'[а-яА-Я]', out) and re.search(r'[a-zA-Z]{2,}\s+[a-zA-Z]{2,}\s+[a-zA-Z]{2,}', out):
             if attempt < max_retries:
                 current_temp = min(0.7, current_temp + 0.1)
                 current_seed += 999
                 continue
        
        # FIX: Улучшенный анти-эхо фильтр: Фильтр "Самозванца" (резервный)
        # Если модель сама добавила "Имя: " в начало, хотя его не было в EN
        rp_match = re.match(r"^[\*\_]*([а-яА-ЯёЁa-zA-Z0-9_\- ]{2,20})[\*\_]*:\s+(.*)", out)
        if rp_match:
            found_label_ru = rp_match.group(1)
            clean_text = rp_match.group(2)
            if name_line:
                out = clean_text
            else:
                en_has_colon = re.match(r"^[\*\_]*([a-zA-Z0-9_\- ]{2,20})[\*\_]*:\s+", en_text)
                if en_has_colon: pass 
                else:
                    if attempt < max_retries:
                         current_temp = min(0.5, current_temp + 0.05)
                         current_seed += 555
                         continue
                    else:
                        out = clean_text

        global _LAST_EN_TEXT, _LAST_RU_TEXT

        # === ЖЕСТКИЙ АНТИ-ЭХО ФИЛЬТР ===
        if out and out == _LAST_RU_TEXT and en_text.strip().lower() != _LAST_EN_TEXT:
            print(f"[DUAL][TR-WARN] LLM Echo detected! English changed but RU is identical: '{out}'")
            if attempt < max_retries:
                current_temp = min(0.8, current_temp + 0.15)
                current_seed += 999
                continue
            else:
                out = "" # Сбрасываем перевод, чтобы не плодить галлюцинации на экране

        # 2. Защита от вывода истории
        is_leak = False
        if len(en_text) > 5 and len(out) > 100 and len(out) > len(en_text) * 4.0:
            is_leak = True
            
        if not is_leak:
            print(f"[DUAL][TR] {len(out)} chars in {dt:.0f} ms, text = {repr(out[:200])}")
            # Сохраняем успешный перевод в память для проверки следующей фразы
            _LAST_EN_TEXT = en_text.strip().lower()
            _LAST_RU_TEXT = out
            return out
        
        if attempt < max_retries:
            current_temp = min(0.8, current_temp + 0.2)
            current_seed += 123456
            continue
            
    with _HISTORY_LOCK:
        _HISTORY_EN.clear()
    try:
        with open(_HISTORY_LOG_PATH, "w", encoding="utf-8") as f: f.write("")
    except: pass
    return ""

def commit_history_manually(en_text: str, ru_text: str = ""):
    """Вручную добавляем фразу в историю (вызывается из оверлея ПОСЛЕ стабилизации)."""
    # Если память выключена или текст пустой — выходим
    if not _MEMORY_ENABLED or not en_text:
        return
    
    # Используем ту же логику разделения (Имя: Текст), что и при переводе
    name, body = _split_name_and_body(en_text)
    
    # Пишем в лог
    _add_dialog_history(name, body, ru_text)