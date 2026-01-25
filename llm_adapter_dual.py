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
_HISTORY_MAX = 25  # Увеличиваем память до 25 фраз
_HISTORY_EN: list[str] = []
_HISTORY_LOG_PATH = os.path.join(os.path.dirname(__file__), "dialog_history.txt")
_MEMORY_ENABLED = True

# ============== LORE DB (персонажи/имена) ================
_LORE_DB_LOCK = threading.Lock()
_LORE_DB_INIT = False
_CHAR_DB: dict[str, dict] = {}  # canon_en_name -> {en, ru, gender, raw}

_RE_MYSTERY = re.compile(r"[?？]+")
_RE_LEADING_DOTS = re.compile(r"^[.…\s]+")
_RE_PAREN_STD = re.compile(r"\(.*?\)")
_RE_PAREN_CJK = re.compile(r"（.*?）")
_RE_NON_WORD = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_RE_SPECIAL_TOKENS = re.compile(r"(?:<\|.*?\|>|&lt;\|.*?\|?&gt;|<s>|</s>|&lt;/s&gt;|<end_of_turn>|<start_of_turn>)", flags=re.IGNORECASE)

def _add_dialog_history(name: str | None, body: str, ru_body: str = "") -> None:
    """Сохраняем реплику вида 'Name: text >>> translation'."""
    if not body:
        return
    body = " ".join((body or "").split())
    if not body:
        return
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
        # не дублируем подряд одинаковые строки
        if _HISTORY_EN and _HISTORY_EN[-1] == line:
            return
        _HISTORY_EN.append(line)
        if len(_HISTORY_EN) > _HISTORY_MAX:
            # храним только последние N
            del _HISTORY_EN[0 : len(_HISTORY_EN) - _HISTORY_MAX]
        try:
            with open(_HISTORY_LOG_PATH, "w", encoding="utf-8") as f:
                for ln in _HISTORY_EN:
                    f.write(ln + "\n")
        except Exception as e:
            print("[CTX] history log write error:", e)


def _build_dialog_context_block(max_chars: int = 5000) -> str:
    """Формируем текстовый блок с последними репликами для промпта."""
    with _HISTORY_LOCK:
        if not _HISTORY_EN:
            return ""
        collected: list[str] = []
        total = 0
        for ln in reversed(_HISTORY_EN):
            extra = len(ln) + 4
            if collected and total + extra > max_chars:
                break
            collected.append(ln)
            total += extra
        collected.reverse()

    numbered = "\n".join(f"{i+1}) {ln}" for i, ln in enumerate(collected))
    return (
        "RECENT CONTEXT (Original >>> Translation):\n"
        f"{numbered}\n"
        "END OF CONTEXT.\n\n"
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
            if not t or t.startswith("#") or t.startswith("//"):
                continue

            # Формат: NameEN -> NameRU | пол: женский/мужской
            m = re.match(
                r"^(?P<en>[^#=\-:|]+?)\s*->\s*(?P<ru>[^|#]+?)\s*(?:\|\s*пол\s*:\s*(?P<gender>[^|#]+))?\s*$",
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
        "LORE SNIPPET (context only, do NOT translate):\n"
        + "\n".join(snippet_lines)
        + "\n"
    )
    return snippet, speaker_gender

# ============== Общая конфигурация слотов ================
@dataclass
class LLMConfig:
    server: str = "http://127.0.0.1:8080"
    model: str = ""
    timeout_s: float = 30.0
    max_tokens: int = 4096
    temp: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repeat_penalty: float = 1.0
    seed: int = 0
    slot_id: int = 0
    source_lang: str = "en"  # en, ja, ko, ch_sim
    # system:
    system_override: str | None = None
    # контекстные файлы:
    lore_path: str | None = None
    phrasebook_path: str | None = None
    max_ctx_chars: int = 30000
    # prompt-cache:
    use_prompt_cache: bool = True

# ======= OCR: минимальный system для считывания текста =====
_OCR_SYSTEM_DEFAULT = (
    "You are an OCR tool.\n"
    "Return EXACTLY the visible text from the image. No translation. No comments.\n"
    "Preserve original line breaks."
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
    if lang_code in ("ja", "ch_sim", "ko", "en"):
        # === РЕЖИМ: Минимализм (Visual Novel Mode) ===
        style_block = (
            "STYLE & TONE\n"
            "- Style: Literary, natural Russian translation.\n"
            "- Brackets: Keep 「...」 and 『...』 distinctions if present.\n"
            "- CRITICAL: Do NOT merge separate sentences.\n"
            "- NOTE: Source text may omit subjects. Infer them from context.\n"
        )
    else:
        # Резерв
        style_block = (
            "STYLE & STRUCTURE\n"
            "- Style: Natural Russian.\n"
        )

    # 3. Собираем итоговый промпт
    return (
        f"You are a professional literary translator from {src_lang} to Russian.\n"
        "Your goal is high-quality localization, preserving the original tone and atmosphere.\n\n"
        "INPUT DATA\n"
        "- LORE SNIPPET: authoritative dictionary for names/terms.\n"
        "- CONTEXT: previous lines (do not translate).\n"
        "- NAME: speaker name (do not translate).\n"
        "- TEXT TO TRANSLATE: the content you must process.\n\n"
        "OUTPUT RULES\n"
        "- Output ONLY Russian text (Cyrillic). No comments.\n"
        f"- Translate ONLY the content under 'TEXT TO TRANSLATE' from {src_lang}.\n"
        "- Do NOT output the NAME line or [F]/[M] tags.\n\n"
        f"{style_block}\n"
        "NAMES & LORE\n"
        "- Use the LORE SNIPPET/PHRASEBOOK as absolute truth.\n"
        "- Use [F]/[M] tags to determine grammatical gender (verb endings).\n\n"
        "CONSISTENCY\n"
        "- If the text repeats, return EXACTLY the same translation.\n"
        "PHRASEBOOK (Exact matches only):\n"
        f"{phrasebook_text[:15000]}\n"
    )

_LORE_INIT = False
_LORE_LOCK = threading.Lock()
_TR_SYSTEM_CACHED = ""
def _ensure_tr_system(cfg: LLMConfig) -> str:
    global _LORE_INIT, _TR_SYSTEM_CACHED
    if _LORE_INIT and _TR_SYSTEM_CACHED:
        return _TR_SYSTEM_CACHED
    with _LORE_LOCK:
        lore = _read_text(cfg.lore_path)[:cfg.max_ctx_chars] if cfg.lore_path else ""
        pb   = _read_text(cfg.phrasebook_path)[:cfg.max_ctx_chars] if cfg.phrasebook_path else ""

        if lore:
            _load_lore_db_from_text(lore)  # ← грузим персонажей/пол в память

        if cfg.system_override:
            sys_txt = (cfg.system_override
                       .replace("{{LORE}}", lore)
                       .replace("{{PHRASEBOOK}}", pb))
        else:
            sys_txt = _build_tr_system(lore, pb, cfg.source_lang)
        _TR_SYSTEM_CACHED = sys_txt
        _LORE_INIT = True
        print(f"[DUAL] translator system ready: lore={len(lore)} chars, pb={len(pb)} chars")
        return sys_txt

def reset_tr_system_cache():
    global _LORE_INIT, _TR_SYSTEM_CACHED
    with _LORE_LOCK:
        _LORE_INIT = False
        _TR_SYSTEM_CACHED = ""

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

    raw_first = lines[0].strip()
    first = re.sub(r"^[.…\s]+", "", raw_first)
    
    # Если строк несколько — проверяем первую как имя
    if len(lines) > 1:
        # Разрешаем ??? как имя, даже если там есть знак вопроса
        is_mystery = bool(re.fullmatch(r"[?？]+", first))
        
        if is_mystery or (len(first) <= 32 and len(first.split()) <= 3 and not any(ch in first for ch in ".!?;:")):
            return first, "\n".join(lines[1:]).lstrip("\n")
    
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
        if canon_line.startswith(cname):
            orig_name_en = _CHAR_DB[cname]["en"]
            
            # Ищем имя в оригинальной строке (регистронезависимо)
            match = re.match(re.escape(orig_name_en), full_line, re.IGNORECASE)
            
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
                    found_name = match.group(0)
                    # Очищаем остаток от разделителей/пробелов в начале
                    rest_body = re.sub(r"^[:\-\–\s]+", "", remainder)
                    
                    if len(lines) > 1:
                        rest_body += "\n" + "\n".join(lines[1:])
                        
                    return found_name, rest_body

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
    
    for attempt in range(max_retries + 1):
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_txt},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{region_png_b64}", "detail": "high"}}
                ]},
            ],
            "temperature": current_temp,
            "top_p": 1.0,
            "top_k": 40,
            "repeat_penalty": current_penalty,
            "max_tokens": 256,  # Жесткий лимит, чтобы не ждать 1.5 секунды при сбое
            "add_generation_prompt": True,
            "slot_id": cfg.slot_id,
            "seed": current_seed,
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

    # Проверка на наличие букв (Latin + CJK + Hangul)
    if not re.search(r'[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', en_text):
        print(f"[DUAL][TR] Ignored non-text input (no letters): {repr(en_text)}")
        return ""
    
    system = _ensure_tr_system(cfg)
    url = cfg.server.rstrip("/") + "/v1/chat/completions"

    # 1. Пробуем разделить Имя и Тело красиво
    name_line, body = _split_name_and_body(en_text or "", cfg.source_lang)
    if not body:
        body = en_text or ""

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

    # 3. Собираем контекст
    if _MEMORY_ENABLED:
        ctx_block = _build_dialog_context_block(max_chars=min(2048, cfg.max_ctx_chars))
    else:
        ctx_block = ""

    lore_snippet, gender_tag = _build_lore_snippet_for_text(name_line, body)
    
    # Если штатный тег пола пуст, но мы нашли его "аварийно" — используем аварийный
    final_gender = gender_tag or forced_gender

    gender_hint = (
        "[F] before a name = female speaker.\n"
        "[M] before a name = male speaker.\n"
    )

    # --- ИНСТРУКЦИЯ ДЛЯ ГРАММАТИЧЕСКОГО РОДА ---
    special_instruction = ""
    
    # Если текст в скобках
    if body.strip().startswith("(") and body.strip().endswith(")"):
        special_instruction += "\nIMPORTANT: The text is a narration/action in parentheses. Translate it as description. DO NOT add a speaker name!\n"
    
    # ПРИНУДИТЕЛЬНОЕ УКАЗАНИЕ ПОЛА (Самое важное!)
    if final_gender == "F":
        special_instruction += "\nIMPORTANT: SPEAKER IS FEMALE. Use feminine grammatical gender (я сделала, я должна).\n"
    elif final_gender == "M":
        special_instruction += "\nIMPORTANT: SPEAKER IS MALE. Use masculine grammatical gender (я сделал, я должен).\n"

    # Определяем название языка для промпта
    lang_map = {
        "en": "English",
        "ja": "Japanese",
        "ch_sim": "Chinese",
        "ko": "Korean"
    }
    src_lang_name = lang_map.get(cfg.source_lang, "English")

    # Формируем промпт
    if name_line:
        tag = ""
        if final_gender in ("F", "M"):
            tag = f"[{final_gender}] "
        name_for_prompt = f"{tag}{name_line}"

        user_content = (
            "You are given recent dialogue CONTEXT, a small LORE SNIPPET and one CURRENT line "
            f"to translate from {src_lang_name} to Russian.\n"
            "LORE SNIPPET is for understanding names and gender only — do NOT translate it.\n"
            "Use [F]/[M] tags on the NAME line to choose correct Russian gender endings.\n"
            f"{special_instruction}\n"
            + gender_hint
            + (lore_snippet or "")
            + (ctx_block or "")
            + "CURRENT LINE:\n"
              "NAME (context only, do NOT translate or output this line):\n"
            f"{name_for_prompt}\n\n"
            "TEXT TO TRANSLATE (output Russian only, keep the same line breaks as in this section):\n"
            f"{body}"
        )
    else:
        # Случай, когда имя внутри текста (как у тебя сейчас)
        user_content = (
            "You are given recent dialogue CONTEXT and a small LORE SNIPPET (if present) "
            f"and one {src_lang_name} text to translate into Russian.\n"
            "LORE SNIPPET is for understanding only — do NOT translate it.\n"
            f"{special_instruction}\n" # <--- Сюда попадет "SPEAKER IS FEMALE"
            + (lore_snippet or "")
            + (ctx_block or "")
            + "TEXT TO TRANSLATE (output Russian only, keep the same line breaks as in this section):\n"
            f"{body}"
        )

    max_out = max(64, min(512, int(len(en_text) * 2.2) + 64))

    # === ЦИКЛ ПОПЫТОК (RETRY LOOP) ===
    current_temp = cfg.temp
    current_seed = int(cfg.seed)
    max_retries = 1 

    for attempt in range(max_retries + 1):
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": current_temp,
            "top_p": cfg.top_p,
            "top_k": int(cfg.top_k),
            "repeat_penalty": float(cfg.repeat_penalty),
            "max_tokens": min(cfg.max_tokens, max_out),
            "add_generation_prompt": True,
            "slot_id": cfg.slot_id,
            "seed": current_seed,
            "cache_prompt": cfg.use_prompt_cache,
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

        # Clean special tokens (like <|im_end|>, </s>, etc.)
        out = _RE_SPECIAL_TOKENS.sub("", out).strip()

        out = _strip_leading_gender_tag(out)
        
        # Intro cleaner
        intro_match = re.match(r"^.*?(?:перевод|перевести|translation|translate).*?:\s*(.*)", out, re.IGNORECASE | re.DOTALL)
        if intro_match:
            clean_part = intro_match.group(1).strip()
            if clean_part:
                out = clean_part

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
        
        # --- Name Echo Cleaner (УЛУЧШЕННЫЙ) ---
        if target_ru_name_for_cleaning:
            esc_name = re.escape(target_ru_name_for_cleaning)
            # Режем "Доберман," "Доберман:" "Доберман - " и даже "Доберман " (просто пробел)
            pattern = r"^[\s\W]*" + esc_name + r"[\s:,\.\-\n]+"
            match_echo = re.match(pattern, out, re.IGNORECASE)
            if match_echo:
                print(f"[DUAL][TR] Name echo detected ('{match_echo.group(0).strip()}'). Stripping.")
                out = out[match_echo.end():].strip()
        
        # Дополнительная чистка для "???", если модель повторила их в начале перевода
        if name_line and re.fullmatch(r"[?？]+", name_line.strip()):
            match_mystery = re.match(r"^[\s]*[?？]+[\s:,\.\-\n]+", out)
            if match_mystery:
                out = out[match_mystery.end():].strip()

        lower = out.lower()
        # 1. Фильтр галлюцинаций JSON
        prompt_echo_snippets = _BAD_PHRASES.get("translation", [])
        if any(s in lower for s in prompt_echo_snippets):
            print(f"[DUAL][TR] prompt echo detected (json), ignoring")
            return ""

        if out.strip().lower() == en_text.strip().lower():
             print(f"[DUAL][TR] Output equals Input (lazy echo), retrying...")
             if attempt < max_retries:
                 current_temp = min(1.0, current_temp + 0.2) # Делаем модель "креативнее"
                 current_seed += 333
                 continue
             else:
                 return "" # Лучше ничего не показать, чем английский поверх английского

        # Б. Нет кириллицы (English only)
        # Если в тексте есть буквы, но нет НИ ОДНОЙ русской буквы
        has_letters = re.search(r'[a-zA-Zа-яА-Я]', out)
        has_cyrillic = re.search(r'[а-яА-ЯёЁ]', out)
        
        if has_letters and not has_cyrillic:
             print(f"[DUAL][TR] No Cyrillic chars found (lazy english), retrying...")
             if attempt < max_retries:
                 current_temp = min(0.9, current_temp + 0.15)
                 current_seed += 444
                 continue
             else:
                 return ""
        
        has_cjk = re.search(r'[\u4e00-\u9fff]', out)
        if has_cjk and not has_cyrillic:
             print(f"[DUAL][TR] CJK chars detected in output (wrong language), retrying...")
             if attempt < max_retries:
                 current_temp = min(0.9, current_temp + 0.15)
                 current_seed += 888
                 continue
             else:
                 return ""
        
        if re.match(r"^\d+\)", out.strip()):
             print(f"[DUAL][TR] Context leak detected (starts with 'N)'), retrying...")
             if attempt < max_retries:
                 current_temp = min(1.0, current_temp + 0.15) # Повышаем креативность, чтобы сбить шаблон
                 current_seed += 666
                 continue
             else:
                 return ""
        
        # 1.5. Фильтр "Полу-перевода"
        if re.search(r'[а-яА-Я]', out) and re.search(r'[a-zA-Z]{2,}\s+[a-zA-Z]{2,}\s+[a-zA-Z]{2,}', out):
             if attempt < max_retries:
                 current_temp = min(0.7, current_temp + 0.1)
                 current_seed += 999
                 continue
        
        # 1.6. Фильтр "Самозванца" (резервный)
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

        # 2. Защита от вывода истории
        is_leak = False
        if len(en_text) > 5 and len(out) > 100 and len(out) > len(en_text) * 4.0:
            is_leak = True
            
        if not is_leak:
            print(f"[DUAL][TR] {len(out)} chars in {dt:.0f} ms, text = {repr(out[:200])}")
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