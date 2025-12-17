# llm_adapter_dual.py — отдельный адаптер под ДВА сервера: OCR и Перевод
# Требует: llama.cpp server с vision (OCR) и text (перевод).
# Порты/модели раздельные.
import os
import re
import time, threading
from dataclasses import dataclass
from typing import Optional
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
_HISTORY_MAX = 10
_HISTORY_EN: list[str] = []
_HISTORY_LOG_PATH = os.path.join(os.path.dirname(__file__), "dialog_history.txt")
_MEMORY_ENABLED = True

# ============== LORE DB (персонажи/имена) ================
_LORE_DB_LOCK = threading.Lock()
_LORE_DB_INIT = False
_CHAR_DB: dict[str, dict] = {}  # canon_en_name -> {en, ru, gender, raw}

def _add_dialog_history(name: Optional[str], body: str) -> None:
    """Сохраняем последнюю реплику вида 'Name: text' (EN) в небольшой буфер."""
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


def _build_dialog_context_block(max_chars: int = 2000) -> str:
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
        "RECENT CONTEXT (do NOT translate, for context only):\n"
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
    """Канонизация имени: режем мусор, приводим к нижнему регистру."""
    s = (s or "").strip()
    # убираем ведущие точки/многоточия
    s = re.sub(r"^[.…\s]+", "", s)
    # убираем содержимое скобок
    s = re.sub(r"\(.*?\)", "", s)
    # всё, что не буквы/цифры, превращаем в пробел
    s = re.sub(r"[^0-9a-zA-Zа-яА-Я]+", " ", s)
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


def _get_char_info_from_name(name_line: Optional[str]) -> Optional[dict]:
    """Находим персонажа по строке имени (из EN)."""
    if not name_line:
        return None
    cname = _canon_name(name_line)
    if not cname:
        return None
    with _LORE_DB_LOCK:
        return _CHAR_DB.get(cname)


def _build_lore_snippet_for_text(
    name_line: Optional[str], body: str, max_items: int = 5
) -> tuple[str, Optional[str]]:
    """
    Для текущей реплики собираем краткий LORE-сниппет + пол говорящего.
    Возвращает (snippet_text, gender_tag), где gender_tag = 'F'/'M'/None.
    """
    snippet_lines: list[str] = []
    speaker_gender: Optional[str] = None

    # 1) Сам спикер
    speaker_info = _get_char_info_from_name(name_line)
    if speaker_info:
        g = speaker_info.get("gender")
        if g == "F":
            g_desc = "female"
        elif g == "M":
            g_desc = "male"
        else:
            g_desc = None

        if g_desc:
            snippet_lines.append(
                f"- {speaker_info['en']} -> {speaker_info['ru']} ({g_desc})"
            )
        else:
            snippet_lines.append(
                f"- {speaker_info['en']} -> {speaker_info['ru']}"
            )

        speaker_gender = g

    # 2) Другие имена/термины в тексте
    tokens = re.findall(r"[0-9a-zA-Zа-яА-Я]+", body or "")
    seen: set[str] = set()

    for tok in tokens:
        cname = _canon_name(tok)
        if not cname or cname in seen:
            continue
        seen.add(cname)

        with _LORE_DB_LOCK:
            info = _CHAR_DB.get(cname)
        if not info:
            continue
        if speaker_info and info is speaker_info:
            continue

        g = info.get("gender")
        if g == "F":
            g_desc = "female"
        elif g == "M":
            g_desc = "male"
        else:
            g_desc = None

        if g_desc:
            snippet_lines.append(f"- {info['en']} -> {info['ru']} ({g_desc})")
        else:
            snippet_lines.append(f"- {info['en']} -> {info['ru']}")

        if len(snippet_lines) >= max_items:
            break

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
    # system:
    system_override: Optional[str] = None
    # контекстные файлы:
    lore_path: Optional[str] = None
    phrasebook_path: Optional[str] = None
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
def _read_text(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _build_tr_system(lore_text: str, phrasebook_text: str) -> str:
    return (
        "You are a professional game localizer.\n"
        "OUTPUT\n"
        "- Russian only (Cyrillic). No English, no comments.\n"
        "- Keep the SAME number of sentences and line breaks as in the source.\n"
        "FIDELITY & STYLE\n"
        "- Translate ONLY the provided English text; natural, idiomatic Russian.\n"
        "- Preserve meaning; you MAY change word order/phraseology for fluency.\n"
        "LORE & NAMES\n"
        "- Use the provided LORE SNIPPET (if any) as authoritative for names/titles/factions/terms.\n"
        "- Respect speaker gender where it’s obvious.\n"
        "CONSISTENCY\n"
        "- If the exact same English text repeats, return EXACTLY the same Russian as before.\n"
        "PHRASEBOOK (apply conservatively, exact matches only):\n"
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
            sys_txt = _build_tr_system(lore, pb)
        _TR_SYSTEM_CACHED = sys_txt
        _LORE_INIT = True
        print(f"[DUAL] translator system ready: lore={len(lore)} chars, pb={len(pb)} chars")
        return sys_txt

def reset_tr_system_cache():
    global _LORE_INIT, _TR_SYSTEM_CACHED
    with _LORE_LOCK:
        _LORE_INIT = False
        _TR_SYSTEM_CACHED = ""

def _split_name_and_body(en_text: str) -> tuple[Optional[str], str]:
    """
    Делим сырой OCR-текст на:
    - name_line: первая строка, если это похоже на имя/неймплейт;
    - body: остальной текст (диалог/система).

    НИЧЕГО не обрезаем в output — это только подготовка входа для LLM.
    """
    if not en_text:
        return None, ""

    # разбиваем на строки и обрезаем пустые сверху/снизу
    lines = [ln.rstrip("\r") for ln in en_text.splitlines()]
    lines = [ln for ln in lines]  # копия
    # убираем leading/trailing пустые
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None, ""

    raw_first = lines[0].strip()
    # убираем ведущие точки/многоточия и пробелы перед кандидатом имени
    first = re.sub(r"^[.…\s]+", "", raw_first)
    rest_lines = lines[1:]

    # если всего одна непустая строка — считаем, что это сразу текст
    if not rest_lines:
        return None, "\n".join(lines)

    # эвристика: имя/роль
    #  - короткое (<= 32 символов)
    #  - до 3 слов
    #  - нет явной "конца предложения" (.!?;:)
    #  - не пустое
    if (
        first
        and len(first) <= 32
        and len(first.split()) <= 3
        and not any(ch in first for ch in ".!?;:")
    ):
        body = "\n".join(rest_lines).lstrip("\n")
        return first, body

    # иначе считаем, что имя отдельно не выделено
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
    """Вызов на OCR-сервер: картинка -> английский текст (без перевода)."""
    url = cfg.server.rstrip("/") + "/v1/chat/completions"
    system_txt = cfg.system_override or _OCR_SYSTEM_DEFAULT
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_txt},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{region_png_b64}", "detail": "high"}}
            ]},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repeat_penalty": 1.0,
        "max_tokens": min(cfg.max_tokens, 1024),
        "add_generation_prompt": True,
        "slot_id": cfg.slot_id,
        "seed": int(cfg.seed),
    }
    t0 = time.perf_counter()
    r = _SESSION.post(url, json=payload, timeout=cfg.timeout_s)
    dt = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        print("[DUAL][OCR] http", r.status_code, "in", f"{dt:.0f} ms", (r.text or "")[:180])
        return ""
    js = r.json()
    out = (js.get("choices", [{}])[0]
              .get("message", {})
              .get("content") or "").strip()

    lower = out.lower()

    # --- Фильтр галлюцинаций OCR (описание вместо текста) ---
    hallucination_markers = _BAD_PHRASES.get("ocr", [])
    if any(m in lower for m in hallucination_markers):
        print(f"[DUAL][OCR] hallucination detected (json filter), ignoring; raw = {repr(out[:200])}")
        return ""

    # Нормальный случай: отдаём распознанный EN
    print(f"[DUAL][OCR] {len(out)} chars in {dt:.0f} ms, text = {repr(out[:200])}")
    return out

def translate_en_to_ru_text(en_text: str, cfg: LLMConfig) -> str:
    """Вызов на сервер перевода: EN -> RU (текст -> текст)."""
    system = _ensure_tr_system(cfg)
    url = cfg.server.rstrip("/") + "/v1/chat/completions"

    # --- Отделяем имя от тела диалога и готовим контекст ---
    name_line, body = _split_name_and_body(en_text or "")
    if not body:
        body = en_text or ""

    # небольшой блок с последними репликами (EN)
    if _MEMORY_ENABLED:
        ctx_block = _build_dialog_context_block(max_chars=min(2048, cfg.max_ctx_chars))
    else:
        ctx_block = ""

    # маленький сниппет из LORE + пол спикера (F/M)
    lore_snippet, gender_tag = _build_lore_snippet_for_text(name_line, body)

    # подсказка про гендерные теги
    gender_hint = (
        "[F] before a name = female speaker.\n"
        "[M] before a name = male speaker.\n"
        "Examples:\n"
        "[F] Sier: I'm tired. -> Сьер: Я устала.\n"
        "[M] Lonnie: I'm tired. -> Лонни: Я устал.\n\n"
    )

    # Формируем user-сообщение: CONTEXT + LORE_SNIPPET + NAME + TEXT
    if name_line:
        tag = ""
        if gender_tag in ("F", "M"):
            tag = f"[{gender_tag}] "
        name_for_prompt = f"{tag}{name_line}"

        user_content = (
            "You are given recent dialogue CONTEXT, a small LORE SNIPPET and one CURRENT line "
            "to translate from English to Russian.\n"
            "LORE SNIPPET is for understanding names and gender only — do NOT translate it.\n"
            "Use [F]/[M] tags on the NAME line to choose correct Russian gender endings.\n\n"
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
        user_content = (
            "You are given recent dialogue CONTEXT and a small LORE SNIPPET (if present) "
            "and one English text to translate into Russian.\n"
            "LORE SNIPPET is for understanding only — do NOT translate it.\n\n"
            + (lore_snippet or "")
            + (ctx_block or "")
            + "TEXT TO TRANSLATE (output Russian only, keep the same line breaks as in this section):\n"
            f"{body}"
        )

    max_out = max(64, min(512, int(len(en_text) * 2.2) + 64))
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": cfg.temp,
        "top_p": cfg.top_p,
        "top_k": int(cfg.top_k),
        "repeat_penalty": float(cfg.repeat_penalty),
        "max_tokens": min(cfg.max_tokens, max_out),
        "add_generation_prompt": True,
        "slot_id": cfg.slot_id,
        "seed": int(cfg.seed),
    }
    t0 = time.perf_counter()
    r = _SESSION.post(url, json=payload, timeout=cfg.timeout_s)
    dt = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        print("[DUAL][TR] http", r.status_code, "in", f"{dt:.0f} ms", (r.text or "")[:180])
        return ""
    js = r.json()
    out = (js.get("choices", [{}])[0]
              .get("message", {})
              .get("content") or "").strip()

    out = _strip_leading_gender_tag(out)
    lower = out.lower()

    # 1. Фильтр по JSON (bad_phrases)
    prompt_echo_snippets = _BAD_PHRASES.get("translation", [])
    if any(s in lower for s in prompt_echo_snippets):
        print(f"[DUAL][TR] prompt echo detected (json), ignoring; len={len(out)}")
        return ""

    # --- ВСТАВКА: ЗАЩИТА ОТ ВЫВОДА ИСТОРИИ (Length Guard) ---
    # Если перевод длиннее 100 символов И при этом в 4 раза длиннее оригинала — это подозрительно.
    # (Коэффициент 4.0 выбран с запасом, русский текст обычно длиннее английского на 20-30%, но не в 4 раза).
    if len(en_text) > 5 and len(out) > 100 and len(out) > len(en_text) * 4.0:
        print(f"[DUAL][TR] Output too long (Context Leak?), ignoring. EN={len(en_text)}, RU={len(out)}")
        return ""
    # --------------------------------------------------------

    # успешный перевод — обновляем диалоговую память
    if _MEMORY_ENABLED:
        _add_dialog_history(name_line, body)

    print(f"[DUAL][TR] {len(out)} chars in {dt:.0f} ms, text = {repr(out[:200])}")
    return out