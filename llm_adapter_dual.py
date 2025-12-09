# llm_adapter_dual.py — отдельный адаптер под ДВА сервера: OCR и Перевод
# Требует: llama.cpp server с vision (OCR) и text (перевод).
# Порты/модели раздельные.

import time, threading
from dataclasses import dataclass
from typing import Optional

import requests

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
        "- Use LORE as authoritative canon for names/titles/factions/terms.\n"
        "- Respect speaker gender where it’s obvious.\n"
        "CONSISTENCY\n"
        "- If the exact same English text repeats, return EXACTLY the same Russian as before.\n"
        "PHRASEBOOK (apply conservatively, exact matches only):\n"
        f"{phrasebook_text[:15000]}\n"
        "=== LORE START ===\n"
        f"{lore_text[:30000]}\n"
        "=== LORE END ==="
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

    first = lines[0].strip()
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
    r = requests.post(url, json=payload, timeout=cfg.timeout_s)
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
    hallucination_markers = (
        "the following text is visible in the image",
        "the text is not visible in the image",
        "no text is visible in the image",
        "there is no readable text in the image",
        "no legible text is visible in the image",
        "The game is a classic, and it's a great way",
        "The final answer is:\nThe final answe",
    )

    if any(m in lower for m in hallucination_markers):
        # Модель описывает картинку, а не считывает текст — игнорируем
        print(f"[DUAL][OCR] hallucinated description, ignoring; raw = {repr(out[:200])}")
        return ""

    # Нормальный случай: отдаём распознанный EN
    print(f"[DUAL][OCR] {len(out)} chars in {dt:.0f} ms, text = {repr(out[:200])}")
    return out

def translate_en_to_ru_text(en_text: str, cfg: LLMConfig) -> str:
    """Вызов на сервер перевода: EN -> RU (текст -> текст)."""
    system = _ensure_tr_system(cfg)
    url = cfg.server.rstrip("/") + "/v1/chat/completions"

    # --- НОВОЕ: отделяем имя от тела диалога ---
    name_line, body = _split_name_and_body(en_text or "")
    if not body:
        body = en_text or ""

    # Формируем user-сообщение в явном формате NAME + TEXT
    if name_line:
        user_content = (
            "You are given a speaker NAME (context only) and English TEXT.\n"
            "Use NAME only for gender, tone and style. Never output NAME.\n\n"
            "NAME (context only, do NOT translate or output this line):\n"
            f"{name_line}\n\n"
            "TEXT TO TRANSLATE (output Russian only, keep the same line breaks as in this section):\n"
            f"{body}"
        )
    else:
        user_content = (
            "Translate the following English text into Russian. "
            "Keep the same line breaks. Do not add or remove lines.\n\n"
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
    r = requests.post(url, json=payload, timeout=cfg.timeout_s)
    dt = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        print("[DUAL][TR] http", r.status_code, "in", f"{dt:.0f} ms", (r.text or "")[:180])
        return ""
    js = r.json()
    out = (js.get("choices", [{}])[0]
              .get("message", {})
              .get("content") or "").strip()

    lower = out.lower()

    # --- Фильтр эха промпта / мета-ответов ---
    prompt_echo_snippets = (
        "вы получили имя говорящего",                   # русский кусок system-promt'а
        "переводите текст исключительно на русский язык",
        "you are given a speaker name",                 # английская версия
        "translate the following english text into russian",
        "```plaintext",                                 # часто в фантомных ответах
        "as an ai language model",                      # на случай типичных фраз
        "Эта игра – настоящий классик, и она отлично",
    )

    if any(s in lower for s in prompt_echo_snippets):
        print(f"[DUAL][TR] prompt/meta echo detected, ignoring; raw = {repr(out[:200])}")
        return ""

    print(f"[DUAL][TR] {len(out)} chars in {dt:.0f} ms, text = {repr(out[:200])}")
    return out