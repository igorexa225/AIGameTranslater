# llm_adapter.py — визуальный адаптер для llama.cpp (две картинки → перевод RU)
# Требует llama.cpp server c vision (Qwen*-VL + mmproj).

import os, time, json, base64, threading, re
from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "Qwen3-VL-8B-Instruct"

# ===================== Конфиг LLM =========================

@dataclass
class LLMConfig:
    enabled: bool = True
    server: Optional[str] = "http://127.0.0.1:8080"
    model: str = DEFAULT_MODEL

    # поведение запроса
    temp: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 15000
    timeout_s: float = 30.0
    budget_ms: int = 4000
    top_k: int = 0
    repeat_penalty: float = 1.0
    seed: int = 0
    slot_id: int = 0

    # лор
    lore_path: str = "assets/game_bible_exilium.txt"
    phrasebook_path: str | None = None
    max_ctx_chars: int = 30000

    # прогрев
    use_prompt_cache: bool = True

    # кастомизация system-подсказки из UI
    system_override: Optional[str] = None


# ================== Лор + System prompt ===================

_LORE_TEXT = ""
_SYSTEM_PROMPT = ""
_LORE_INIT = False
_LORE_LOCK = threading.Lock()
_TM_LOCK = threading.Lock()
_TM_EN2RU: dict[str, str] = {}
_TM_ORDER: list[str] = []
_TM_MAX = 512  # максимум записей в кэше

_EN_WS_RX = re.compile(r"\s+")
def _canon_en(s: str) -> str:
    """Грубое нормирование английского текста для ключа TM."""
    s = (s or "").replace("\ufeff", "")
    s = s.lower()
    s = _EN_WS_RX.sub(" ", s)
    return s.strip()

def reset_lore_cache():
    """Принудительно пересобрать лор/системный промпт при следующем init_lore_once()."""
    global _LORE_TEXT, _SYSTEM_PROMPT, _LORE_INIT
    _LORE_TEXT = ""
    _SYSTEM_PROMPT = ""
    _LORE_INIT = False

def _read_file(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _build_system_prompt(lore_text: str) -> str:
    return (
        "You are a professional game localizer.\n"
        "GOAL\n"
        "- Read short in-game dialogue / UI text from images and translate it to Russian.\n"
        "OUTPUT FORMAT\n"
        "- When the user asks for JSON, you MUST respond with a SINGLE JSON object only.\n"
        '- For this project the JSON shape is usually {"en": "...", "ru": "..."}. '
        '"en" = original English text, "ru" = Russian translation.\n'
        "- Do not add comments or extra fields.\n"
        "RUSSIAN TRANSLATION RULES\n"
        "- Russian only (Cyrillic) inside the \"ru\" field. No English there.\n"
        "- Keep the SAME number of sentences and line breaks as in the source.\n"
        "- Do not add quotation marks unless they exist in the source.\n"
        "SCOPE & FIDELITY\n"
        "- Translate ONLY the visible text from the image; do not invent missing content.\n"
        "- Preserve meaning and tone. Write NATURAL, IDIOMATIC Russian: allow rewording, "
        "change word order, and choose Russian collocations if it improves readability "
        "without adding information.\n"
        "- Keep placeholders/tags exactly as is.\n"
        "LORE & NAMES\n"
        "- Use LORE as the single source of truth for names, titles, factions, and terms.\n"
        "- Keep canonical forms from LORE; decline personal names correctly in Russian.\n"
        "- Determine the SPEAKER’S GENDER from LORE or the name and use correct "
        "feminine/masculine forms.\n"
        "CONSISTENCY\n"
        "- If the exact same English text repeats, return EXACTLY the same Russian "
        "translation as before.\n"
        "- If only minor changes between current and previous frames, keep terminology "
        "and style consistent.\n"
        "=== PHRASEBOOK START ===\n"
        "{{PHRASEBOOK}}\n"
        "=== PHRASEBOOK END ===\n"
        "=== LORE START ===\n"
        f"{(lore_text or '(empty)')[:30000]}\n"
        "=== LORE END ==="
    )

def init_lore_once(cfg: LLMConfig) -> bool:
    """Читает лор ОДИН РАЗ и собирает system. Учитывает system_override, если он не пустой."""
    global _LORE_TEXT, _SYSTEM_PROMPT, _LORE_INIT
    if _LORE_INIT:
        return True
    try:
        with _LORE_LOCK:
            if _LORE_INIT:
                return True
            text = _read_file(cfg.lore_path).strip()
            _LORE_TEXT = (text or "")[:cfg.max_ctx_chars]

            # NEW: читаем phrasebook (если указан)
            pb_text = _read_file(getattr(cfg, "phrasebook_path", "")).strip() if getattr(cfg, "phrasebook_path", None) else ""

            if cfg.system_override:
                sys_override = cfg.system_override
                # подстановка и LORE, и PHRASEBOOK
                sys_prompt = sys_override.replace("{{LORE}}", _LORE_TEXT).replace("{{PHRASEBOOK}}", pb_text)
                print("[LLM] system: override + injected LORE/PHRASEBOOK")
            else:
                sys_prompt = _build_system_prompt(_LORE_TEXT)
                # подставляем PHRASEBOOK в билдер
                sys_prompt = sys_prompt.replace("{{PHRASEBOOK}}", pb_text)
                print("[LLM] system: default builder with lore + phrasebook")

            _SYSTEM_PROMPT = sys_prompt
            _LORE_INIT = True
            print(f"[LLM] lore loaded once: {len(_LORE_TEXT)} chars from {cfg.lore_path or '(none)'}; phrasebook: {len(pb_text)} chars")
            return True
    except Exception as e:
        print("[LLM] lore init error:", e)
        _LORE_TEXT = ""
        _SYSTEM_PROMPT = _build_system_prompt("")
        _LORE_INIT = True
        return False

def get_system_preview(cfg: LLMConfig) -> str:
    lore_text = (_read_file(cfg.lore_path) or "")[: cfg.max_ctx_chars]
    pb_text = _read_file(getattr(cfg, "phrasebook_path", "")).strip() if getattr(cfg, "phrasebook_path", None) else ""
    base = (cfg.system_override or _build_system_prompt(lore_text))
    return base.replace("{{LORE}}", lore_text).replace("{{PHRASEBOOK}}", pb_text)


# ===================== Прогрев кэша =======================

def preload_prompt_cache(cfg: LLMConfig) -> bool:
    """Записывает system в кеш слота сервера llama.cpp (если сервер поддерживает cache_prompt)."""
    if not (cfg.enabled and cfg.server and cfg.use_prompt_cache):
        return False

    init_lore_once(cfg)
    system = _SYSTEM_PROMPT
    url = cfg.server.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}],
        "cache_prompt": True,
        "add_generation_prompt": False,
        "max_tokens": 16,
        "slot_id": cfg.slot_id,
    }
    try:
        import requests, time as _t
        for _ in range(60):
            r = requests.post(url, json=payload, timeout=max(1.0, float(cfg.timeout_s)))
            if r.status_code == 200:
                print(f"[LLM] prompt cache warmed (slot_id={cfg.slot_id})")
                return True
            if r.status_code == 503 and "Loading model" in (r.text or ""):
                _t.sleep(0.5); continue
            print("[LLM] warmup HTTP:", r.status_code, (r.text or "")[:200])
            return False
        print("[LLM] warmup: model still loading, giving up")
        return False
    except Exception as e:
        print("[LLM] warmup error:", e)
        return False


# =========== ЕДИНСТВЕННЫЙ ВЫЗОВ: картинка → перевод =======

def vision_translate_from_images(region_png_b64: str,
                                full_png_b64: Optional[str],
                                cfg: LLMConfig,
                                tgt_lang: str = "ru") -> str:
    """
    Принимает: base64 PNG обрезанной области (region_png_b64),
    опционально полный кадр окна (full_png_b64 — сейчас не используется).
    Возвращает строку перевода на русский.
    """
    if not (cfg.enabled and cfg.server and region_png_b64):
        return ""

    init_lore_once(cfg)
    system = _SYSTEM_PROMPT
    url = cfg.server.rstrip("/") + "/v1/chat/completions"

    def _content_variant(kind: str):
        txt = {
            "type": "text",
            "text": (
                "This image is a cropped in-game dialogue or UI box.\n"
                "1) Read ALL clearly visible ENGLISH text EXACTLY as it appears "
                "(preserve line breaks).\n"
                "2) Translate it into natural Russian, using the system instructions and LORE.\n"
                "3) Respond with a SINGLE JSON object ONLY, in one line:\n"
                '   {\"en\": \"<original English text>\", \"ru\": \"<Russian translation>\"}\n'
                "- Use \\n inside strings for line breaks.\n"
                "- Do NOT add comments, extra fields, or any text before/after the JSON."
            ),
        }
        if kind == "A":
            return [txt, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{region_png_b64}", "detail": "high"}}]
        if kind == "B":
            return [txt, {"type": "image_url", "image_url": f"data:image/png;base64,{region_png_b64}"}]
        # old/compat
        return [txt, {"type": "image", "image_url": {"url": f"data:image/png;base64,{region_png_b64}"}}]

    import requests
    t0_all = time.perf_counter()

    for kind in ("A", "B", "C"):
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": _content_variant(kind)}
            ],
            "temperature": cfg.temp,
            "top_p": cfg.top_p,
            "top_k": int(getattr(cfg, "top_k", 0)),
            "repeat_penalty": float(getattr(cfg, "repeat_penalty", 1.0)),
            "seed": int(getattr(cfg, "seed", 0)),
            "max_tokens": min(15000, cfg.max_tokens),
            "add_generation_prompt": True,
            "slot_id": cfg.slot_id,
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=float(cfg.timeout_s))
            dt = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                print(f"[LLM] vision kind={kind} http={r.status_code} in {dt:.0f} ms")
                continue
            js = r.json()
            out = (js.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

            if not out:
                print(f"[LLM] vision empty kind={kind} in {dt:.0f} ms")
                continue

            # иногда модели заворачивают в ```json ... ```
            txt = out.strip()
            if txt.startswith("```"):
                # убираем первую строку с ```...
                first_nl = txt.find("\n")
                if first_nl != -1:
                    txt = txt[first_nl + 1:]
                if txt.endswith("```"):
                    txt = txt[:-3]
                txt = txt.strip()

            en_text: str = ""
            ru_text: str = ""

            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    en_text = str(obj.get("en", "") or "").strip()
                    ru_text = str(obj.get("ru", "") or "").strip()
                else:
                    # на всякий случай: если это массив/что-то ещё
                    ru_text = str(obj).strip()
            except Exception:
                # не получилось распарсить как JSON — считаем, что это просто RU как раньше
                ru_text = out.strip()

            # --- Translation Memory по EN ---
            if en_text:
                key = _canon_en(en_text)
                with _TM_LOCK:
                    cached = _TM_EN2RU.get(key)
                    if cached:
                        # уже переводили этот английский — возвращаем канонический RU
                        ru_final = cached
                    else:
                        ru_final = ru_text
                        if ru_final:
                            _TM_EN2RU[key] = ru_final
                            _TM_ORDER.append(key)
                            # LRU-чистка
                            if len(_TM_ORDER) > _TM_MAX:
                                old_key = _TM_ORDER.pop(0)
                                _TM_EN2RU.pop(old_key, None)
            else:
                ru_final = ru_text

            print(
                f"[LLM] vision ok kind={kind} "
                f"EN={len(en_text)} chars, RU={len(ru_final)} chars in {dt:.0f} ms "
                f"(total {(time.perf_counter()-t0_all)*1000:.0f} ms)"
            )

            return ru_final
            print(f"[LLM] vision empty kind={kind} in {dt:.0f} ms")
        except Exception as e:
            print(f"[LLM] vision error kind={kind}:", e)

    return ""
