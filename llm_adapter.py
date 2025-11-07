# llm_adapter.py — визуальный адаптер для llama.cpp (две картинки → перевод RU)
# Требует llama.cpp server c vision (Qwen*-VL + mmproj).

import os, time, json, base64, threading
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
        "HARD RULES:\n"
        "- Translate into Russian, except for words that are not translated (see the game_bible file).\n"
        "- Don't change the meaning, do not invent characters,\n"
        "- Use the hints in the game_bible file.\n"
        "- Adapt the text for literary reading,\n"
        "- Refer to the rules of the Russian language for translation, \n"
        "  factions or items. Do not replace common nouns with lore entities.\n"
        "- Use LORE to resolve proper names, titles, and the SPEAKER'S GENDER.\n"
        "- If a female speaker is known from LORE or the name in the line, use feminine forms in Russian.\n"
        "- Keep the SAME number of sentences and line breaks as in the source.\n"
        "- Use the glossary exactly when a source token matches its key.\n"
        "- Use the current frame and the previous frame to ensure the most accurate translation in the context of the dialogue.\n"
        "- If the exact same source text appears repeatedly, reuse the same Russian translation verbatim.\n"
        "- When translating a dialogue, take into account who is speaking using the lore. If a female character is speaking, change the endings of the words to the appropriate ones.\n"
        "- For personal names from the lore adapt gender/case (e.g. “Mayling wrote” → “Мейлинг написала”).\n"
        "- Read ONLY text from Image.\n"
        "- If the text is not complete, translate it anyway.\n"
        "- Output translation ONLY (no source, no comments).\n"
        "- If you receive an image with the same text, DON'T translate it.\n"
        "- If the text for translation is not visible, display the word ((ЗАТУП...)).\n"
        "Context (optional):\n"
        "Use game lore as authoritative canon for names/factions/terms.\n\n"
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

            if cfg.system_override:
                sys_override = cfg.system_override
                if "{{LORE}}" in sys_override:
                    sys_prompt = sys_override.replace("{{LORE}}", _LORE_TEXT)
                    print("[LLM] system: override + LORE injected")
                else:
                    sys_prompt = sys_override
                    print("[LLM] system: override (no lore)")
            else:
                sys_prompt = _build_system_prompt(_LORE_TEXT)
                print("[LLM] system: default builder with lore")

            _SYSTEM_PROMPT = sys_prompt
            _LORE_INIT = True
            print(f"[LLM] lore loaded once: {len(_LORE_TEXT)} chars from {cfg.lore_path or '(none)'}")
            return True
    except Exception as e:
        print("[LLM] lore init error:", e)
        _LORE_TEXT = ""
        _SYSTEM_PROMPT = _build_system_prompt("")
        _LORE_INIT = True
        return False

def get_system_preview(cfg: LLMConfig) -> str:
    """Возвращает текст активного system-prompt (с учётом override и текущего лора)."""
    lore_text = (_read_file(cfg.lore_path) or "")[: cfg.max_ctx_chars]
    if cfg.system_override:
        return cfg.system_override.replace("{{LORE}}", lore_text)
    return _build_system_prompt(lore_text)


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
        txt = {"type": "text",
               "text": "This image is a cropped dialogue box. Read ONLY the visible text and return the Russian translation. Use LORE for names and gender. Output translation ONLY."}
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
            out = (js.get("choices",[{}])[0].get("message",{}).get("content") or "").strip()
            if out:
                print(f"[LLM] vision ok kind={kind} {len(out)} chars in {dt:.0f} ms "
                      f"(total {(time.perf_counter()-t0_all)*1000:.0f} ms)")
                return out
            print(f"[LLM] vision empty kind={kind} in {dt:.0f} ms")
        except Exception as e:
            print(f"[LLM] vision error kind={kind}:", e)

    return ""
