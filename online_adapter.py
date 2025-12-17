try:
    from deep_translator import GoogleTranslator
    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False

def is_available() -> bool:
    """Проверяет, установлена ли библиотека."""
    return _HAS_GOOGLE

def translate_text(text: str, source: str = 'auto', target: str = 'ru') -> str:
    """
    Отправляет текст в Google Translate.
    Возвращает строку с переводом или текст ошибки.
    """
    if not text or not text.strip():
        return ""

    if not _HAS_GOOGLE:
        return "[Error] Библиотека 'deep-translator' не найдена.\nУстановите её: pip install deep-translator"

    try:
        # GoogleTranslator сам разбивает длинный текст на чанки (до 5000 символов)
        translator = GoogleTranslator(source=source, target=target)
        result = translator.translate(text)
        return result if result else ""
    except Exception as e:
        print(f"[ONLINE] Translation error: {e}")
        return f"[Error] Google Translate: {e}"