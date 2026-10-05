# fullscreen_overlay.py — Полностью переписанный модуль полноэкранного перевода
# Принцип: Максимальная изоляция от основной логики, стабильность, отсутствие вылетов.

import sys, time, ctypes, base64, re
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import cv2
from difflib import SequenceMatcher

from PySide6.QtCore import (Qt, QRect, QPoint, QThread, Signal, QObject, QTimer, QRunnable, QThreadPool,
                            QAbstractNativeEventFilter, QPropertyAnimation, QEasingCurve)
from PySide6.QtGui  import (QPainter, QColor, QFont, QKeySequence, QShortcut, QGuiApplication, QFontMetrics)
from PySide6.QtWidgets import (QWidget, QApplication)

import win32gui, win32ui, win32con, win32api
import ctypes.wintypes as wt

DEBUG_MODE = False  # Сохранение отладочных картинок (для отладки)

# --- Импорты адаптеров (для доступа к нейросетям) ---
import online_adapter
try:
    from llm_adapter_dual import extract_en_from_image   # нужен только режиму ONLINE
except Exception as e:
    print(f"[FULLSCREEN] llm_adapter_dual не загрузился: {e}")

# --- EasyOCR ---
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None
    _EASYOCR_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent

# ====================== Утилиты захвата и UI ======================

def _bgr_to_png_b64(img, max_side: int = 1400) -> str:
    """Конвертация BGR изображения в base64 PNG."""
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side/float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok: return ""
    return base64.b64encode(enc).decode("ascii")

# --- Windows Acrylic Blur ---
_HRESULT = getattr(wt, "HRESULT", ctypes.c_long)
class ACCENT_POLICY(ctypes.Structure):
    _fields_=[("AccentState",ctypes.c_int),("AccentFlags",ctypes.c_int),
              ("GradientColor",ctypes.c_uint),("AnimationId",ctypes.c_int)]
class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
    _fields_=[("Attribute",ctypes.c_int),("Data",ctypes.c_void_p),("SizeOfData",ctypes.c_size_t)]
user32 = ctypes.WinDLL("user32.dll")
SetWindowCompositionAttribute = user32.SetWindowCompositionAttribute
SetWindowCompositionAttribute.argtypes=[wt.HWND, ctypes.POINTER(WINDOWCOMPOSITIONATTRIBDATA)]
SetWindowCompositionAttribute.restype=_HRESULT
RegisterHotKey = user32.RegisterHotKey
UnregisterHotKey = user32.UnregisterHotKey

def enable_acrylic(hwnd:int, tint_abgr:int=0x40101010):
    policy = ACCENT_POLICY()
    policy.AccentState = 4 # ACCENT_ENABLE_ACRYLICBLURBEHIND
    policy.GradientColor = tint_abgr
    data = WINDOWCOMPOSITIONATTRIBDATA()
    data.Attribute = 19 # WCA_ACCENT_POLICY
    data.SizeOfData = ctypes.sizeof(policy)
    data.Data = ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p)
    try:
        SetWindowCompositionAttribute(wt.HWND(hwnd), ctypes.byref(data))
    except Exception as e:
        print("[ACRYLIC] error:", e)

def _grab_full_window(hwnd):
    """Захватывает всё окно целиком."""
    try:
        # Получаем размеры
        try:
            left, top, right, bottom = win32gui.DwmGetWindowAttribute(hwnd, 9)
        except:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        w, h = max(1, right - left), max(1, bottom - top)

        # Захват через PrintWindow
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(bmp)
        
        # PW_RENDERFULLCONTENT = 2 (или 3 для Windows 8.1+)
        ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
        
        bmp_info = bmp.GetInfo()
        bmp_str = bmp.GetBitmapBits(True)
        
        img = np.frombuffer(bmp_str, dtype=np.uint8)
        img.shape = (bmp_info['bmHeight'], bmp_info['bmWidth'], 4)
        
        # Очистка
        win32gui.DeleteObject(bmp.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        # RGBA -> BGR
        return img[:, :, :3].copy(), (left, top, w, h)
    except Exception as e:
        # print(f"[GRAB] Error: {e}")
        return None, None

# ====================== EasyOCR Singleton ======================
_EASYOCR_READER = None
_CURRENT_LANG = None

def get_easyocr(lang="en"):
    global _EASYOCR_READER, _CURRENT_LANG
    if not _EASYOCR_AVAILABLE: return None
    
    if _EASYOCR_READER and _CURRENT_LANG == lang:
        return _EASYOCR_READER
        
    print(f"[FULLSCREEN] Initializing EasyOCR for {lang}...")
    langs = [lang]
    if lang != "en": langs.append("en")
    
    model_dir = BASE_DIR / "models" / "easyocr"
    try: model_dir.mkdir(parents=True, exist_ok=True)
    except: pass
    
    try:
        # Обычно сюда не попадаем: основной модуль передаёт свой общий ридер (ocr_reader_factory).
        try:
            import torch
            use_gpu = bool(torch.cuda.is_available())
        except Exception:
            use_gpu = False
        _EASYOCR_READER = easyocr.Reader(langs, gpu=use_gpu, model_storage_directory=str(model_dir))
        _CURRENT_LANG = lang
    except Exception as e:
        print(f"[FULLSCREEN] EasyOCR init failed: {e}")
        _EASYOCR_READER = None
    return _EASYOCR_READER

# ====================== UI: Панель текста ======================

class TextPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(None)
        # WindowTransparentForInput: окно «прозрачно» для мыши на уровне Windows. Одного
        # атрибута WA_TransparentForMouseEvents мало — он действует только внутри нашего
        # приложения, и панелька над кнопкой игры не давала по ней кликнуть.
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                            | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.parent_overlay = parent
        self.text = ""
        self.bg_mode = "blur"
        self.bg_alpha = 64
        self.font_family = "Segoe UI"
        self.font_size = 20
        self.font_bold = False
        self.box_bg_mode = "solid"
        self.box_bg_alpha = 180
        self.ocr_h = 0  # Запоминаем оригинальную высоту бокса
        self.ocr_w = 0  # Запоминаем оригинальную ширину бокса
        self.line_h = 0 # высота одной строки оригинала — по ней подбирается размер шрифта
        self._px, self._fit_w, self._fit_h, self._fit_key = 16, 0, 0, None
        self.generation = 0  # Версия текущего текста
        self._anim = None    # текущая анимация переезда

    def glide_to(self, x: int, y: int, ms: int):
        """Плавно переезжает в точку: текст в игре сдвинулся — перевод едет следом."""
        cur = self.pos()
        dist = abs(cur.x() - x) + abs(cur.y() - y)
        if self._anim is not None:
            self._anim.stop()
        # скрытую панельку и прыжок через пол-экрана не анимируем
        if not self.isVisible() or dist < 2 or dist > 900:
            self.move(x, y)
            return
        self._anim = QPropertyAnimation(self, b"pos", self)
        self._anim.setDuration(int(ms))
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.setStartValue(cur)
        self._anim.setEndValue(QPoint(x, y))
        self._anim.start()

    def apply_acrylic(self):
        hwnd = int(self.winId())
        if self.bg_mode == "blur":
            tint = (self.bg_alpha << 24) | 0x101010
            enable_acrylic(hwnd, tint)
        else:
            # Disable acrylic
            policy = ACCENT_POLICY(); policy.AccentState = 0
            data = WINDOWCOMPOSITIONATTRIBDATA(); data.Attribute = 19
            data.SizeOfData = ctypes.sizeof(policy)
            data.Data = ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p)
            SetWindowCompositionAttribute(wt.HWND(hwnd), ctypes.byref(data))

    PAD_X, PAD_Y = 6, 2

    def _font(self, px: int) -> QFont:
        f = QFont(self.font_family)
        f.setPixelSize(int(px))
        f.setBold(self.font_bold)
        return f

    def _fit(self):
        """Подбирает размер шрифта под оригинал: (размер в пикселях, ширина, высота панельки).
        Берём высоту строки оригинала; если перевод длиннее и не влезает в его рамку —
        уменьшаем шрифт, но не больше чем до 60%; не влезло и так — панелька растёт."""
        flags = Qt.TextWordWrap | Qt.AlignLeft
        if self.line_h > 0:
            base = max(11, min(200, int(self.line_h * 0.8)))
        else:
            base = max(11, int(self.font_size * 1.33))      # размера оригинала нет — берём из настроек
        min_px = max(11, int(base * 0.6))
        box_w = max(40, self.ocr_w)
        wrap_w = box_w - 2 * self.PAD_X
        one_line = self.line_h <= 0 or self.ocr_h < 1.6 * self.line_h

        for px in range(base, min_px - 1, -1):
            fm = QFontMetrics(self._font(px))
            if one_line:
                if fm.horizontalAdvance(self.text) <= wrap_w:
                    return px, box_w, max(self.ocr_h, fm.height() + 2 * self.PAD_Y)
            else:
                r = fm.boundingRect(0, 0, wrap_w, 10000, flags, self.text)
                if r.height() + 2 * self.PAD_Y <= self.ocr_h:
                    return px, box_w, self.ocr_h

        fm = QFontMetrics(self._font(min_px))
        if one_line:
            # одна строка: сначала расширяемся вправо (до 1000px), потом переносим
            box_w = min(1000, max(box_w, fm.horizontalAdvance(self.text) + 2 * self.PAD_X))
        r = fm.boundingRect(0, 0, box_w - 2 * self.PAD_X, 10000, flags, self.text)
        return min_px, box_w, max(self.ocr_h, r.height() + 2 * self.PAD_Y)

    def update_size(self):
        """Пересчитываем шрифт и размер панели под текущий перевод и рамку оригинала."""
        if not self.text: return
        key = (self.text, self.ocr_w, self.ocr_h, self.line_h, self.font_family, self.font_bold)
        if key != self._fit_key:
            self._fit_key = key
            self._px, self._fit_w, self._fit_h = self._fit()
        if self._fit_w != self.width() or self._fit_h != self.height():
            self.resize(self._fit_w, self._fit_h)
            self.apply_acrylic()

    def paintEvent(self, _e):
        if not self.text: return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setFont(self._font(self._px))
        rect = self.rect()

        # Подложка на всю рамку оригинала: перевод часто короче, и без неё из-под него
        # торчит хвост исходного текста ("Пока ничего.or now.")
        alpha = self.bg_alpha if self.bg_mode == "solid" else (self.box_bg_alpha if self.box_bg_mode == "solid" else 0)
        if alpha > 0:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, alpha))
            p.drawRoundedRect(rect, 4, 4)

        # Текст — по левому краю, по высоте посередине рамки, как стоял оригинал
        flags = Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignVCenter
        text_rect = rect.adjusted(self.PAD_X, self.PAD_Y, -self.PAD_X, -self.PAD_Y)
        p.setPen(QColor(0, 0, 0, 220))
        p.drawText(text_rect.translated(1, 1), flags, self.text)
        p.setPen(QColor(255, 255, 255, 255))
        p.drawText(text_rect, flags, self.text)

# ====================== Перевод одного блока ======================
#
# Каждый блок текста — свой запрос, и перевод показывается сразу, как пришёл.
# Модель получает картинку блока и сразу отдаёт перевод: читать и переводить двумя
# запросами (как в DUAL) здесь незачем — режим нужен, чтобы понимать происходящее,
# а не для качества.

import requests

_FS_SESSION = requests.Session()
_FS_LANG = {"en": "English", "ja": "Japanese", "ch_sim": "Chinese", "ko": "Korean", "ru": "Russian"}
_RE_FS_TAGS = re.compile(r"(?:<\|.*?\|>|</?s>|<end_of_turn>|<start_of_turn>)", re.I)


def translate_block_image(b64_png: str, server: str, lang: str = "en", timeout_s: float = 20.0) -> str:
    """Картинка одного блока -> русский перевод одним запросом. Пусто, если текста нет или сбой."""
    src = _FS_LANG.get(lang, "English")
    system = (
        f"You translate on-screen video game text from {src} to Russian.\n"
        "The image shows ONE piece of text cut from a game screen: a menu item, button, label, "
        "item description, hint or a line of dialogue.\n"
        "Reply with the Russian translation ONLY: no quotes, no notes, no original text.\n"
        "Translate EVERY word, including short labels in capitals (SALE, START, NEW).\n"
        "Leave unchanged only numbers and the keyboard key itself (ESC, Ctrl, Tab, a single letter); "
        "the word \"Key\" is translated: \"R Key\" -> \"Клавиша R\".\n"
        "If there is no readable text in the image, reply with a single dash: -"
    )
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}", "detail": "high"}}
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
        "cache_prompt": True,
        "stop": ["<|im_end|>", "<|im_start|>", "</s>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "[/INST]"],
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = _FS_SESSION.post(server.rstrip("/") + "/v1/chat/completions", json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code}: {(r.text or '')[:120]}")
    out = (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "")
    out = _RE_FS_TAGS.sub("", out).strip().strip('"«»“”').strip()
    return "" if out in ("-", "—", "–", "") else out


def worth_translating(text: str, lang: str = "en") -> bool:
    """Есть ли в блоке что переводить: цифры, значки и одиночные буквы модель не трогает."""
    t = text or ""
    if lang in ("ja", "ch_sim", "ko"):
        if re.search(r"[぀-ヿ一-鿿가-힯]", t):
            return True
    return len(re.findall(r"[A-Za-z]", t)) >= 2


class BlockTranslationTask(QRunnable):
    """Перевод одного блока. Запускается из очереди с приоритетом (крупные блоки — первыми)."""

    def __init__(self, worker, rid, gen, key, text, crop):
        super().__init__()
        self.worker, self.rid, self.gen, self.key, self.text, self.crop = worker, rid, gen, key, text, crop

    def run(self):
        wk = self.worker
        # Пока запрос ждал в очереди, блок мог исчезнуть или смениться — тогда он уже не нужен
        r = wk.active_regions.get(self.rid)
        if r is None or r['generation'] != self.gen:
            return
        ov = wk.overlay
        ru = ""
        t0 = time.perf_counter()
        try:
            mode = ov.work_mode_idx  # 0=SOLO, 1=DUAL, 2=ONLINE
            b64 = _bgr_to_png_b64(self.crop)
            if mode == 2:
                # ONLINE: модель только читает, переводит Google
                en = extract_en_from_image(b64, ov.ocr_cfg_dual) or self.text
                ru = online_adapter.translate_text(en) if en else ""
            else:
                cfg = ov.ocr_cfg_dual if mode == 1 else ov.llm_cfg
                ru = translate_block_image(b64, cfg.server, ov.source_lang, float(getattr(cfg, "timeout_s", 20.0)))
        except Exception as e:
            print(f"[FULLSCREEN] блок {self.rid}: перевод не удался: {e}")
        wk._task_done(self.rid, self.gen, self.key, ru, (time.perf_counter() - t0) * 1000.0)

# ====================== Hotkeys ======================

class FullscreenHotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay

    def nativeEventFilter(self, et, msgptr):
        if et != "windows_generic_MSG": return False, 0
        msg = wt.MSG.from_address(int(msgptr))
        if msg.message == 0x0312: # WM_HOTKEY
            hid = msg.wParam
            if hid == 101: self.overlay.quit_app()     # Pause
            elif hid == 102: self.overlay.toggle_pause() # Alt+-
            return True, 0
        return False, 0

# ====================== Слежение за блоками текста ======================
#
# EasyOCR здесь — только "глаза слежения": где на экране текст и что там примерно
# написано. Блок опознаётся по тексту, а не по координатам: тот же текст в другом месте —
# тот же блок, его перевод остаётся при нём, и панелька просто переезжает следом.

SAME_TEXT = 0.75        # похожесть, начиная с которой текст считается тем же (дрожание OCR)
DIFF_TEXT = 0.40        # ниже — текст точно другой, старый перевод прячем сразу
MISS_LIMIT = 2          # сколько кадров подряд может не находиться блок без перевода
LOST_KEEP_SEC = 1.5     # сколько секунд держим на экране перевод блока, который перестал находиться
CHANGE_HITS = 2         # сколько кадров подряд должен продержаться "немного другой" текст
RETRY_SEC = 15.0        # запрос завис дольше этого — считаем потерянным и шлём заново
RETRY_PAUSE = 3.0       # пауза перед повтором после неудачного ответа
MAX_TRIES = 3
MEMORY_MAX = 3000       # сколько переводов держать в памяти (оригинал -> перевод)
MOVE_DEADZONE_PX = 4    # рамки EasyOCR дрожат на пару пикселей — на такие сдвиги не реагируем
FORCE_OCR_SEC = 2.0     # даже на неподвижной картинке перечитываем экран не реже


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (s or "").lower())).strip()


def _text_sim(a: str, b: str) -> float:
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return 1.0 if na == nb else 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _overlap(a, b) -> float:
    """Доля БОЛЬШЕГО прямоугольника, накрытая пересечением (0..1): рамки должны быть
    сопоставимы по размеру, мелкая рамка внутри крупной «тем же местом» не считается."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return (iw * ih) / float(max(1, max(aw * ah, bw * bh)))


def group_lines(dets: list) -> list:
    """
    Склеивает куски, на которые EasyOCR разрезал один текст, и НЕ склеивает соседние пункты
    списков, меню и таблиц — это главное требование: близкий, но разный по смыслу текст
    должен остаться раздельным.

    1) По горизонтали: куски на одной строке с маленьким просветом (меньше 0.8 высоты) —
       это одна строка, разрезанная посередине. У кнопок и вкладок просвет больше.
    2) По вертикали: только перенос длинного абзаца — строки выровнены по левому краю, стоят
       вплотную, одного размера, и верхняя не короче нижней. Списки по центру, сетки кнопок
       и короткие пункты меню под это не подходят и остаются отдельными блоками.
    """
    def merged(a, b, sep=" "):
        x1, y1 = min(a['x'], b['x']), min(a['y'], b['y'])
        x2 = max(a['x'] + a['w'], b['x'] + b['w'])
        y2 = max(a['y'] + a['h'], b['y'] + b['h'])
        return {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1, 'text': a['text'] + sep + b['text'],
                'conf': min(a.get('conf', 1.0), b.get('conf', 1.0)),
                'line_h': a.get('line_h', a['h']), 'lines': a.get('lines', 1) + (b.get('lines', 1) if sep == "\n" else 0)}

    # --- 1. куски одной строки ---
    items = sorted(({**d, 'line_h': d['h'], 'lines': 1} for d in dets), key=lambda d: (d['y'] + d['h'] / 2.0, d['x']))
    lines = []
    for d in sorted(items, key=lambda d: d['x']):
        best = None
        for ln in lines:
            h = min(ln['h'], d['h'])
            same_row = abs((ln['y'] + ln['h'] / 2.0) - (d['y'] + d['h'] / 2.0)) < 0.5 * h
            similar = max(ln['h'], d['h']) <= 1.6 * h
            gap = d['x'] - (ln['x'] + ln['w'])
            if same_row and similar and -0.5 * h <= gap < 0.8 * h:
                best = ln
                break
        if best is None:
            lines.append(d)
        else:
            lines[lines.index(best)] = merged(best, d)

    # --- 2. перенос абзаца ---
    lines.sort(key=lambda d: (d['y'], d['x']))
    out, used = [], set()
    for i, a in enumerate(lines):
        if i in used:
            continue
        cur = a
        while True:
            lh = cur['line_h']
            nxt = None
            for j, b in enumerate(lines):
                if j in used or j == i or b is cur or b['y'] <= cur['y']:
                    continue
                gap = b['y'] - (cur['y'] + cur['h'])
                last_w = cur.get('last_w', cur['w'])
                d_left = abs(b['x'] - cur['x'])
                # у списка по центру совпадают середины строк, а не левые края — это не абзац
                d_mid = abs((b['x'] + b['w'] / 2.0) - (cur['x'] + last_w / 2.0))
                if (d_left < 0.3 * lh and not d_mid < d_left     # общий левый край, и это не центровка
                        and -0.3 * lh <= gap < 0.8 * lh          # строки вплотную
                        and max(b['h'], lh) <= 1.3 * min(b['h'], lh)   # один размер шрифта
                        and last_w >= 0.75 * b['w']              # верхняя строка не короче нижней
                        and last_w >= 6 * lh):                   # и достаточно длинная, чтобы быть перенесённой
                    nxt = j
                    break
            if nxt is None:
                break
            used.add(nxt)
            b = lines[nxt]
            cur = merged(cur, b, sep="\n")
            cur['line_h'] = lh
            cur['last_w'] = b['w']
        out.append(cur)
    for d in out:
        d['text'] = d['text'].replace("\n", " ")
    return out


def match_regions(regions: dict, dets: list):
    """
    Сопоставляет блоки прошлого кадра с найденными сейчас.
    regions: {rid: {'rect': (x, y, w, h), 'text': str, ...}}; dets: [{'x','y','w','h','text'}]
    Возвращает (same, changed, new, lost):
      same    — [(rid, di)] тот же текст (где бы он ни оказался);
      changed — [(rid, di)] на том же месте другой текст;
      new     — [di] новые блоки;  lost — [rid] не найденные в этом кадре.
    """
    def center(r):
        return r[0] + r[2] / 2.0, r[1] + r[3] / 2.0

    def drect(d):
        return (d['x'], d['y'], d['w'], d['h'])

    # 1) тот же текст; среди одинаковых надписей ("OK", "x1") пару выбираем по близости
    pairs = []
    for rid, r in regions.items():
        rc = center(r['rect'])
        for di, d in enumerate(dets):
            s = _text_sim(d['text'], r['text'])
            if s >= SAME_TEXT:
                dc = center(drect(d))
                pairs.append((((rc[0] - dc[0]) ** 2 + (rc[1] - dc[1]) ** 2) ** 0.5, -s, rid, di))
    pairs.sort()
    used_r, used_d, same = set(), set(), []
    for _dist, _s, rid, di in pairs:
        if rid in used_r or di in used_d:
            continue
        used_r.add(rid); used_d.add(di); same.append((rid, di))

    # 2) то же место, другой текст (сменилась реплика, счётчик, подпись)
    cand = []
    for rid, r in regions.items():
        if rid in used_r:
            continue
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            ov = _overlap(r['rect'], drect(d))
            if ov >= 0.3:
                cand.append((-ov, rid, di))
    cand.sort()
    changed = []
    for _ov, rid, di in cand:
        if rid in used_r or di in used_d:
            continue
        used_r.add(rid); used_d.add(di); changed.append((rid, di))

    new = [di for di in range(len(dets)) if di not in used_d]
    lost = [rid for rid in regions if rid not in used_r]
    return same, changed, new, lost


class FullscreenCaptureWorker(QThread):
    # Сигналы для управления GUI из потока (БЕЗОПАСНО). Координаты — экранные, логические.
    create_panel = Signal(int, int, int, int, int, int) # id, x, y, w, h, высота строки
    update_panel_pos = Signal(int, int, int, int, int, int)
    update_panel_text = Signal(int, str, int) # rid, text, generation
    update_panel_gen = Signal(int, int)       # rid, generation (сброс текста)
    delete_panel = Signal(int)

    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self._stop = False
        # {id: {'rect': (x,y,w,h) в пикселях кадра, 'text', 'generation', 'state', ...}}
        self.active_regions = {}
        self.next_id = 0
        self.pool = QThreadPool()
        # одновременно столько запросов, сколько слотов у сервера: лишние всё равно ждали бы там
        self.pool.setMaxThreadCount(max(1, int(getattr(overlay, "parallel", 2) or 2)))
        # Память переводов: меню, инвентарь и подсказки повторяются — второй раз перевод
        # показывается сразу, без модели. Ключ — нормализованный текст от EasyOCR.
        self.memory = {}
        self._mem_hits = 0
        self._req_ms = []
        self._last_sig = None       # уменьшенный кадр для проверки "изменилось ли что-то"
        self._last_dets = []
        self._last_ocr_t = 0.0
        self._stat_t = time.time()
        self._stat = {"frames": 0, "ocr": 0, "grab_ms": 0.0, "ocr_ms": 0.0}

    def stop(self):
        self._stop = True

    # ---------- захват и поиск текста ----------

    def _frame_changed(self, img) -> bool:
        """Дешёвая проверка по уменьшенному кадру, по клеткам: смена одной строки текста
        заметна в своей клетке, хотя в среднем по экрану теряется."""
        small = cv2.cvtColor(cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        prev, self._last_sig = self._last_sig, small
        if prev is None or prev.shape != small.shape:
            return True
        diff = cv2.absdiff(small, prev).astype(np.float32)
        tiles = diff.reshape(9, 10, 16, 10).mean(axis=(1, 3))     # клетки 10x10 пикселей
        return float(tiles.max()) > 3.0

    def _detect(self, img) -> list:
        factory = getattr(self.overlay, "ocr_reader_factory", None) or get_easyocr
        reader = factory(self.overlay.source_lang)
        if not reader:
            return None
        scale = 0.7
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        results = reader.readtext(small, paragraph=False)

        candidates = []
        for bbox, text, conf in results:
            if conf < 0.3 or not text.strip():
                continue
            # Фильтр мелочи
            if (bbox[2][0] - bbox[0][0]) < 15 or (bbox[2][1] - bbox[0][1]) < 10:
                continue
            tl, br = bbox[0], bbox[2]
            candidates.append({
                'x': int(tl[0] / scale), 'y': int(tl[1] / scale),
                'w': int((br[0] - tl[0]) / scale), 'h': int((br[1] - tl[1]) / scale),
                'text': text, 'conf': conf
            })
        # Фильтруем перекрытия (оставляем только мелкие/точные боксы), потом собираем
        # разрезанные строки и перенесённые абзацы обратно в целые блоки
        return group_lines(self._filter_overlapping_boxes(candidates))

    # ---------- основной цикл ----------

    def run(self):  # noqa: C901
        print("[FULLSCREEN] Worker started")
        while not self._stop:
            # Пауза
            if self.overlay.paused:
                time.sleep(0.2)
                continue

            t_cycle = time.perf_counter()
            hwnd = self.overlay.bound_hwnd
            if not hwnd or not win32gui.IsWindow(hwnd):
                time.sleep(0.2)
                continue

            # 1. Захват
            img, win_rect = _grab_full_window(hwnd)
            if img is None:
                time.sleep(0.2)
                continue
            t_grab = time.perf_counter()

            # 2. Поиск текста. Картинка не менялась — берём рамки прошлого кадра без OCR.
            try:
                if self._frame_changed(img) or (time.time() - self._last_ocr_t) > FORCE_OCR_SEC:
                    dets = self._detect(img)
                    if dets is None:
                        time.sleep(0.5)
                        continue
                    self._last_dets = dets
                    self._last_ocr_t = time.time()
                    self._stat["ocr"] += 1
                    self._stat["ocr_ms"] += (time.perf_counter() - t_grab) * 1000.0
                else:
                    dets = self._last_dets
            except Exception as e:
                print(f"[FULLSCREEN] OCR Error: {e}")
                time.sleep(0.5)
                continue

            # 3. Слежение
            self._track(img, dets, win_rect)

            # 4. Статистика раз в 10 секунд и выдержка FPS
            self._stat["frames"] += 1
            self._stat["grab_ms"] += (t_grab - t_cycle) * 1000.0
            elapsed = time.perf_counter() - t_cycle
            self.overlay.frame_ms = int(max(120, min(500, elapsed * 1000.0)))
            if time.time() - self._stat_t > 10.0:
                s = self._stat
                print(f"[FULLSCREEN] за 10 с: кадров {s['frames']}, из них с OCR {s['ocr']} "
                      f"(в среднем {s['ocr_ms'] / max(1, s['ocr']):.0f} мс), "
                      f"захват {s['grab_ms'] / max(1, s['frames']):.0f} мс, блоков сейчас {len(self.active_regions)}; "
                      f"переводов от модели {len(self._req_ms)}"
                      + (f" (в среднем {sum(self._req_ms) / len(self._req_ms):.0f} мс)" if self._req_ms else "")
                      + f", из памяти {self._mem_hits}, в памяти {len(self.memory)}")
                self._req_ms, self._mem_hits = [], 0
                self._stat = {"frames": 0, "ocr": 0, "grab_ms": 0.0, "ocr_ms": 0.0}
                self._stat_t = time.time()
            delay = 1.0 / max(0.1, self.overlay.capture_fps)
            if elapsed < delay:
                time.sleep(delay - elapsed)

    def _to_screen(self, rect, win_rect):
        """Пиксели кадра -> экранные координаты Qt (с учётом положения окна и масштаба Windows)."""
        left, top = win_rect[0], win_rect[1]
        dpr = float(getattr(self.overlay, "dpr", 1.0) or 1.0)
        x, y, w, h = rect
        return (int((left + x) / dpr), int((top + y) / dpr), int(w / dpr), int(h / dpr))

    def _line_h(self, rdata) -> int:
        dpr = float(getattr(self.overlay, "dpr", 1.0) or 1.0)
        return int(rdata.get('line_h', rdata['rect'][3]) / dpr)

    def _emit_pos(self, rid, rdata, win_rect, force=False):
        scr = self._to_screen(rdata['rect'], win_rect)
        last = rdata.get('shown')
        dz = max(MOVE_DEADZONE_PX, int(0.3 * min(scr[3], 60)))
        if (not force and last is not None
                and abs(scr[0] - last[0]) < dz and abs(scr[1] - last[1]) < dz
                and abs(scr[2] - last[2]) < 3 * dz and abs(scr[3] - last[3]) < 2 * dz):
            return
        rdata['shown'] = scr
        self.update_panel_pos.emit(rid, *scr, self._line_h(rdata))

    def _reset_text(self, rid, rdata, text):
        """Текст блока сменился: старый перевод прячем, новый запросим после стабилизации."""
        rdata['text'] = text
        rdata['generation'] += 1
        rdata['hits'] = 0
        rdata['state'] = 'new'
        rdata['tries'] = 0
        rdata['alt_text'], rdata['alt_hits'] = "", 0
        self.update_panel_gen.emit(rid, rdata['generation'])

    def _track(self, img, dets, win_rect):
        now = time.time()
        same, changed, new, lost = match_regions(self.active_regions, dets)

        for rid, di in same:
            d, rdata = dets[di], self.active_regions[rid]
            rdata['rect'] = (d['x'], d['y'], d['w'], d['h'])
            rdata['line_h'] = d.get('line_h', d['h'])
            rdata['missed'] = 0
            rdata['seen_t'] = now
            rdata['hits'] += 1
            rdata['alt_text'], rdata['alt_hits'] = "", 0
            self._emit_pos(rid, rdata, win_rect)

        for rid, di in changed:
            d, rdata = dets[di], self.active_regions[rid]
            rdata['rect'] = (d['x'], d['y'], d['w'], d['h'])
            rdata['line_h'] = d.get('line_h', d['h'])
            rdata['missed'] = 0
            rdata['seen_t'] = now
            self._emit_pos(rid, rdata, win_rect)
            if _text_sim(d['text'], rdata['text']) < DIFF_TEXT:
                # текст явно другой — старый перевод неуместен, прячем сразу
                print(f"[FULLSCREEN] Content changed: {rdata['text'][:30]} → {d['text'][:30]}")
                self._reset_text(rid, rdata, d['text'])
            elif _text_sim(d['text'], rdata.get('alt_text', "")) >= SAME_TEXT:
                # "немного другой" текст держится несколько кадров — это не дрожание OCR
                rdata['alt_hits'] += 1
                if rdata['alt_hits'] >= CHANGE_HITS:
                    self._reset_text(rid, rdata, d['text'])
            else:
                rdata['alt_text'], rdata['alt_hits'] = d['text'], 1

        for di in new:
            d = dets[di]
            rid = self.next_id
            self.next_id += 1
            rdata = {'rect': (d['x'], d['y'], d['w'], d['h']), 'text': d['text'], 'generation': 0,
                     'hits': 0, 'missed': 0, 'state': 'new', 'tries': 0, 'req_time': 0.0,
                     'alt_text': "", 'alt_hits': 0, 'shown': None, 'seen_t': now,
                     'line_h': d.get('line_h', d['h'])}
            self.active_regions[rid] = rdata
            rdata['shown'] = self._to_screen(rdata['rect'], win_rect)
            self.create_panel.emit(rid, *rdata['shown'], self._line_h(rdata))

        # Блок не найден: даём ему несколько кадров, перевод пока остаётся на месте
        for rid in lost:
            rdata = self.active_regions[rid]
            rdata['missed'] += 1
            # На живом фоне EasyOCR то видит строку, то нет. Переведённый блок держим по
            # времени, чтобы перевод не мигал; остальные убираем быстро.
            if rdata['state'] == 'done':
                gone = (now - rdata.get('seen_t', now)) > LOST_KEEP_SEC
            else:
                gone = rdata['missed'] > MISS_LIMIT
            if gone:
                self.delete_panel.emit(rid)
                del self.active_regions[rid]

        # На перевод. Блок = сразу запрос: не ждём, пока прочитается всё остальное.
        lang = self.overlay.source_lang
        # крупные блоки (диалог, описание) идут в очередь первыми, мелкие подписи — следом
        by_size = sorted(self.active_regions.items(), key=lambda kv: -(kv[1]['rect'][2] * kv[1]['rect'][3]))
        for rid, rdata in by_size:
            if rdata['missed']:
                continue
            if rdata['state'] == 'pending' and now - rdata['req_time'] > RETRY_SEC:
                print(f"[FULLSCREEN] нет ответа {RETRY_SEC:.0f} с: {rdata['text'][:30]}")
                rdata['state'] = 'new'
            if rdata['state'] != 'new':
                continue
            if not worth_translating(rdata['text'], lang):
                rdata['state'] = 'skip'          # цифры, значки — переводить нечего
                continue
            ru = self._memory_get(rdata['text'])
            if ru:
                rdata['state'] = 'done'
                self._mem_hits += 1
                self.update_panel_text.emit(rid, ru, rdata['generation'])
                continue
            if now < rdata.get('retry_at', 0.0):
                continue
            if rdata['tries'] >= MAX_TRIES:
                print(f"[FULLSCREEN] перевод не получен за {MAX_TRIES} попытки: {rdata['text'][:30]}")
                rdata['state'] = 'failed'
                continue
            self._request(img, rid, rdata, now)

    # ---------- память переводов ----------

    def _memory_get(self, text: str) -> str:
        key = _norm_text(text)
        if not key:
            return ""
        hit = self.memory.get(key)
        if hit or len(key) < 8:
            return hit or ""
        # распознавание дрожит: ищем почти такой же текст той же длины
        for k, v in self.memory.items():
            if abs(len(k) - len(key)) <= 2 and SequenceMatcher(None, k, key).ratio() >= 0.92:
                return v
        return ""

    # ---------- запрос перевода ----------

    def _request(self, img, rid, rdata, now):
        x, y, w, h = rdata['rect']
        pad = 6
        y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            rdata['state'] = 'skip'
            return
        rdata['state'] = 'pending'
        rdata['req_time'] = now
        rdata['tries'] += 1
        task = BlockTranslationTask(self, rid, rdata['generation'], _norm_text(rdata['text']), rdata['text'], crop.copy())
        # крупные блоки (диалог, описание) — первыми, мелкие подписи — потом
        self.pool.start(task, int(min(10_000, (w * h) // 100)))

    def _task_done(self, rid, gen, key, ru, ms):
        """Ответ по блоку (вызывается из потока перевода)."""
        if ru:
            if len(self.memory) >= MEMORY_MAX:
                self.memory.clear()
            if key:
                self.memory[key] = ru
            self._req_ms.append(ms)
        rdata = self.active_regions.get(rid)
        if rdata is None or rdata['generation'] != gen:
            return                                   # блок уже исчез или сменился
        if ru:
            rdata['state'] = 'done'
            self.update_panel_text.emit(rid, ru, gen)
        else:
            rdata['state'] = 'new'
            rdata['retry_at'] = time.time() + RETRY_PAUSE

    def _filter_overlapping_boxes(self, boxes):
        """
        Удаляет боксы, которые перекрывают друг друга.
        Оставляет самые маленькие (чтобы не склеивать строки).
        """
        # Сортируем по площади (сначала маленькие)
        boxes.sort(key=lambda b: b['w'] * b['h'])

        keep = []
        for b in boxes:
            x, y, w, h = b['x'], b['y'], b['w'], b['h']

            is_bad = False
            for k in keep:
                kx, ky, kw, kh = k['x'], k['y'], k['w'], k['h']

                # Пересечение
                ix1 = max(x, kx); iy1 = max(y, ky)
                ix2 = min(x+w, kx+kw); iy2 = min(y+h, ky+kh)
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                intersection = iw * ih

                # Если пересечение больше 20% от площади УЖЕ СОХРАНЕННОГО (маленького) бокса
                # Значит текущий (b) - это какой-то большой кусок, накрывающий (k). Выкидываем (b).
                if intersection > (kw * kh) * 0.2:
                    is_bad = True
                    break

            if not is_bad:
                keep.append(b)
        return keep

# ====================== Main Overlay Class ======================

class FullscreenOverlay(QWidget):
    def __init__(self, on_quit=None):
        super().__init__()
        self.setWindowTitle("Fullscreen Translator (Plugin Mode)")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                            | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        
        # Config placeholders (will be set by main app)
        self.llm_cfg = None
        self.ocr_cfg_dual = None
        self.tr_cfg_dual = None
        self.source_lang = "en"
        self.work_mode_idx = 0
        self.capture_fps = 1.0  # Дефолтное значение
        self.ocr_reader_factory = None   # общий EasyOCR-ридер основного модуля (на видеокарте)
        self.frame_ms = 250              # сколько длится один кадр слежения — столько же едет панелька
        self.parallel = 2                # сколько запросов сервер обрабатывает одновременно (его --parallel)
        try:
            self.dpr = float(QGuiApplication.primaryScreen().devicePixelRatio())
        except Exception:
            self.dpr = 1.0
        self.bound_hwnd = None
        self.paused = False
        
        # Visual settings
        self.bg_mode = "blur"
        self.bg_alpha = 64
        self.font_family = "Segoe UI"
        self.font_size = 20
        self.font_bold = False
        self.box_bg_mode = "solid"
        self.box_bg_alpha = 180
        
        self.panels = {} # {id: TextPanel}
        self.worker = None
        self._on_quit = on_quit
        
        # Hotkey to exit
        QShortcut(QKeySequence("Esc"), self, activated=self.quit_app)
        
        # Global Hotkeys
        self._hk = FullscreenHotkeyFilter(self)
        QApplication.instance().installNativeEventFilter(self._hk)
        
        # Fullscreen geometry
        self.resize(QGuiApplication.primaryScreen().size())
        self.move(0, 0)

    def start_worker(self):
        if self.worker: return
        self.worker = FullscreenCaptureWorker(self)
        
        # Connect signals
        self.worker.create_panel.connect(self.on_create_panel)
        self.worker.update_panel_pos.connect(self.on_update_pos)
        self.worker.update_panel_text.connect(self.on_update_text)
        self.worker.update_panel_gen.connect(self.on_update_gen)
        self.worker.delete_panel.connect(self.on_delete_panel)
        
        # Воркер теперь использует QThreadPool внутри, но сам он тоже поток
        self.worker.start()
        
    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None
            
    def showEvent(self, e):
        super().showEvent(e)
        hwnd = int(self.winId())
        # Без Ctrl: его нажатие долетает до игры раньше, чем срабатывает комбинация,
        # и новелла успевает заскипать диалог. Alt текст не скипает.
        # 101 = Pause (Quit), 102 = Alt+- (Toggle Pause) — как в оконном режиме
        if not RegisterHotKey(hwnd, 101, 0x0000, 0x13):
            print("[HOTKEY] НЕ удалось зарегистрировать Pause (занята другим приложением?)")
        if not RegisterHotKey(hwnd, 102, 0x0001, 0xBD):
            print("[HOTKEY] НЕ удалось зарегистрировать Alt+- (занята другим приложением?)")

    def closeEvent(self, e):
        hwnd = int(self.winId())
        UnregisterHotKey(hwnd, 101)
        UnregisterHotKey(hwnd, 102)
        self.stop_worker()
        for p in self.panels.values():
            p.close()
        super().closeEvent(e)

    def toggle_pause(self):
        self.paused = not self.paused
        for p in self.panels.values():
            p.setVisible(not self.paused and bool(p.text))

    # --- Slots for Worker ---
    def on_create_panel(self, pid, x, y, w, h, lh=0):
        if pid in self.panels: return
        
        p = TextPanel(self)
        # Apply settings
        p.bg_mode = self.bg_mode
        p.bg_alpha = self.bg_alpha
        p.font_family = self.font_family
        p.font_size = self.font_size
        p.font_bold = self.font_bold
        p.box_bg_mode = self.box_bg_mode
        p.box_bg_alpha = self.box_bg_alpha
        
        p.ocr_w, p.ocr_h, p.line_h = w, h, lh
        p.setGeometry(x, y, w, h)
        # Панелька создаётся скрытой и появляется только вместе с переводом: пустая она
        # лишь замыливала оригинал, пока перевода ещё нет.
        self.panels[pid] = p

    def on_update_pos(self, pid, x, y, w, h, lh=0):
        if pid in self.panels:
            p = self.panels[pid]
            p.ocr_h = h  # Обновляем базовую высоту
            p.ocr_w = w  # Обновляем базовую ширину
            p.line_h = lh
            p.glide_to(x, y, self.frame_ms)
            if p.text:
                p.update_size() # Проверяем, влезает ли текущий текст
            else:
                p.resize(w, h)

    def on_update_gen(self, pid, gen):
        """Текст изменился -> сбрасываем старый перевод."""
        if pid in self.panels:
            p = self.panels[pid]
            p.generation = gen
            p.text = "" # Очищаем, чтобы не висел старый текст
            p.hide()

    def on_update_text(self, pid, text, gen):
        if pid in self.panels and text:
            # Принимаем перевод ТОЛЬКО если поколение совпадает
            if self.panels[pid].generation == gen:
                p = self.panels[pid]
                p.text = text
                p.update_size() # Ресайзим под новый текст
                if not self.paused and not p.isVisible():
                    p.show()
                    p.apply_acrylic()
                p.repaint()

    def on_delete_panel(self, pid):
        if pid in self.panels:
            try:
                self.panels[pid].close()
                self.panels[pid].deleteLater()
                del self.panels[pid]
            except: pass

    def quit_app(self):
        if self._on_quit:
            self._on_quit()
        self.close()
