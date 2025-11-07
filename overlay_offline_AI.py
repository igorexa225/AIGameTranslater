# overlay_offline_AI.py — Конфиг + оверлей (две картинки → перевод через llama.cpp)
# Требует: PySide6, pywin32, numpy, opencv-python, requests
# Запуск: python overlay_offline_AI.py

import sys, os, time, re, subprocess, base64, difflib, ctypes, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import cv2

from PySide6.QtCore import (Qt, QRect, QPoint, QTimer, QEvent, QThread, Signal, QAbstractNativeEventFilter)
from PySide6.QtGui  import (QPainter, QColor, QPen, QFont, QKeySequence, QShortcut, QGuiApplication, QImage)
from PySide6.QtWidgets import (QApplication, QWidget, QMessageBox, QDialog,
    QLabel, QPushButton, QComboBox, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFontComboBox, QFileDialog, QPlainTextEdit, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox)

import win32con, win32gui, win32ui, win32api, win32process
import ctypes.wintypes as wt

# ======================== CONFIG (по умолчанию) =========================

# захват/обновление
MAX_OCR_FPS       = 1.0        # 0.5–2.0 кадров/сек
STALE_RECHECK_SEC = 2.1
STABILITY_HITS    = 1
MIN_CHAR_DIFF_FOR_UPDATE = 0
THUMB_SIZE        = (64, 24)
MIN_THUMB_DELTA   = 0.010

# UI/поведение
RENDER_MODE       = "smooth"   # "smooth" | "instant"
HANDLE            = 15
REQUIRE_BOUND_WINDOW = True

# Пути
BASE_DIR = Path(__file__).resolve().parent

PREFS_FILE = BASE_DIR / "config" / "ui_prefs.json"

def _load_prefs() -> dict:
    try:
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_prefs(d: dict):
    try:
        PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[UI] save prefs error:", e)

# llama.cpp server — ищем EXE и в models\lmm\llama.cpp, и в models\llm\llama.cpp
SERVER_CANDIDATES = [
    BASE_DIR / "models" / "lmm" / "llama.cpp" / ("llama-server.exe" if os.name == "nt" else "llama-server"),
    BASE_DIR / "models" / "llm" / "llama.cpp" / ("llama-server.exe" if os.name == "nt" else "llama-server"),
    BASE_DIR / "models" / "llama.cpp" / ("llama-server.exe" if os.name == "nt" else "llama-server"),
]

def _pick_server_exe() -> Optional[Path]:
    for p in SERVER_CANDIDATES:
        if p.exists():
            return p
    return None

SERVER_EXE = _pick_server_exe()

# Адаптер LLM
from llm_adapter import LLMConfig, preload_prompt_cache, vision_translate_from_images, reset_lore_cache, get_system_preview, init_lore_once

llm_cfg = LLMConfig(
    enabled=True,
    server="http://127.0.0.1:8080",
    model="Qwen3-VL-8B-Instruct",
    temp=0.2, top_p=0.9, max_tokens=15000, timeout_s=30.0, budget_ms=4000,
    slot_id=0, use_prompt_cache=True,
    lore_path=str(BASE_DIR / "assets" / "game_bible_exilium.txt"), max_ctx_chars=30000,
)

# ====================== Утилиты/канон ====================

_WS_RX   = re.compile(r"\s+")
_PUN_DASH = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]+")
_QT_RX    = re.compile(r"[«»“”„]")
_APO_RX   = re.compile(r"[’`´]")
_DOTS_RX  = re.compile(r"[.…]+")
_ZW_RX    = re.compile(r"[\u200B\u200C\u200D\u2060]")
_NBSP_RX  = re.compile(r"\u00A0")

def norm_ru(s: str) -> str:
    s = (s or "").replace("\ufeff","")
    s = _ZW_RX.sub("", s)
    s = _NBSP_RX.sub(" ", s)
    s = _QT_RX.sub('"', s)
    s = _APO_RX.sub("'", s)
    s = _PUN_DASH.sub("-", s)
    s = _DOTS_RX.sub("…", s)
    s = re.sub(r"\s+([,.:;!?])", r"\1", s)
    s = re.sub(r"([(\[«])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]»])", r"\1", s)
    s = _WS_RX.sub(" ", s)
    return s.strip()

def canon_ru(s: str) -> str:
    return norm_ru(s).lower()

def _thumb(gray: np.ndarray) -> np.ndarray:
    th = cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return th.astype(np.float32)/255.0

def _thumb_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return float(np.mean(np.abs(a-b)))

def _bgr_to_png_b64(img, max_side: int = 1400) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side/float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".png", img)
    if not ok: return ""
    return base64.b64encode(enc).decode("ascii")


# ================= Windows blur / PrintWindow =============

_HRESULT = getattr(wt, "HRESULT", ctypes.c_long)
class ACCENT_POLICY(ctypes.Structure):
    _fields_=[("AccentState",ctypes.c_int),("AccentFlags",ctypes.c_int),
              ("GradientColor",ctypes.c_uint),("AnimationId",ctypes.c_int)]
class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
    _fields_=[("Attribute",ctypes.c_int),("Data",ctypes.c_void_p),("SizeOfData",ctypes.c_size_t)]
WCA_ACCENT_POLICY=19
ACCENT_ENABLE_ACRYLICBLURBEHIND=4
user32 = ctypes.windll.user32
SetWindowCompositionAttribute = user32.SetWindowCompositionAttribute
SetWindowCompositionAttribute.argtypes=[wt.HWND, ctypes.POINTER(WINDOWCOMPOSITIONATTRIBDATA)]
SetWindowCompositionAttribute.restype=_HRESULT
def enable_acrylic(hwnd:int, tint_abgr:int=0x40101010):
    policy=ACCENT_POLICY(); policy.AccentState=ACCENT_ENABLE_ACRYLICBLURBEHIND
    data=WINDOWCOMPOSITIONATTRIBDATA(); data.Attribute=WCA_ACCENT_POLICY
    data.SizeOfData=ctypes.sizeof(policy)
    data.Data=ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p)
    SetWindowCompositionAttribute(wt.HWND(hwnd), ctypes.byref(data))

def _get_window_rect(hwnd):
    try:
        left,top,right,bottom = win32gui.DwmGetWindowAttribute(hwnd,9)
    except Exception:
        left,top,right,bottom = win32gui.GetWindowRect(hwnd)
    return left,top,right,bottom

def _printwindow_to_bgr(hwnd):
    left,top,right,bottom = _get_window_rect(hwnd)
    w,h=max(1,right-left), max(1,bottom-top)
    hwndDC=win32gui.GetWindowDC(hwnd)
    mfcDC=win32ui.CreateDCFromHandle(hwndDC)
    saveDC=mfcDC.CreateCompatibleDC()
    bmp=win32ui.CreateBitmap(); bmp.CreateCompatibleBitmap(mfcDC,w,h)
    saveDC.SelectObject(bmp)
    res=ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(),3)
    if res!=1:
        saveDC.BitBlt((0,0),(w,h),mfcDC,(0,0),win32con.SRCCOPY)
    info=bmp.GetInfo(); data=bmp.GetBitmapBits(True)
    img=np.frombuffer(data,dtype=np.uint8); img.shape=(info['bmHeight'],info['bmWidth'],4)
    bgr=img[...,:3].copy()
    win32gui.DeleteObject(bmp.GetHandle()); saveDC.DeleteDC(); mfcDC.DeleteDC(); win32gui.ReleaseDC(hwnd,hwndDC)
    return bgr,(left,top,w,h)

def _grab_from_bound(hwnd, region_px):
    full, (wx, wy, ww, hh) = _printwindow_to_bgr(hwnd)
    rx, ry, rw, rh = region_px

    # пересечение
    ix1 = max(rx, wx); iy1 = max(ry, wy)
    ix2 = min(rx + rw, wx + ww); iy2 = min(ry + rh, wy + hh)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, full

    sx1, sy1 = ix1 - wx, iy1 - wy
    sx2, sy2 = ix2 - wx, iy2 - wy
    sub = full[sy1:sy2, sx1:sx2]

    out = np.zeros((rh, rw, 3), dtype=np.uint8)
    dx, dy = ix1 - rx, iy1 - ry
    h = min(sub.shape[0], rh - dy); w = min(sub.shape[1], rw - dx)
    if h <= 0 or w <= 0:
        return None, full
    out[dy:dy + h, dx:dx + w] = sub[:h, :w]
    return out, full


# ===== DPI-aware координаты из Qt в физические пиксели ===
def _logical_to_physical(rect: QRect, overlay) -> tuple[int,int,int,int]:
    if overlay.bound_hwnd:
        try:
            dpi = user32.GetDpiForWindow(int(overlay.bound_hwnd))
            scale = dpi / 96.0 if dpi else 1.0
        except Exception:
            scale = 1.0
    else:
        scr = QGuiApplication.screenAt(rect.center())
        scale = float(scr.devicePixelRatio()) if scr else float(overlay.devicePixelRatioF() or 1.0)
    return (int(rect.x()*scale), int(rect.y()*scale),
            int(rect.width()*scale), int(rect.height()*scale))


# =============== Перечень пользовательских окон ==========
def list_user_windows(self_hwnd:int|None=None):
    out=[]
    def _is_system_class(cls:str):
        bad={"Button","Shell_TrayWnd","Progman","WorkerW","#32770","#32768","Scrollbar"}
        return cls in bad
    def enum_cb(hwnd, lParam):
        try:
            if self_hwnd and hwnd==self_hwnd: return True
            if not win32gui.IsWindowVisible(hwnd): return True
            if win32gui.GetWindow(hwnd, win32con.GW_OWNER): return True
            title=win32gui.GetWindowText(hwnd)
            if not title or title.strip()=="":
                return True
            cls=win32gui.GetClassName(hwnd)
            if _is_system_class(cls): return True
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            out.append((title, cls, pid, hwnd))
        except Exception:
            pass
        return True
    win32gui.EnumWindows(enum_cb, None)
    out.sort(key=lambda t: t[0].lower())
    return out


# =================== UI: Text Panel ======================

class RegionPanel(QWidget):
    def __init__(self, parent_overlay:'Overlay'):
        super().__init__(None)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.parent_overlay = parent_overlay
        self.text = ""
        self.margin = 12

    def sync_text(self, s: str):
        if s != self.text:
            self.text = s
            self.update()

    def set_geometry(self, rect: QRect):
        self.setGeometry(rect)

    def apply_acrylic(self):
        enable_acrylic(int(self.winId()))

    def paintEvent(self, _e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        ov=self.parent_overlay
        f=QFont(getattr(ov,'font_family','Segoe UI'), int(getattr(ov,'font_size', 22)))
        f.setBold(bool(getattr(ov,'font_bold', False)))
        p.setFont(f)
        p.setPen(QColor(0,0,0,220));   p.drawText(rect.translated(1,1), Qt.TextWordWrap|Qt.AlignLeft|Qt.AlignVCenter, self.text)
        p.setPen(QColor(255,255,255)); p.drawText(rect, Qt.TextWordWrap|Qt.AlignLeft|Qt.AlignVCenter, self.text)


# ===================== Data Model ========================

@dataclass
class RegionModel:
    display_rect: QRect
    capture_rect: QRect
    selected: bool=False
    split_mode: bool=False

    typing_speed_cps: int=80
    typing_accum: float = 0.0
    display_text: str=""
    target_ru: str=""
    display_idx: int=0
    typing_timer: Optional[QTimer]=field(default=None, repr=False)

    i_text: str = ""                 # мгновенный полный RU
    last_rendered_instant: str = ""  # чтобы не перерисовывать одинаковое
    seq: int = 0                     # «эпоха» входного текста
    commit_seq: int = 0              # с какой эпохой был последний коммит
    last_thumb: Optional[np.ndarray] = field(default=None, repr=False)
    last_ocr_ts: float = 0.0

    # канон/анти-дубли
    last_ru_canon: str = ""
    last_sent_ru_canon: str = ""
    last_sent_ru_raw: str = ""
    pending_ru_raw: str = ""
    pending_ru_canon: str = ""
    pending_hits: int = 0
    is_frozen: bool = False

    panel: Optional[RegionPanel]=field(default=None, repr=False)


# ==================== Capture Worker =====================

class CaptureWorker(QThread):
    textReady = Signal(int, str)
    def __init__(self, overlay:'Overlay', interval_ms=120):
        super().__init__()
        self.overlay = overlay
        self.interval_ms = interval_ms
        self._stop = False
        self._rr = 0

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            try:
                if REQUIRE_BOUND_WINDOW:
                    if (not self.overlay.enabled_overlay) or self.overlay.paused:
                        self.msleep(250); continue
                    if not self.overlay.bound_hwnd:
                        self.msleep(250); continue
                    if not win32gui.IsWindow(self.overlay.bound_hwnd):
                        print("[grab] bound window lost → pause")
                        self.overlay.bound_hwnd = None
                        self.msleep(250); continue

                if not self.overlay.regions:
                    self.msleep(self.interval_ms); continue

                idx = self._rr % len(self.overlay.regions); self._rr += 1
                r = self.overlay.regions[idx]

                now = time.monotonic()
                min_interval = 1.0 / MAX_OCR_FPS
                stale = (now - r.last_ocr_ts) >= STALE_RECHECK_SEC
                if (now - r.last_ocr_ts) < min_interval and not stale:
                    self.msleep(self.interval_ms); continue

                cap = r.capture_rect if r.split_mode else r.display_rect
                px, py, pw, ph = _logical_to_physical(cap, self.overlay)
                if pw < 10 or ph < 10:
                    self.msleep(self.interval_ms); continue

                hwnd = self.overlay.bound_hwnd
                if hwnd:
                    frame, full = _grab_from_bound(hwnd, (px, py, pw, ph))
                    if frame is None:
                        self.msleep(self.interval_ms); continue
                else:
                    screen = QGuiApplication.primaryScreen()
                    img = screen.grabWindow(0, px, py, pw, ph).toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                    import numpy as _np
                    arr = _np.frombuffer(img.constBits(), dtype=_np.uint8).reshape(img.height(), img.width(), 4)
                    frame = arr[:,:,:3].copy()
                    full  = None

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                th = _thumb(gray)
                changed = True
                if r.last_thumb is not None:
                    delta = _thumb_delta(th, r.last_thumb)
                    changed = (delta >= MIN_THUMB_DELTA)

                if not changed and not stale:
                    self.msleep(self.interval_ms); continue

                r.last_thumb = th
                r.last_ocr_ts = now

                # LLM: одна картинка (регион) → перевод
                reg_b64  = _bgr_to_png_b64(frame, max_side=1024)
                full_b64 = None
                ru = vision_translate_from_images(reg_b64, full_b64, self.overlay.llm_cfg)
                print(f"[LLM-OCR] got {len(ru) if ru else 0} chars")

                if ru:
                    try:
                        ru_canon = canon_ru(ru)
                    except Exception:
                        ru_canon = (ru or "").strip().lower()

                    # если первый раз — выводим сразу
                    if not (r.last_sent_ru_raw or r.pending_ru_raw):
                        r.last_sent_ru_raw   = ru
                        r.last_sent_ru_canon = ru_canon
                        r.pending_ru_raw = r.pending_ru_canon = ""
                        r.pending_hits = 0
                        self.textReady.emit(idx, ru)
                        self.msleep(self.interval_ms); continue

                    # Запрет микродёрганий — если отличий почти нет
                    prev_last = getattr(r, "last_sent_ru_canon", "") or ""
                    prev_pend = getattr(r, "pending_ru_canon", "") or ""
                    from difflib import SequenceMatcher
                    sim = SequenceMatcher(None, prev_last, ru_canon).ratio() if prev_last else 0.0
                    if sim >= 0.995:
                        self.msleep(self.interval_ms); continue

                    # Стабилизация: нужен одинаковый канон ровно STABILITY_HITS раз
                    if r.pending_ru_canon == ru_canon:
                        r.pending_hits += 1
                    else:
                        r.pending_ru_canon = ru_canon
                        r.pending_ru_raw   = ru
                        r.pending_hits = 1

                    if r.pending_hits < STABILITY_HITS:
                        self.msleep(self.interval_ms); continue

                    r.last_sent_ru_canon = ru_canon
                    r.last_sent_ru_raw   = ru
                    r.pending_hits = 0
                    self.textReady.emit(idx, ru)

                self.msleep(self.interval_ms)
            except Exception as e:
                print("Worker loop error:", e)
                self.msleep(160)


# ========== Hotkey Filter (глобальные хоткеи) ============

WM_HOTKEY, MOD_ALT, MOD_CONTROL = 0x0312, 0x0001, 0x0002
VK_Q, VK_F1, VK_F2, VK_N, VK_B, VK_D, VK_TAB, VK_UP, VK_DOWN, VK_OEM_4, VK_OEM_6 = 0x51,0x70,0x71,0x4E,0x42,0x44,0x09,0x26,0x28,0xDB,0xDD

class HotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, overlay:'Overlay', on_quit=None):
        super().__init__()
        self.overlay=overlay
        self._on_quit = on_quit
    def nativeEventFilter(self, et, msgptr):
        if et!="windows_generic_MSG": return False,0
        msg=wt.MSG.from_address(int(msgptr))
        if msg.message==WM_HOTKEY:
            hid=msg.wParam
            if   hid==1: self.overlay.toggle_overlay()
            elif hid==2: self.overlay.toggle_edit_mode()
            elif hid==3: self.overlay.add_region()
            elif hid==4: self.overlay.bind_window_under_cursor()
            elif hid==5: self.overlay.toggle_split_selected()
            elif hid==6: self.overlay.switch_edit_target()
            elif hid==7: self.overlay.change_speed(+10)
            elif hid==8: self.overlay.change_speed(-10)
            elif hid==9: self.overlay.change_preview_lag(+2)
            elif hid==10:self.overlay.change_preview_lag(-2)
            elif hid==11:self.overlay.quit_to_config()
            return True,0
        return False,0


# === helper: длина общего префикса (для smooth печати)
def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i


# ======================== Overlay ========================

class Overlay(QWidget):
    def __init__(self, on_quit=None, start_worker_immediately: bool=False):
        super().__init__()
        self.setWindowTitle("On-screen Translator — LLM Vision")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._drag = False
        self._drag_mode = None   # "move" | "tl" | "tr" | "bl" | "br"
        self._drag_idx  = -1
        self._drag_start_pt   = QPoint()
        self._drag_start_rect = QRect()

        self.edit_mode=True
        self.enabled_overlay=True
        self.paused=False
        self.bound_hwnd: Optional[int]=None
        self.edit_target='display'

        # шрифт панели
        self.font_family = "Segoe UI"
        self.font_size   = 22
        self.font_bold   = False

        self.llm_cfg = llm_cfg  # для доступа из воркера
        self._on_quit = on_quit

        # первый регион
        scr=QGuiApplication.primaryScreen().availableGeometry()
        w,h=900,220; cx,cy=scr.center().x()-w//2, scr.center().y()-h//2
        first=RegionModel(QRect(cx,cy,w,h), QRect(cx,cy,w,h), selected=True)
        first.panel=RegionPanel(self); first.panel.set_geometry(first.display_rect); first.panel.show(); first.panel.apply_acrylic()
        self.regions:[RegionModel]=[first]

        self.worker=None
        if start_worker_immediately:
            self.start_worker()

        # локальные хоткеи
        for seq in ["+", "=", Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Up]:
            QShortcut(QKeySequence(seq), self, activated=lambda s=+10: self.change_speed(s))
        for seq in ["-", "_", Qt.Key_Minus, Qt.Key_Underscore, Qt.Key_Down]:
            QShortcut(QKeySequence(seq), self, activated=lambda s=-10: self.change_speed(s))
        QShortcut(QKeySequence("["), self, activated=lambda: self.change_preview_lag(-2))
        QShortcut(QKeySequence("]"), self, activated=lambda: self.change_preview_lag(+2))
        QShortcut(QKeySequence("F1"), self, activated=self.toggle_overlay)
        QShortcut(QKeySequence("F2"), self, activated=self.toggle_edit_mode)
        QShortcut(QKeySequence("N"),  self, activated=self.add_region)
        QShortcut(QKeySequence("Delete"), self, activated=self.delete_selected)
        QShortcut(QKeySequence("Tab"), self, activated=self.switch_edit_target)
        QShortcut(QKeySequence("Esc"), self, activated=lambda: QApplication.quit())

        # глобальные хоткеи
        self._hk=HotkeyFilter(self, on_quit=on_quit)
        QApplication.instance().installNativeEventFilter(self._hk)

        self.resize(scr.width(),scr.height()); self.move(scr.x(),scr.y())

    def dump_regions_for_prefs(self):
        out = []
        for r in getattr(self, "regions", []):
            out.append({
                "display_rect": [r.display_rect.x(), r.display_rect.y(), r.display_rect.width(), r.display_rect.height()],
                "capture_rect": [r.capture_rect.x(), r.capture_rect.y(), r.capture_rect.width(), r.capture_rect.height()],
                "split_mode": bool(r.split_mode),
                "typing_speed_cps": int(getattr(r, "typing_speed_cps", 80)),
            })
        return out

    def load_regions_from_prefs(self, lst):
        # закрыть старые панели
        try:
            for r in getattr(self, "regions", []):
                if getattr(r, "panel", None):
                    r.panel.close()
        except Exception:
            pass
        self.regions = []

        if not lst:
            # если пусто — создать одну стандартную область
            scr = QGuiApplication.primaryScreen().availableGeometry()
            w, h = 900, 220
            cx, cy = scr.center().x() - w//2, scr.center().y() - h//2
            first = RegionModel(QRect(cx, cy, w, h), QRect(cx, cy, w, h), selected=True)
            first.panel = RegionPanel(self)
            first.panel.set_geometry(first.display_rect)
            first.panel.show()
            first.panel.apply_acrylic()
            self.regions = [first]
            self.update()
            return

        for it in lst:
            dx, dy, dw, dh = it.get("display_rect", [0, 0, 900, 220])
            cx, cy, cw, ch = it.get("capture_rect", [dx, dy, dw, dh])
            split = bool(it.get("split_mode", False))
            spd   = int(it.get("typing_speed_cps", 80))
            rm = RegionModel(QRect(dx, dy, dw, dh), QRect(cx, cy, cw, ch),
                        selected=False, split_mode=split, typing_speed_cps=spd)
            rm.panel = RegionPanel(self)
            rm.panel.set_geometry(rm.display_rect)
            rm.panel.show()
            rm.panel.apply_acrylic()
            self.regions.append(rm)

        if self.regions:
            self.regions[-1].selected = True
        self.update()
    
    def start_worker(self):
        if self.worker is None:
            self.worker=CaptureWorker(self, interval_ms=140)
            self.worker.textReady.connect(self.on_text_ready)
            self.worker.start()

    def stop_worker(self):
        if self.worker is not None:
            try:
                self.worker.stop(); self.worker.wait(800)
            except Exception: pass
            self.worker=None

    # ---- рисуем рамки в режиме правки ----
    def paintEvent(self,_e):
        if not self.enabled_overlay or not self.edit_mode: return
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing,True)
        for r in self.regions:
            pen=QPen(QColor(0,200,255,230) if r.selected else QColor(255,255,255,120)); pen.setWidth(4); p.setPen(pen)
            p.drawRect(r.display_rect); self._handles(p, r.display_rect, QColor(0,200,255,230))
            if r.split_mode:
                pen2=QPen(QColor(255,165,0,220)); pen2.setStyle(Qt.DashLine); pen2.setWidth(4); p.setPen(pen2)
                p.drawRect(r.capture_rect); self._handles(p, r.capture_rect, QColor(255,165,0,220))
            tip=QRect(r.display_rect.left(), r.display_rect.top()-24, 900, 20)
            p.fillRect(tip, QColor(0,0,0,110))
            p.setPen(QColor(255,255,255)); p.setFont(QFont("Segoe UI",10))
            m = "[CAPTURE]" if (self.edit_target=='capture' and r.split_mode) else "[DISPLAY]"
            p.drawText(tip, Qt.AlignLeft|Qt.AlignVCenter,
                       f"{m}  Скорость: {r.typing_speed_cps} | (+/-/↑/↓, [ ]) | Ctrl+Alt+D — split | Tab — переключить | Ctrl+Alt+Q — выход в настройки")

    def _active_rect(self, r):
        return r.capture_rect if (self.edit_target == 'capture' and r.split_mode) else r.display_rect

    def _set_active_rect(self, r, rect: QRect):
        rect = QRect(rect)
        if self.edit_target == 'capture' and r.split_mode:
            r.capture_rect = rect
        else:
            r.display_rect = rect
            if r.panel:
                r.panel.set_geometry(r.display_rect)
                r.panel.sync_text(r.display_text)

    def _hit_handle(self, rect: QRect, pos: QPoint):
        hs = HANDLE
        tl = QRect(rect.left()-hs//2,  rect.top()-hs//2,    hs, hs)
        tr = QRect(rect.right()-hs//2, rect.top()-hs//2,    hs, hs)
        bl = QRect(rect.left()-hs//2,  rect.bottom()-hs//2, hs, hs)
        br = QRect(rect.right()-hs//2, rect.bottom()-hs//2, hs, hs)
        if tl.contains(pos): return "tl"
        if tr.contains(pos): return "tr"
        if bl.contains(pos): return "bl"
        if br.contains(pos): return "br"
        if rect.contains(pos): return "move"
        return None

    def _update_cursor(self, pos: QPoint):
        for r in reversed(self.regions):
            hit = self._hit_handle(self._active_rect(r), pos)
            if hit in ("tl","br"):
                self.setCursor(Qt.SizeFDiagCursor); return
            if hit in ("tr","bl"):
                self.setCursor(Qt.SizeBDiagCursor); return
            if hit == "move":
                self.setCursor(Qt.SizeAllCursor);   return
        self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, e):
        if not (self.enabled_overlay and self.edit_mode): return
        if e.button() != Qt.LeftButton: return
        pos = e.position().toPoint()
        for i in range(len(self.regions)-1, -1, -1):
            r = self.regions[i]
            hit = self._hit_handle(self._active_rect(r), pos)
            if hit:
                for k in self.regions: k.selected = False
                r.selected = True
                self._drag = True; self._drag_mode = hit; self._drag_idx  = i
                self._drag_start_pt = pos; self._drag_start_rect = QRect(self._active_rect(r))
                self.update(); return

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        if not (self.enabled_overlay and self.edit_mode):
            return
        if not self._drag:
            self._update_cursor(pos)
            return
        if not (0 <= self._drag_idx < len(self.regions)): return
        r = self.regions[self._drag_idx]
        start = self._drag_start_rect
        dx = pos.x() - self._drag_start_pt.x()
        dy = pos.y() - self._drag_start_pt.y()

        MIN_W, MIN_H = 80, 50
        rect = QRect(start)

        if self._drag_mode == "move":
            rect.moveTo(start.x() + dx, start.y() + dy)
        else:
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            if "t" in self._drag_mode: y1 = start.top() + dy
            if "b" in self._drag_mode: y2 = start.bottom() + dy
            if "l" in self._drag_mode: x1 = start.left() + dx
            if "r" in self._drag_mode: x2 = start.right() + dx
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            rect = QRect(QPoint(x1, y1), QPoint(x2, y2))

        if rect.width()  < MIN_W: rect.setWidth(MIN_W)
        if rect.height() < MIN_H: rect.setHeight(MIN_H)

        self._set_active_rect(r, rect)
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._drag:
            self._drag = False; self._drag_mode = None; self._drag_idx  = -1
            self._update_cursor(e.position().toPoint())

    def _handles(self,p,rect,color):
        hs=HANDLE
        p.fillRect(QRect(rect.left()-hs//2,  rect.top()-hs//2, hs, hs), color)
        p.fillRect(QRect(rect.right()-hs//2, rect.top()-hs//2, hs, hs), color)
        p.fillRect(QRect(rect.left()-hs//2,  rect.bottom()-hs//2, hs, hs), color)
        p.fillRect(QRect(rect.right()-hs//2, rect.bottom()-hs//2, hs, hs), color)

    def on_text_ready(self, idx:int, ru:str):
        if REQUIRE_BOUND_WINDOW and not self.bound_hwnd:
            return
        if not (0 <= idx < len(self.regions)): return
        if not self.enabled_overlay or self.paused: return

        r = self.regions[idx]
        # INSTANT
        if RENDER_MODE == "instant":
            if ru != r.last_rendered_instant:
                r.last_rendered_instant = ru
                r.target_ru = r.i_text = ru
                r.display_idx = len(ru)
                r.display_text = ru
                if r.panel: r.panel.sync_text(ru)
            return

        # SMOOTH
        if ru != r.i_text:
            r.i_text = ru
            common = _lcp_len(r.i_text, r.target_ru)
            r.target_ru = r.i_text
            if common < r.display_idx: r.display_idx = common

        self._ensure_timer(r)
        if r.panel:
            r.display_text = r.target_ru[:r.display_idx]
            r.panel.sync_text(r.display_text)

    def _ensure_timer(self, r: 'RegionModel'):
        if RENDER_MODE != "smooth": return
        if r.typing_timer is None:
            r.typing_timer = QTimer(self)
            r.typing_timer.timeout.connect(lambda rr=r: self._tick(rr))
            r.typing_timer.start(16)
        elif not r.typing_timer.isActive():
            r.typing_timer.start(16)

    def _tick(self, r: 'RegionModel'):
        # эмуляция печати
        r.typing_accum += r.typing_speed_cps * 0.016
        k = int(r.typing_accum)
        if k <= 0: return
        r.typing_accum -= k
        if r.display_idx >= len(r.target_ru):
            if r.typing_timer and r.typing_timer.isActive(): r.typing_timer.stop()
            return
        r.display_idx = min(len(r.target_ru), r.display_idx + k)
        r.display_text = r.target_ru[:r.display_idx]
        if r.panel: r.panel.sync_text(r.display_text)

    # управление
    def showEvent(self,e:QEvent):
        super().showEvent(e)
        hwnd=int(self.winId())
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11)
        except Exception: pass
        ctypes.windll.user32.RegisterHotKey(hwnd,1,0x0002|0x0001,0x70) # Ctrl+Alt+F1
        ctypes.windll.user32.RegisterHotKey(hwnd,2,0x0002|0x0001,0x71) # Ctrl+Alt+F2
        ctypes.windll.user32.RegisterHotKey(hwnd,3,0x0002|0x0001,0x4E) # Ctrl+Alt+N
        ctypes.windll.user32.RegisterHotKey(hwnd,4,0x0002|0x0001,0x42) # Ctrl+Alt+B
        ctypes.windll.user32.RegisterHotKey(hwnd,5,0x0002|0x0001,0x44) # Ctrl+Alt+D
        ctypes.windll.user32.RegisterHotKey(hwnd,6,0x0002|0x0001,0x09) # Ctrl+Alt+Tab
        ctypes.windll.user32.RegisterHotKey(hwnd,7,0x0002|0x0001,0x26) # Ctrl+Alt+Up
        ctypes.windll.user32.RegisterHotKey(hwnd,8,0x0002|0x0001,0x28) # Ctrl+Alt+Down
        ctypes.windll.user32.RegisterHotKey(hwnd,9,0x0002|0x0001,0xDD) # Ctrl+Alt+]
        ctypes.windll.user32.RegisterHotKey(hwnd,10,0x0002|0x0001,0xDB)# Ctrl+Alt+[
        ctypes.windll.user32.RegisterHotKey(hwnd,11,0x0002|0x0001,0x51)# Ctrl+Alt+Q
        for r in self.regions:
            if r.panel is None:
                r.panel = RegionPanel(self); r.panel.show(); r.panel.apply_acrylic()
            r.panel.set_geometry(r.display_rect); r.panel.sync_text(r.display_text)

    def closeEvent(self,e:QEvent):
        hwnd=int(self.winId())
        for hid in (1,2,3,4,5,6,7,8,9,10,11): ctypes.windll.user32.UnregisterHotKey(hwnd,hid)
        self.stop_worker()
        for r in self.regions:
            try:
                if r.panel: r.panel.close()
            except Exception: pass
        super().closeEvent(e)

    def toggle_overlay(self):
        self.enabled_overlay=not self.enabled_overlay
        for r in self.regions:
            if r.panel: r.panel.setVisible(self.enabled_overlay)
        self.update()

    def toggle_edit_mode(self):
        self.edit_mode=not self.edit_mode
        for r in self.regions:
            if r.panel: r.panel.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.update()

    def switch_edit_target(self):
        self.edit_target = 'capture' if (self.edit_target=='display') else 'display'
        QMessageBox.information(self,"Режим правки","Правим CAPTURE" if self.edit_target=='capture' else "Правим DISPLAY")

    def change_speed(self, delta:int):
        for r in self.regions:
            if r.selected:
                r.typing_speed_cps = max(5, min(400, r.typing_speed_cps + delta))
                break
        self.update()

    def change_preview_lag(self, delta:int):
        pass

    def bind_window_under_cursor(self):
        x,y=win32api.GetCursorPos()
        hwnd=win32gui.WindowFromPoint((x,y))
        if hwnd==int(self.winId()):
            hwnd=win32gui.WindowFromPoint((x,y+5))
        if hwnd and win32gui.IsWindow(hwnd):
            self.bound_hwnd=hwnd
            title=win32gui.GetWindowText(hwnd) or "без заголовка"
            QMessageBox.information(self,"Привязка окна",f"Захват привязан к окну:\n{title}")
        else:
            QMessageBox.warning(self,"Привязка окна","Не удалось определить окно под курсором.")

    def add_region(self):
        scr=self.geometry(); w,h=900,220
        x=scr.center().x()-w//2; y=scr.center().y()-h//2
        for r in self.regions: r.selected=False
        r=RegionModel(QRect(x,y,w,h), QRect(x,y,w,h), selected=True)
        r.panel=RegionPanel(self); r.panel.set_geometry(r.display_rect); r.panel.show(); r.panel.apply_acrylic()
        r.last_sent_ru_raw = r.last_sent_ru_canon = ""
        r.pending_ru_raw = r.pending_ru_canon = ""
        r.pending_hits = 0
        self.regions.append(r); self.update()

    def delete_selected(self):
        keep=[]
        for r in self.regions:
            if r.selected:
                try:
                    if r.panel: r.panel.close()
                except Exception: pass
            else:
                keep.append(r)
        self.regions = keep or self.regions
        if self.regions: self.regions[-1].selected=True
        self.update()

    def toggle_split_selected(self):
        for r in self.regions:
            if r.selected:
                if not r.split_mode:
                    r.split_mode = True
                    r.capture_rect = QRect(r.display_rect)   # старт — копия
                    self.edit_target = 'capture'
                    QMessageBox.information(self, "Split",
                        "Раздельный режим. Сейчас правим CAPTURE (Tab — переключить).")
                else:
                    r.split_mode = False
                    r.capture_rect = QRect(r.display_rect)
                break
        self.update()

    def quit_to_config(self):
        # 1) остановить захватчик
        self.stop_worker()
        # 2) жёстко остановить llama-server (если запущен через автозапуск)
        try:
            _stop_llama_server()
        except Exception:
            pass
        # 3) вернуть оверлей в режим редактирования (область остаётся)
        self.enabled_overlay = True
        self.edit_mode = True
        for r in getattr(self, 'regions', []):
            if getattr(r, 'panel', None) is None:
                r.panel = RegionPanel(self)
                r.panel.show()
                r.panel.apply_acrylic()
            r.panel.set_geometry(r.display_rect)
            r.panel.sync_text(r.display_text)
        # 4) показать оверлей (на случай, если был спрятан) и вернуть окно настроек
        self.show()
        if callable(self._on_quit):
            try:
                self._on_quit()
            except Exception:
                pass


# =================== Автостарт llama-server ===============

def _q(p): return f'"{str(p)}"'

def scan_gguf(roots: List[Path]) -> Tuple[List[str], List[str]]:
    models=[]; projs=[]
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob("*.gguf"):
            name=p.name.lower()
            if "mmproj" in name: projs.append(str(p))
            else: models.append(str(p))
    models.sort(); projs.sort()
    return models, projs

def rebuild_server_cmd(server_exe: Path, model_path:str, mmproj_path:str,
                       host:str, port:int, ctx:int, batch:int, ubatch:int, parallel:int, ngl:int) -> str:
    cmd = f"{_q(server_exe)} "
    if model_path:  cmd += f"-m {_q(model_path)} "
    if mmproj_path: cmd += f"--mmproj {_q(mmproj_path)} "
    cmd += f"-ngl {ngl} --ctx-size {ctx} --batch-size {batch} --ubatch-size {ubatch} --parallel {parallel} "
    cmd += f"--host {host} --port {port}"
    return cmd

def _ping_llama(url: str, timeout: float = 0.5) -> bool:
    try:
        import requests
        r = requests.get(url.rstrip("/") + "/health", timeout=timeout)
        return r.status_code < 500
    except Exception:
        try:
            import requests
            r = requests.get(url.rstrip("/") + "/v1/models", timeout=timeout)
            return r.status_code < 500
        except Exception:
            return False

_LLAMA_PROC = None

def _ensure_llama_server(cmd: str, server_url: str):
    """Стартуем сервер, если не отвечает; ждём, прогреваем промпт."""
    global _LLAMA_PROC
    if _ping_llama(server_url):
        print(f"[LLM] server ok @ {server_url}")
        return
    if not cmd:
        print("[LLM] LLAMA_SERVER_CMD is empty — skip autostart"); return
    try:
        flags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        )
        _LLAMA_PROC = subprocess.Popen(cmd, shell=True, creationflags=flags)
        print(f"[LLM] llama-server started (pid={_LLAMA_PROC.pid})")
        for _ in range(120):
            time.sleep(0.5)
            if _ping_llama(server_url):
                print("[LLM] llama-server поднялся")
                break
            if _LLAMA_PROC.poll() is not None:
                print("[LLM] llama-server exited early code", _LLAMA_PROC.returncode)
                break
    except Exception as e:
        print("[LLM] автозапуск не удался:", e)

def _stop_llama_server():
    global _LLAMA_PROC
    if _LLAMA_PROC and _LLAMA_PROC.poll() is None:
        try:
            _LLAMA_PROC.terminate()
            try:
                _LLAMA_PROC.wait(timeout=3)
            except Exception:
                _LLAMA_PROC.kill()
        except Exception:
            pass
    _LLAMA_PROC = None


# ====================== Настроечный GUI ===================

class ConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Translator — Настройки")
        self.setGeometry(120,120,980,760)

        # --- controls ---
        self.btnStart = QPushButton("Начать перевод")

        # окна/процессы
        self.cbWindow = QComboBox(); self.btnRefreshWin=QPushButton("Обновить список")

        # шрифт панели
        self.fontFamily = QFontComboBox()
        self.fontSize   = QSpinBox(); self.fontSize.setRange(8,72); self.fontSize.setValue(22)
        self.fontBold   = QCheckBox("Жирный")

        # модель + mmproj
        self.cbModel  = QComboBox()
        self.cbMmproj = QComboBox()
        self.btnRescan = QPushButton("Сканировать models/lmm и models/llm")

        # настройки модели
        self.spMaxTok = QSpinBox(); self.spMaxTok.setRange(16, 16000); self.spMaxTok.setValue(int(getattr(llm_cfg,'max_tokens',15000)))
        self.spTO     = QDoubleSpinBox(); self.spTO.setRange(1.0, 120.0); self.spTO.setDecimals(1); self.spTO.setValue(float(getattr(llm_cfg,'timeout_s',30.0)))
        self.spTemp   = QDoubleSpinBox(); self.spTemp.setRange(0.0, 1.5); self.spTemp.setSingleStep(0.05); self.spTemp.setValue(float(getattr(llm_cfg,'temp',0.2)))
        self.spTopP   = QDoubleSpinBox(); self.spTopP.setRange(0.0, 1.0); self.spTopP.setSingleStep(0.05); self.spTopP.setValue(float(getattr(llm_cfg,'top_p',0.9)))
        
        # доп. параметры генерации
        self.spTopK  = QSpinBox();      self.spTopK.setRange(0, 10000); self.spTopK.setValue(0)
        self.spRP    = QDoubleSpinBox(); self.spRP.setRange(0.1, 2.0);  self.spRP.setSingleStep(0.05); self.spRP.setValue(1.0)
        self.spSeed  = QSpinBox();       self.spSeed.setRange(0, 2_000_000_000); self.spSeed.setValue(0)
        self.spSlot  = QSpinBox();       self.spSlot.setRange(0, 7);    self.spSlot.setValue(int(getattr(llm_cfg,'slot_id',0)))
        self.cbCache = QCheckBox("cache_prompt (прогрев)"); self.cbCache.setChecked(bool(getattr(llm_cfg,'use_prompt_cache',True)))

        # серверные параметры (llama-server)
        self.spCtx   = QSpinBox();       self.spCtx.setRange(1024, 65536); self.spCtx.setValue(15000)
        self.spBatch = QSpinBox();       self.spBatch.setRange(1, 4096);   self.spBatch.setValue(256)
        self.spUBatch= QSpinBox();       self.spUBatch.setRange(1, 4096);  self.spUBatch.setValue(64)
        self.spPar   = QSpinBox();       self.spPar.setRange(1, 16);       self.spPar.setValue(1)
        self.spNGL   = QSpinBox();       self.spNGL.setRange(0, 999);      self.spNGL.setValue(999)
        self.edHost  = QLineEdit("127.0.0.1")
        self.spPort  = QSpinBox();       self.spPort.setRange(1, 65535);   self.spPort.setValue(8080)

        # константы/поведение
        self.spFps    = QDoubleSpinBox(); self.spFps.setRange(0.1, 5.0); self.spFps.setSingleStep(0.1); self.spFps.setValue(float(MAX_OCR_FPS))
        self.spDelta  = QDoubleSpinBox(); self.spDelta.setRange(0.001, 0.2); self.spDelta.setSingleStep(0.001); self.spDelta.setValue(float(MIN_THUMB_DELTA))
        self.cbSmooth = QCheckBox("Плавный вывод (smooth)"); self.cbSmooth.setChecked(RENDER_MODE=="smooth")
        self.spHandle = QSpinBox(); self.spHandle.setRange(6, 30); self.spHandle.setValue(int(HANDLE))

        # лор
        self.edLore  = QLineEdit(str(llm_cfg.lore_path or ""))
        self.btnLore = QPushButton("Выбрать…")

        # промпт
        self.btnPrompt  = QPushButton("Настроить промпт…")
        self.btnPreview = QPushButton("Предпросмотр system…")

        # --- layout ---
        top = QVBoxLayout(self)
        top.addWidget(self.btnStart)

        gbWin = QGroupBox("Окно для захвата"); lw = QHBoxLayout(gbWin)
        lw.addWidget(QLabel("Окно:")); lw.addWidget(self.cbWindow, 1); lw.addWidget(self.btnRefreshWin)
        top.addWidget(gbWin)

        gbModel = QGroupBox("Модель и mmproj"); fm = QFormLayout(gbModel)
        fm.addRow("Модель .gguf:", self.cbModel)
        fm.addRow("mmproj .gguf:", self.cbMmproj)
        fm.addRow(self.btnRescan)
        top.addWidget(gbModel)

        gbLore = QGroupBox("Лор (game_bible)"); fl = QHBoxLayout(gbLore)
        fl.addWidget(QLabel("Файл лора:")); fl.addWidget(self.edLore, 1); fl.addWidget(self.btnLore)
        top.addWidget(gbLore)

        gbFont = QGroupBox("Шрифт панели"); ff = QFormLayout(gbFont)
        ff.addRow("Гарнитура:", self.fontFamily)
        ff.addRow("Размер:", self.fontSize)
        ff.addRow("", self.fontBold)
        top.addWidget(gbFont)

        gbLLM = QGroupBox("Настройки модели"); flm = QFormLayout(gbLLM)
        flm.addRow("max_tokens:", self.spMaxTok)
        flm.addRow("timeout, сек:", self.spTO)
        flm.addRow("temperature:", self.spTemp)
        flm.addRow("top_p:", self.spTopP)
        top.addWidget(gbLLM)
        
        gbGen = QGroupBox("Доп. параметры генерации"); fgen = QFormLayout(gbGen)
        fgen.addRow("top_k:", self.spTopK)
        fgen.addRow("repeat_penalty:", self.spRP)
        fgen.addRow("seed:", self.spSeed)
        fgen.addRow("slot_id:", self.spSlot)
        fgen.addRow("", self.cbCache)
        top.addWidget(gbGen)

        gbSrv = QGroupBox("Сервер (llama.cpp)"); fsrv = QFormLayout(gbSrv)
        fsrv.addRow("ctx-size:", self.spCtx)
        fsrv.addRow("batch-size:", self.spBatch)
        fsrv.addRow("ubatch-size:", self.spUBatch)
        fsrv.addRow("parallel:", self.spPar)
        fsrv.addRow("ngl:", self.spNGL)
        fsrv.addRow("host:", self.edHost)
        fsrv.addRow("port:", self.spPort)
        top.addWidget(gbSrv)

        gbK = QGroupBox("Константы"); fk = QFormLayout(gbK)
        fk.addRow("FPS захвата:", self.spFps)
        fk.addRow("MIN_THUMB_DELTA:", self.spDelta)
        fk.addRow("", self.cbSmooth)
        fk.addRow("Размер ручек рамки:", self.spHandle)
        top.addWidget(gbK)

        top.addWidget(self.btnPrompt)
        top.addWidget(self.btnPreview)
        top.addStretch(1)

        # events
        self.btnRefreshWin.clicked.connect(self._fill_windows)
        self.btnRescan.clicked.connect(self._rescan_models)
        self.btnLore.clicked.connect(self._browse_lore)
        self.btnPrompt.clicked.connect(self._edit_prompt)
        self.btnPreview.clicked.connect(self._preview_system)
        self.btnStart.clicked.connect(self._start)

        # --- создаём ОВЕРЛЕЙ СРАЗУ (без стартa воркера) для настройки области ---
        self.overlay = Overlay(on_quit=self._back_from_overlay, start_worker_immediately=False)
        self.overlay.show()
        self.raise_()  # держим окно настроек сверху

        self._rescan_models()
        self._fill_windows()
        self._load_prefs_into_ui()
    
    def _load_prefs_into_ui(self):
        p = _load_prefs()
        # шрифт
        fam = p.get("font_family")
        if fam:
            from PySide6.QtGui import QFont
            self.fontFamily.setCurrentFont(QFont(fam))
        self.fontSize.setValue(int(p.get("font_size", 22)))
        self.fontBold.setChecked(bool(p.get("font_bold", False)))
        # модель
        model = p.get("model_path", "")
        if model:
            i = self.cbModel.findText(model)
            if i >= 0: self.cbModel.setCurrentIndex(i)
        mmproj = p.get("mmproj_path", "")
        if mmproj:
            i = self.cbMmproj.findText(mmproj)
            if i >= 0: self.cbMmproj.setCurrentIndex(i)
        # лор
        lore = p.get("lore_path")
        if lore: self.edLore.setText(lore)
        # LLM
        self.spMaxTok.setValue(int(p.get("max_tokens", 15000)))
        self.spTO.setValue(float(p.get("timeout_s", 30.0)))
        self.spTemp.setValue(float(p.get("temp", 0.2)))
        self.spTopP.setValue(float(p.get("top_p", 0.9)))
        self.spTopK.setValue(int(p.get("top_k", 0)))
        self.spRP.setValue(float(p.get("repeat_penalty", 1.0)))
        self.spSeed.setValue(int(p.get("seed", 0)))
        self.spSlot.setValue(int(p.get("slot_id", 0)))
        self.cbCache.setChecked(bool(p.get("use_prompt_cache", True)))

        self.spCtx.setValue(int(p.get("ctx", 15000)))
        self.spBatch.setValue(int(p.get("batch", 256)))
        self.spUBatch.setValue(int(p.get("ubatch", 64)))
        self.spPar.setValue(int(p.get("parallel", 1)))
        self.spNGL.setValue(int(p.get("ngl", 999)))
        self.edHost.setText(p.get("host", "127.0.0.1"))
        try: self.spPort.setValue(int(p.get("port", 8080)))
        except Exception: pass
        # константы
        self.spFps.setValue(float(p.get("fps", 1.0)))
        self.spDelta.setValue(float(p.get("min_thumb_delta", 0.01)))
        self.cbSmooth.setChecked(p.get("render_mode", "smooth") == "smooth")
        self.spHandle.setValue(int(p.get("handle", 15)))
        # окно
        want = p.get("window_title", "")
        if want:
            i = self.cbWindow.findText(want)
            if i >= 0: self.cbWindow.setCurrentIndex(i)
        # system override (если был сохранён)
        sys_ovr = p.get("system_override")
        if sys_ovr:
            llm_cfg.system_override = sys_ovr
        #регионы
        regions = p.get("regions")
        if regions:
            try:
                self.overlay.load_regions_from_prefs(regions)
            except Exception as e:
                print("[UI] load regions prefs error:", e)
            
    def _collect_prefs(self) -> dict:
        prefs = {
            "font_family": self.fontFamily.currentFont().family(),
            "font_size": int(self.fontSize.value()),
            "font_bold": bool(self.fontBold.isChecked()),
            "model_path": self.cbModel.currentText().strip(),
            "mmproj_path": self.cbMmproj.currentText().strip(),
            "lore_path": self.edLore.text().strip(),
            "max_tokens": int(self.spMaxTok.value()),
            "timeout_s": float(self.spTO.value()),
            "temp": float(self.spTemp.value()),
            "top_p": float(self.spTopP.value()),
            "fps": float(self.spFps.value()),
            "min_thumb_delta": float(self.spDelta.value()),
            "render_mode": "smooth" if self.cbSmooth.isChecked() else "instant",
            "handle": int(self.spHandle.value()),
            "window_title": self.cbWindow.currentText().strip(),
            "system_override": llm_cfg.system_override or "",
            "top_k": int(self.spTopK.value()),
            "repeat_penalty": float(self.spRP.value()),
            "seed": int(self.spSeed.value()),
            "slot_id": int(self.spSlot.value()),
            "use_prompt_cache": bool(self.cbCache.isChecked()),
            "ctx": int(self.spCtx.value()),
            "batch": int(self.spBatch.value()),
            "ubatch": int(self.spUBatch.value()),
            "parallel": int(self.spPar.value()),
            "ngl": int(self.spNGL.value()),
            "host": self.edHost.text().strip(),
            "port": int(self.spPort.value())
        }
        try:
            prefs["regions"] = self.overlay.dump_regions_for_prefs()
        except Exception as e:
            print("[UI] dump regions prefs error:", e)
        return prefs

    def _back_from_overlay(self):
        try:
            p = _load_prefs()
            p.update(self._collect_prefs())
            _save_prefs(p)
        except Exception as e:
            print("[UI] save on back error:", e)
        # Вызывается по Ctrl+Alt+Q из оверлея
        _stop_llama_server()
        self.showNormal(); self.activateWindow(); self._fill_windows()

    def _fill_windows(self):
        self.cbWindow.clear(); self._hwnd_map = {}
        try:
            wins = list_user_windows()
            for title, cls, pid, hwnd in wins:
                text=f"{title}  (PID {pid}, {cls})"
                self.cbWindow.addItem(text)
                self._hwnd_map[text]=hwnd
        except Exception as e:
            QMessageBox.warning(self,"Список окон",f"Ошибка: {e}")

    def _rescan_models(self):
        self.cbModel.clear(); self.cbMmproj.clear()
        roots = [BASE_DIR/"models"/"lmm", BASE_DIR/"models"/"llm"]
        models, projs = scan_gguf(roots)
        for p in models: self.cbModel.addItem(p)
        for p in projs:  self.cbMmproj.addItem(p)

    def _browse_lore(self):
        path,_ = QFileDialog.getOpenFileName(self,"Выберите файл лора", str(BASE_DIR), "Text (*.txt);;All (*.*)")
        if path:
            try:
                p = Path(path)
                try: rel = p.relative_to(BASE_DIR); self.edLore.setText(str(rel))
                except Exception: self.edLore.setText(str(p))
            except Exception:
                self.edLore.setText(path)

    def _edit_prompt(self):
        txt = getattr(llm_cfg, "system_override", "") or ""
        dlg = QDialog(self); dlg.setWindowTitle("System prompt — редактирование")
        lay = QVBoxLayout(dlg)
        te = QPlainTextEdit(txt); te.setMinimumSize(700,500); lay.addWidget(te)
        btn = QPushButton("Сохранить и закрыть"); lay.addWidget(btn)
        def _save():
            llm_cfg.system_override = te.toPlainText().strip() or None
            try:
                p = _load_prefs()
                p["system_override"] = llm_cfg.system_override or ""
                _save_prefs(p)
            except Exception:
                pass
            QMessageBox.information(dlg,"Промпт","Сохранено. Вступит в силу при следующем запросе.")
            dlg.close()
        btn.clicked.connect(_save)
        dlg.setModal(True); dlg.resize(760,540); dlg.exec()

    def _preview_system(self):
        # применить текущий путь лора в объект конфигурации
        lore_path = (self.edLore.text() or "").strip()
        if lore_path and not os.path.isabs(lore_path):
            lore_path = str((BASE_DIR / lore_path).resolve())
        llm_cfg.lore_path = lore_path
        reset_lore_cache()
        try:
            init_lore_once(llm_cfg)
        except Exception as e:
            print("[LLM] preview init error:", e)
        text = get_system_preview(llm_cfg) or "(empty)"
        dlg = QDialog(self); dlg.setWindowTitle("System prompt — предпросмотр")
        lay = QVBoxLayout(dlg)
        te  = QPlainTextEdit(text); te.setReadOnly(True); te.setMinimumSize(780,520)
        lay.addWidget(te)
        btn = QPushButton("Закрыть"); lay.addWidget(btn); btn.clicked.connect(dlg.close)
        dlg.setModal(True); dlg.resize(820,560); dlg.exec()

    def _start(self):
        # применяем настройки
        global MAX_OCR_FPS, MIN_THUMB_DELTA, RENDER_MODE, HANDLE, SERVER_EXE

        sel = self.cbWindow.currentText(); hwnd = getattr(self, "_hwnd_map", {}).get(sel)

        llm_cfg.max_tokens = int(self.spMaxTok.value())
        llm_cfg.timeout_s  = float(self.spTO.value())
        llm_cfg.temp       = float(self.spTemp.value())
        llm_cfg.top_p      = float(self.spTopP.value())
        llm_cfg.top_k          = int(self.spTopK.value())
        llm_cfg.repeat_penalty = float(self.spRP.value())
        llm_cfg.seed           = int(self.spSeed.value())
        llm_cfg.slot_id        = int(self.spSlot.value())
        llm_cfg.use_prompt_cache = bool(self.cbCache.isChecked())

        MAX_OCR_FPS    = float(self.spFps.value())
        MIN_THUMB_DELTA= float(self.spDelta.value())
        RENDER_MODE    = "smooth" if self.cbSmooth.isChecked() else "instant"
        HANDLE         = int(self.spHandle.value())

        # применим лор
        lore_path = (self.edLore.text() or "").strip()
        if lore_path and not os.path.isabs(lore_path):
            lore_path = str((BASE_DIR / lore_path).resolve())
        llm_cfg.lore_path = lore_path
        reset_lore_cache()
        
        host = (self.edHost.text() or "127.0.0.1").strip()
        port = int(self.spPort.value())
        llm_cfg.server = f"http://{host}:{port}"

        # команда сервера
        if SERVER_EXE is None:
            SERVER_EXE = _pick_server_exe()
        if SERVER_EXE is None or not SERVER_EXE.exists():
            QMessageBox.critical(self, "llama-server", "Не найден llama-server.exe в models\\lmm\\llama.cpp или models\\llm\\llama.cpp")
            print("[LLM] server exe not found in expected locations")
            return

        model_path  = self.cbModel.currentText().strip()
        mmproj_path = self.cbMmproj.currentText().strip()
        cmd = rebuild_server_cmd(
            SERVER_EXE, model_path, mmproj_path,
            host=host, port=port,
            ctx=int(self.spCtx.value()),
            batch=int(self.spBatch.value()),
            ubatch=int(self.spUBatch.value()),
            parallel=int(self.spPar.value()),
            ngl=int(self.spNGL.value()),
        )
        print("[LLM] server exe:", SERVER_EXE)
        print("[LLM] model:", model_path)
        print("[LLM] mmproj:", mmproj_path)
        try:
            _save_prefs(self._collect_prefs())
        except Exception as e:
            print("[UI] save on start error:", e)

        # спрячем конфиг, запустим сервер и прогрев, затем включим воркер
        self.hide()
        _ensure_llama_server(cmd, llm_cfg.server)
        try:
            if llm_cfg.enabled and llm_cfg.use_prompt_cache:
                preload_prompt_cache(llm_cfg)
        except Exception as e:
            print("[LLM] warmup error:", e)

        # шрифт панели
        self.overlay.font_family = self.fontFamily.currentFont().family()
        self.overlay.font_size   = int(self.fontSize.value())
        self.overlay.font_bold   = bool(self.fontBold.isChecked())
        if hwnd: self.overlay.bound_hwnd = hwnd

        # запускаем воркер и скрываем режим правки
        self.overlay.start_worker()
        self.overlay.edit_mode = False
        self.overlay.update()
        try:
            _save_prefs(self._collect_prefs())
        except Exception as e:
            print("[UI] save on start error:", e)
            
    def closeEvent(self, e):
        try:
            _save_prefs(self._collect_prefs())
        except Exception as ex:
            print("[UI] save on close error:", ex)
        super().closeEvent(e)


# ========================= main ==========================

def main():
    app = QApplication(sys.argv)
    w = ConfigWindow()
    w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
