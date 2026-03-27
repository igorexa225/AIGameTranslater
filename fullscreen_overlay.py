# fullscreen_overlay.py — Полностью переписанный модуль полноэкранного перевода
# Принцип: Максимальная изоляция от основной логики, стабильность, отсутствие вылетов.

import sys, time, ctypes, base64, re
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import cv2
from difflib import SequenceMatcher

from PySide6.QtCore import (Qt, QRect, QThread, Signal, QObject, QTimer, QRunnable, QThreadPool, QAbstractNativeEventFilter)
from PySide6.QtGui  import (QPainter, QColor, QFont, QKeySequence, QShortcut, QGuiApplication, QFontMetrics)
from PySide6.QtWidgets import (QWidget, QApplication)

import win32gui, win32ui, win32con, win32api
import ctypes.wintypes as wt

DEBUG_MODE = True  # Включаем сохранение отладочных картинок

# --- Импорты адаптеров (для доступа к нейросетям) ---
import online_adapter
from llm_adapter import vision_translate_from_images
try:
    from llm_adapter_dual import (
        extract_en_from_image,
        translate_en_to_ru_text,
    )
except Exception:
    pass

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
        _EASYOCR_READER = easyocr.Reader(langs, gpu=False, model_storage_directory=str(model_dir))
        _CURRENT_LANG = lang
    except Exception as e:
        print(f"[FULLSCREEN] EasyOCR init failed: {e}")
        _EASYOCR_READER = None
    return _EASYOCR_READER

# ====================== UI: Панель текста ======================

class TextPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(None)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
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
        self.generation = 0  # Версия текущего текста

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

    def update_size(self):
        """Пересчитываем высоту панели, чтобы текст влезал."""
        if not self.text: return
        
        f = QFont(self.font_family, self.font_size)
        f.setBold(self.font_bold)
        fm = QFontMetrics(f)
        
        lines = self.text.split('\n')
        
        # 1. Рассчитываем необходимую ширину
        max_line_w = 0
        for line in lines:
            lw = fm.horizontalAdvance(line)
            if lw > max_line_w: max_line_w = lw
            
        padding = 16
        needed_w = max_line_w + padding
        # Ширина: не меньше оригинала, но может быть больше (до 1000px, было 600)
        new_w = max(self.ocr_w, needed_w)
        new_w = min(new_w, 1000)
        
        # 2. Рассчитываем высоту с учетом новой ширины
        text_w = new_w - padding
        total_h = 8 # top padding
        for line in lines:
            if not line.strip():
                total_h += fm.height()
                continue
            r = fm.boundingRect(0, 0, text_w, 10000, Qt.TextWordWrap | Qt.AlignLeft, line)
            total_h += r.height() + 4 # Добавляем 4px отступа между параграфами
            
        total_h += 8 # bottom padding
        
        # Высота не меньше оригинального бокса, но может быть больше
        new_h = max(self.ocr_h, total_h)
        if new_w != self.width() or new_h != self.height():
            self.resize(new_w, new_h)
            self.apply_acrylic()

    def paintEvent(self, _e):
        if not self.text: return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        
        f = QFont(self.font_family, self.font_size)
        f.setBold(self.font_bold)
        p.setFont(f)
        
        rect = self.rect()
        
        # 1. Общий фон панели (если выбран solid)
        if self.bg_mode == "solid":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, self.bg_alpha))
            p.drawRoundedRect(rect, 4, 4)
        
        # 2. Фон под строками (Box BG)
        lines = self.text.split('\n')
        fm = p.fontMetrics()
        current_y = rect.y() + 8 # отступ сверху
        
        for line in lines:
            if not line.strip():
                current_y += fm.height()
                continue
                
            # Вычисляем прямоугольник текста
            line_rect = fm.boundingRect(rect.x() + 8, current_y, rect.width() - 16, 10000, Qt.TextWordWrap | Qt.AlignLeft, line)
            
            # Рисуем подложку под строкой
            if self.box_bg_mode == "solid" and self.box_bg_alpha > 0:
                pad_x, pad_y = 4, 2
                bg_rect = line_rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, self.box_bg_alpha))
                p.drawRoundedRect(bg_rect, 4, 4)
            
            # Тень текста
            p.setPen(QColor(0, 0, 0, 220))
            p.drawText(line_rect.translated(1, 1), Qt.TextWordWrap | Qt.AlignLeft, line)
            
            # Сам текст
            p.setPen(QColor(255, 255, 255, 255))
            p.drawText(line_rect, Qt.TextWordWrap | Qt.AlignLeft, line)
            
            current_y += line_rect.height() + 4 # Учитываем отступ при отрисовке

# ====================== Async Translation Task ======================

class TranslationSignals(QObject):
    result = Signal(int, str, int) # rid, text, generation

class TranslationTask(QRunnable):
    def __init__(self, rid, crop_bgr, overlay, ocr_text, generation):
        super().__init__()
        self.rid = rid
        self.crop_bgr = crop_bgr
        self.overlay = overlay
        self.ocr_text = ocr_text
        self.generation = generation
        self.signals = TranslationSignals()

    def run(self):
        try:
            b64 = _bgr_to_png_b64(self.crop_bgr)
            mode = self.overlay.work_mode_idx # 0=SOLO, 1=DUAL, 2=ONLINE
            ru_text = ""
            
            if mode == 0: # SOLO
                # Принудительно ограничиваем токены для стабильности
                cfg = self.overlay.llm_cfg
                old_tok = cfg.max_tokens
                cfg.max_tokens = 1024 
                _, ru_text = vision_translate_from_images(b64, None, cfg)
                cfg.max_tokens = old_tok
                
            elif mode == 1: # DUAL
                # OCR
                en = extract_en_from_image(b64, self.overlay.ocr_cfg_dual)
                if en:
                    # TR
                    cfg = self.overlay.tr_cfg_dual
                    old_tok = cfg.max_tokens
                    cfg.max_tokens = 1024
                    ru_text = translate_en_to_ru_text(en, cfg)
                    cfg.max_tokens = old_tok
                    
            elif mode == 2: # ONLINE
                # Просто переводим текст от EasyOCR (быстро и дешево)
                ru_text = online_adapter.translate_text(self.ocr_text)
            
            if ru_text:
                self.signals.result.emit(self.rid, ru_text, self.generation)
                
        except Exception as e:
            print(f"[FULLSCREEN] Task error: {e}")

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
            if hid == 101: self.overlay.quit_app()     # Ctrl+Alt+Q
            elif hid == 102: self.overlay.toggle_pause() # Ctrl+Alt+F1
            return True, 0
        return False, 0

# ====================== Worker Logic ======================

class FullscreenCaptureWorker(QThread):
    # Сигналы для управления GUI из потока (БЕЗОПАСНО)
    create_panel = Signal(int, int, int, int, int) # id, x, y, w, h
    update_panel_pos = Signal(int, int, int, int, int)
    update_panel_text = Signal(int, str, int) # rid, text, generation
    update_panel_gen = Signal(int, int)       # rid, generation (сброс текста)
    delete_panel = Signal(int)
    
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self._stop = False
        self.active_regions = {} # {id: {'rect': (x,y,w,h), 'text': str, 'trans': str}}
        self.next_id = 0
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(3) # Ограничиваем кол-во одновременных переводов

    def stop(self):
        self._stop = True

    def run(self):
        print("[FULLSCREEN] Worker started")
        while not self._stop:
            # Пауза
            if self.overlay.paused:
                time.sleep(0.2)
                continue

            # Используем настройку FPS из меню
            delay = 1.0 / max(0.1, self.overlay.capture_fps)
            time.sleep(delay)
            
            hwnd = self.overlay.bound_hwnd
            if not hwnd or not win32gui.IsWindow(hwnd):
                continue
                
            # 1. Захват
            img, win_rect = _grab_full_window(hwnd)
            if img is None: continue
            
            # 2. EasyOCR (Детекция + Текст)
            reader = get_easyocr(self.overlay.source_lang)
            if not reader: continue
            
            try:
                # Уменьшаем для скорости
                h, w = img.shape[:2]
                scale = 0.7
                small = cv2.resize(img, (0,0), fx=scale, fy=scale)
                
                results = reader.readtext(small, paragraph=False)

            except Exception as e:
                print(f"[FULLSCREEN] OCR Error: {e}")
                continue
                
            # 3. Обработка результатов
            # Сначала собираем все кандидаты в список
            candidates = []
            for bbox, text, conf in results:
                if conf < 0.3 or not text.strip(): continue
                
                # Фильтр мелочи
                if (bbox[2][0] - bbox[0][0]) < 15 or (bbox[2][1] - bbox[0][1]) < 10: continue

                tl = bbox[0]
                br = bbox[2]
                x = int(tl[0] / scale)
                y = int(tl[1] / scale)
                bw = int((br[0] - tl[0]) / scale)
                bh = int((br[1] - tl[1]) / scale)
                
                candidates.append({
                    'x': x, 'y': y, 'w': bw, 'h': bh,
                    'text': text, 'conf': conf
                })

            # Фильтруем перекрытия (оставляем только мелкие/точные боксы)
            filtered_candidates = self._filter_overlapping_boxes(candidates)

            current_frame_ids = set()
            
            for item in filtered_candidates:
                x, y, bw, bh = item['x'], item['y'], item['w'], item['h']
                text = item['text']
                
                # Ищем совпадение с существующими регионами
                matched_id = -1
                for rid, rdata in self.active_regions.items():
                    rx, ry, rw, rh = rdata['rect']
                    # Центры
                    cx1, cy1 = x + bw/2, y + bh/2
                    cx2, cy2 = rx + rw/2, ry + rh/2
                    dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
                    
                    if dist < 20: # Уменьшил с 50 до 20, чтобы боксы не прыгали на соседние слова
                        matched_id = rid
                        break
                
                if matched_id != -1:
                    # Обновляем существующий
                    rdata = self.active_regions[matched_id]
                    current_frame_ids.add(matched_id)
                    
                    # Обновляем позицию
                    rdata['rect'] = (x, y, bw, bh)
                    self.update_panel_pos.emit(matched_id, x, y, bw, bh)
                    
                    # --- ПРОВЕРКА 1: Изменилась ли картинка? (Image Hash) ---
                    # Это предотвращает дерганье перевода, если OCR "скачет", а картинка статична
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(img.shape[1], x+bw), min(img.shape[0], y+bh)
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        # Считаем хэш (уменьшенная копия)
                        small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_LINEAR)
                        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                        
                        last_hash = rdata.get('last_hash')
                        is_same_image = False
                        
                        if last_hash is not None:
                            # MSE (Mean Squared Error)
                            err = np.sum((small_gray.astype("float") - last_hash.astype("float")) ** 2)
                            err /= float(small_gray.shape[0] * small_gray.shape[1])
                            if err < 50: # Порог схожести
                                is_same_image = True
                        
                        rdata['last_hash'] = small_gray
                        
                        if is_same_image:
                            # Картинка не изменилась -> считаем текст стабильным
                            rdata['stability'] = rdata.get('stability', 0) + 1
                            if rdata['stability'] == 1: # Переводим только когда стабилизировалось (1 кадр выдержки)
                                print(f"[FULLSCREEN] Image stable, translating: {rdata['text'][:30]}...")
                                self._queue_translation(matched_id, img, (x,y,bw,bh), rdata['text'], rdata.get('generation', 0))
                            continue 

                    # --- ПРОВЕРКА 2: Изменился ли текст? ---
                    sim = SequenceMatcher(None, text, rdata['text']).ratio()
                    if sim < 0.85: # Текст изменился
                        print(f"[FULLSCREEN] Content changed: {rdata['text']} -> {text}")
                        rdata['text'] = text
                        rdata['stability'] = 0 # Сброс стабильности (не переводим пока не устоится)
                        rdata['generation'] = rdata.get('generation', 0) + 1
                        self.update_panel_gen.emit(matched_id, rdata['generation'])
                    else:
                        # Текст тот же -> повышаем стабильность
                        rdata['stability'] = rdata.get('stability', 0) + 1
                        if rdata['stability'] == 1: # 1 кадр подтверждения
                            print(f"[FULLSCREEN] Text stable, translating: {text[:30]}...")
                            self._queue_translation(matched_id, img, (x,y,bw,bh), text, rdata.get('generation', 0))
                else:
                    # Новый регион
                    new_id = self.next_id
                    self.next_id += 1
                    
                    self.active_regions[new_id] = {
                        'rect': (x, y, bw, bh),
                        'text': text,
                        'trans': "",
                        'last_hash': None,
                        'stability': 0, # Новый регион всегда нестабилен
                        'generation': 0
                    }
                    current_frame_ids.add(new_id)
                    
                    # Инициализируем хэш для нового региона
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(img.shape[1], x+bw), min(img.shape[0], y+bh)
                    if x2 > x1 and y2 > y1:
                        small = cv2.resize(img[y1:y2, x1:x2], (32, 32), interpolation=cv2.INTER_LINEAR)
                        self.active_regions[new_id]['last_hash'] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                    # Создаем панель
                    self.create_panel.emit(new_id, x, y, bw, bh)
                    # self._queue_translation(...) # УБРАНО: Не переводим сразу, ждем стабильности!
            
            # 4. Удаление старых
            to_delete = []
            for rid in self.active_regions:
                if rid not in current_frame_ids:
                    to_delete.append(rid)
            
            for rid in to_delete:
                self.delete_panel.emit(rid)
                del self.active_regions[rid]

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

    def _queue_translation(self, rid, full_img, rect, ocr_text, generation):
        """Добавляет задачу на перевод в пул потоков."""
        x, y, w, h = rect
        # Crop with margin
        h_img, w_img = full_img.shape[:2]
        # Уменьшаем отступы, чтобы не захватывать соседние строки!
        x1 = max(0, x - 2)
        y1 = max(0, y)     # Убираем вертикальный отступ полностью (0px)
        x2 = min(w_img, x + w + 2)
        y2 = min(h_img, y + h) # Убираем вертикальный отступ полностью (0px)
        
        crop = full_img[y1:y2, x1:x2]
        if crop.size == 0: return
        
        # --- DEBUG: Сохраняем то, что уходит на перевод ---
        if DEBUG_MODE:
            try:
                d_dir = BASE_DIR / "debug_crops"
                d_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d_dir / f"crop_{rid}_{int(time.time()*100)}.png"), crop)
            except Exception: pass

        # Создаем задачу
        task = TranslationTask(rid, crop.copy(), self.overlay, ocr_text, generation)
        task.signals.result.connect(self.update_panel_text) # Через сигнал воркера
        self.pool.start(task)

# ====================== Main Overlay Class ======================

class FullscreenOverlay(QWidget):
    def __init__(self, on_quit=None):
        super().__init__()
        self.setWindowTitle("Fullscreen Translator (Plugin Mode)")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        
        # Config placeholders (will be set by main app)
        self.llm_cfg = None
        self.ocr_cfg_dual = None
        self.tr_cfg_dual = None
        self.source_lang = "en"
        self.work_mode_idx = 0
        self.capture_fps = 1.0  # Дефолтное значение
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
        # 101 = Ctrl+Alt+Q (Quit)
        RegisterHotKey(hwnd, 101, 0x0002 | 0x0001, 0x51) 
        # 102 = Ctrl+Alt+F1 (Toggle Pause)
        RegisterHotKey(hwnd, 102, 0x0002 | 0x0001, 0x70)

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
            p.setVisible(not self.paused)

    # --- Slots for Worker ---
    def on_create_panel(self, pid, x, y, w, h):
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
        
        p.setGeometry(x, y, w, h)
        p.show()
        p.apply_acrylic()
        self.panels[pid] = p
        if self.paused: p.hide()
        
    def on_update_pos(self, pid, x, y, w, h):
        if pid in self.panels:
            p = self.panels[pid]
            p.ocr_h = h  # Обновляем базовую высоту
            p.ocr_w = w  # Обновляем базовую ширину
            p.setGeometry(x, y, w, h)
            p.update_size() # Проверяем, влезает ли текущий текст
            
    def on_update_gen(self, pid, gen):
        """Текст изменился -> сбрасываем старый перевод."""
        if pid in self.panels:
            self.panels[pid].generation = gen
            self.panels[pid].text = "" # Очищаем, чтобы не висел старый текст
            self.panels[pid].update()

    def on_update_text(self, pid, text, gen):
        if pid in self.panels and text:
            # Принимаем перевод ТОЛЬКО если поколение совпадает
            if self.panels[pid].generation == gen:
                p = self.panels[pid]
                p.text = text
                p.update_size() # Ресайзим под новый текст
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
