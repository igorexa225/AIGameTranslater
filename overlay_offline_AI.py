# overlay_offline_AI.py — Конфиг + оверлей (две картинки → перевод через llama.cpp)
# Требует: PySide6, pywin32, numpy, opencv-python, requests
# Запуск: python overlay_offline_AI.py

import sys, os, time, re, subprocess, base64, difflib, ctypes, json, requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import cv2
from difflib import SequenceMatcher

from PySide6.QtCore import (Qt, QRect, QPoint, QTimer, QEvent, QThread, Signal, QAbstractNativeEventFilter, QSize)
from PySide6.QtGui  import (QPainter, QColor, QPen, QFont, QKeySequence, QShortcut, QGuiApplication, QImage, QIcon)
from PySide6.QtWidgets import (QApplication, QWidget, QMessageBox, QDialog,
    QLabel, QPushButton, QComboBox, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFontComboBox, QFileDialog, QPlainTextEdit, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QScrollArea, QTreeWidget, QTreeWidgetItem, QProgressBar, QHeaderView,
    QSlider, QStackedWidget, QToolButton, QFrame)

import win32con, win32gui, win32ui, win32api, win32process
import ctypes.wintypes as wt

#========================== Вызов EasyOCR ==============================

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None
    _EASYOCR_AVAILABLE = False

_easyocr_reader = None

def get_easyocr_reader():
    """Ленивое создание EasyOCR-ридера (только на CPU, gpu=False)."""
    global _easyocr_reader
    if not _EASYOCR_AVAILABLE:
        print("[OCR-BOXES] EasyOCR not available (import failed)")
        return None
    if _easyocr_reader is None:
        try:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print("[OCR-BOXES] EasyOCR reader initialised (gpu=False)")
        except Exception as e:
            print("[OCR-BOXES] EasyOCR init error:", e)
            _easyocr_reader = None
    return _easyocr_reader

# ======================== CONFIG (по умолчанию) =========================

# захват/обновление
MAX_OCR_FPS       = 1.0        # 0.5–2.0 кадров/сек

# UI/поведение
RENDER_MODE       = "smooth"   # "smooth" | "instant"
HANDLE            = 15
BORDER_WIDTH      = 4
SPLIT_BORDER_WIDTH = 4
REQUIRE_BOUND_WINDOW = True
DRAW_OVER_ORIGINAL = False
PANEL_BG_MODE  = "blur"   # "blur" | "solid" | "none"
PANEL_BG_ALPHA = 64       # 0–255, для blur = альфа tint'а, для solid = прозрачность чёрного фона
BOX_BG_MODE  = "solid"    # "panel" | "solid" | "none"
BOX_BG_ALPHA = 180        # 0–255, для solid = прозрачность чёрной подложки под строкой

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

try:
    from llm_adapter_dual import (
        LLMConfig as LLMConfigDual,
        extract_en_from_image,
        translate_en_to_ru_text,
        preload_prompt_cache_ocr,
        preload_prompt_cache_tr,
        reset_tr_system_cache,
        _split_name_and_body,
    )
except Exception:
    LLMConfigDual = None

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
_EN_WS_RX = re.compile(r"\s+")

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


def canon_en(s: str) -> str:
    s = (s or "").replace("\ufeff", "")
    # нормализуем пробелы и регистр
    s = _EN_WS_RX.sub(" ", s)
    s = s.strip().lower()
    if not s:
        return ""
    # выкидываем пунктуацию
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # убираем простые артикли, которые часто "прыгают" в OCR/LLM
    tokens = [t for t in s.split() if t not in ("a", "an", "the")]
    return " ".join(tokens)

def detect_text_boxes(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
    """
    Детектор прямоугольников с текстом на EasyOCR.
    Возвращает список (x, y, w, h, text) в пикселях.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    h, w = frame_bgr.shape[:2]
    if h < 10 or w < 10:
        return []

    reader = get_easyocr_reader()
    if reader is None:
        return []

    try:
        # EasyOCR ожидает RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # detail=1 -> bbox + text + conf, paragraph=False -> отдельные строки
        results = reader.readtext(
            rgb,
            detail=1,
            paragraph=False,
        )

        boxes: List[Tuple[int, int, int, int, str]] = []
        # отсечём совсем мелкий шум: ~0.3% площади региона
        min_area = (w * h) * 0.003

        for res in results:
            # формат результата: (bbox, text, conf) или [bbox, text, conf]
            if not res or len(res) < 2:
                continue

            bbox = res[0]
            text = str(res[1] or "").strip()
            if not text:
                # пустой текст нам не нужен
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            bw, bh = x1 - x0, y1 - y0

            if bw < 5 or bh < 5:
                continue

            area = bw * bh
            if area < min_area:
                continue

            # фильтр по форме:
            # сильно "высокие и узкие" боксы (почти вертикальные) выкидываем,
            # чтобы меньше цеплять перила/части одежды
            aspect = bw / float(bh) if bh > 0 else 999.0
            if aspect < 1.2 and bw < w * 0.4:
                continue

            boxes.append((int(x0), int(y0), int(bw), int(bh), text))

        return boxes

    except Exception as e:
        print("[OCR-BOXES] detect_text_boxes_easyocr error:", e)
        return []

def _filter_boxes_by_llm_text(
    boxes_with_text: List[Tuple[int, int, int, int, str]],
    en_full: str,
) -> List[Tuple[int, int, int, int]]:
    """
    Пытаемся выбрать только те боксы, которые соответствуют основному диалогу
    из LLM-OCR (en_full), и выкинуть имя / мусор.

    boxes_with_text: [(x, y, w, h, text), ...]
    возвращаем:      [(x, y, w, h), ...]
    """

    if not boxes_with_text:
        return []

    en_full = en_full or ""

    # 1) Разделяем полный текст на NAME + BODY той же логикой, что и в переводчике
    try:
        name_line, body = _split_name_and_body(en_full)
    except Exception:
        name_line, body = None, en_full

    name_c = canon_en(name_line) if name_line else ""
    body_c = canon_en(body) if body else canon_en(en_full)

    if not body_c:
        # если тело пустое — лучше вернуть все боксы как есть
        return [(x, y, w, h) for (x, y, w, h, _t) in boxes_with_text]

    first_body_word = body_c.split()[0] if body_c.split() else ""

    good: List[Tuple[int, int, int, int]] = []

    for (x, y, w, h, txt) in boxes_with_text:
        t = (txt or "").strip()
        if not t:
            continue

        t_c = canon_en(t)

        # 2) Если этот бокс очень похож на строку NAME — выкидываем его
        if name_c:
            r_name = SequenceMatcher(None, t_c, name_c).ratio()
            if r_name >= 0.8:
                # почти точно nameplate
                continue

        # 3) Короткие подписи (1–3 слова, без точки и т.п.) — потенциальный мусор,
        #    но если они явно совпадают с началом BODY, считаем диалогом
        is_short_label = (
            len(t_c) > 0
            and len(t_c.split()) <= 3
            and not any(ch in t_c for ch in ".!?:")
        )

        prefix_match = False
        if first_body_word:
            if t_c.startswith(first_body_word):
                prefix_match = True
            else:
                fw = t_c.split()[0]
                if fw == first_body_word:
                    prefix_match = True

        # 4) Похож ли текст бокса на начало основного текста?
        body_prefix = body_c[: len(t_c) + 16]
        ratio = SequenceMatcher(None, t_c, body_prefix).ratio()

        if prefix_match or ratio >= 0.6:
            # это кусок основного диалога
            good.append((x, y, w, h))
        else:
            # если это короткий лейбл и он не совпал с началом BODY — скорее всего имя/мусор
            if not is_short_label:
                # длинную строку всё-таки лучше не терять вообще
                good.append((x, y, w, h))

    if not good:
        # на всякий случай, если всё отфильтровали — не ломаем поведение
        return [(x, y, w, h) for (x, y, w, h, _t) in boxes_with_text]

    good.sort(key=lambda b: b[1])  # стабильный порядок по вертикали
    return good

def _bgr_to_png_b64(img, max_side: int = 1400) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side/float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".png", img)
    if not ok: return ""
    return base64.b64encode(enc).decode("ascii")

#====================== Загрузчик =======================

class DownloadWorker(QThread):
    progress = Signal(int)         # %
    status   = Signal(str)         # текст статуса
    done     = Signal(bool, str)   # ok, msg

    def __init__(self, url: str, dest_path: str, chunk: int = 1<<20):
        super().__init__()
        self.url = url
        self.dest_path = dest_path
        self.chunk = chunk
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.status.emit("Запрос...")
            with requests.get(self.url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                tmp = self.dest_path + ".part"
                got = 0
                with open(tmp, "wb") as f:
                    for ch in r.iter_content(chunk_size=self.chunk):
                        if self._cancel:
                            self.status.emit("Отменено")
                            try: os.remove(tmp)
                            except: pass
                            self.done.emit(False, "Отменено")
                            return
                        if ch:
                            f.write(ch)
                            got += len(ch)
                            if total:
                                self.progress.emit(int(got * 100 / total))
                os.replace(tmp, self.dest_path)
            self.progress.emit(100)
            self.done.emit(True, "Готово")
        except Exception as e:
            self.done.emit(False, f"Ошибка: {e}")

#==================== Окно загрузчика =========================

class ModelHubDialog(QDialog):
    def __init__(self, parent=None, default_models_dir:str=""):
        super().__init__(parent)
        self.setWindowTitle("Скачать модели (GGUF / mmproj)")
        self.setModal(True)
        self.resize(1260, 520)

        self.default_dir = default_models_dir or str((BASE_DIR / "models" / "llm").resolve())
        os.makedirs(self.default_dir, exist_ok=True)

        lay = QVBoxLayout(self)

        # Пресеты
        gb = QGroupBox("Рекомендованные пресеты")
        v = QVBoxLayout(gb)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Название", "Тип", "Формат", "Примечание", "Размер"])
        v.addWidget(self.tree)
        lay.addWidget(gb)

        # — наполняем пресеты (URL — примеры-плейсхолдеры; подставьте свои)
        presets = [
            # OCR / VL
            {"name": "Qwen3-VL-2B-Instruct (OCR)", "kind": "OCR (VL)", "fmt": "GGUF Q4_K_M", "note":"Vision-модель для OCR (распознавалка текста)", "size": "~1.1 GB",
             "url":"https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3-VL-2B-Instruct-Q4_K_M.gguf?download=true", "filename":"Qwen3-VL-2B-Instruct (OCR).gguf"},
            {"name": "Qwen3-VL-2B-mmproj", "kind": "mmproj", "fmt": "F16/BF16", "note":"Проектор к Qwen-VL-2B (название будет просто mmproj-F16!!!) (необходим для Qwen-B2)", "size": "~800 MB",
             "url":"https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-F16.gguf?download=true", "filename":"Qwen3-VL-2B-mmproj.gguf"},
            # Translator / TEXT
            {"name": "Qwen3-4B-Instruct", "kind": "Translator (TEXT)", "fmt": "GGUF Q4_K_M", "note":"Только текст для дуал режима", "size": "~2.5 GB",
             "url":"https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf?download=true", "filename":"Qwen3-4B-Instruct.gguf"},
            {"name": "Qwen3-4B-Instruct", "kind": "Translator (TEXT)", "fmt": "GGUF Q4_NL", "note":"Только текст для дуал режима (ТОЛЬКО ДЛЯ RTX 40 СЕРИИ)", "size": "~2.4 GB",
             "url":"https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-IQ4_NL.gguf?download=true", "filename":"Qwen3-4B-Instruct.gguf"},
            {"name": "Hunyuan-MT-7B-GGUF", "kind": "Translator (TEXT)", "fmt": "GGUF Q4_K_M", "note":"Хорошее алтернатива между качеством и качеством). Переводит очень хорошо, читать приятно, но абсолютно клал болт на лор и фразбук", "size": "~4.6 GB",
             "url":"https://huggingface.co/mradermacher/Hunyuan-MT-7B-GGUF/resolve/main/Hunyuan-MT-7B.Q4_K_M.gguf?download=true", "filename":"Hunyuan-MT-7B-GGUF.gguf"},
            # Всё в одном
            {"name": "Qwen3-8B-VL-Instruct", "kind": "Translator/OCR (All in one)", "fmt": "GGUF Q4_K_M", "note":"Модель для соло режима (И читает и переводит)", "size": "~5 GB",
             "url":"https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf?download=true", "filename":"Qwen3-8B-VL-Instruct.gguf"},
            {"name": "Qwen3-8B-VL-Instruct", "kind": "Translator/OCR (All in one)", "fmt": "GGUF Q4_NL", "note":"Модель для соло режима (И читает и переводит)(ТОЛЬКО ДЛЯ RTX 40 СЕРИИ)", "size": "~4.8 GB",
             "url":"https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf?download=true", "filename":"Qwen3-8B-VL-Instruct.gguf"},
            {"name": "Qwen3-8B-VL-Instruct-mmproj", "kind": "Translator/OCR (All in one)", "fmt": "GGUF F16", "note":"Гляделка для модели 8B", "size": "~1.2 GB",
             "url":"https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-F16.gguf?download=true", "filename":"Qwen3-8B-VL-Instruct-mmproj.gguf"},
            {"name": "Qwen3-4B-VL-Instruct", "kind": "Translator/OCR (All in one)", "fmt": "GGUF Q4_K_M", "note":"Альтернатива для соло 8B (лечге и хуже по качеству)(ТОЛЬКО ДЛЯ RTX 40 СЕРИИ)", "size": "~2.4 GB",
             "url":"https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3-VL-4B-Instruct-Q4_K_M.gguf?download=true", "filename":"Qwen3-4B-VL-Instruct.gguf"},
            {"name": "Qwen3-4B-VL-Instruct", "kind": "Translator/OCR (All in one)", "fmt": "GGUF Q4_NL", "note":"Альтернатива для соло 8B (лечге и хуже по качеству)", "size": "~2.5 GB",
             "url":"https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3-VL-4B-Instruct-IQ4_NL.gguf?download=true", "filename":"Qwen3-4B-VL-Instruct.gguf"},
            {"name": "Qwen3-4B-VL-Instruct-mmproj", "kind": "Translator/OCR (All in one)", "fmt": "GGUF F16", "note":"Гляделка для 4B", "size": "~800 MB",
             "url":"https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-F16.gguf?download=true", "filename":"Qwen3-4B-VL-Instruct-mmproj.gguf"},
        ]
        self._presets = presets
        for p in presets:
            it = QTreeWidgetItem([p["name"], p["kind"], p["fmt"], p["note"], p["size"]])
            it.setData(0, Qt.UserRole, p)
            self.tree.addTopLevelItem(it)
            self._autosize_columns()
            self.tree.setUniformRowHeights(True)                 # быстрее
            self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Поля URL/путь
        form = QFormLayout()
        self.edUrl = QLineEdit()
        self.edPath = QLineEdit(os.path.join(self.default_dir, ""))  # папка
        self.btnBrowse = QPushButton("…")
        row = QHBoxLayout()
        row.addWidget(self.edPath, 1)
        row.addWidget(self.btnBrowse)
        form.addRow("URL файла:", self.edUrl)
        form.addRow("Папка сохранения:", QWidget())
        form.itemAt(form.rowCount()-1, QFormLayout.FieldRole).widget().setLayout(row)
        lay.addLayout(form)

        # Кнопки действий
        row2 = QHBoxLayout()
        self.btnDl   = QPushButton("Скачать")
        self.btnClose= QPushButton("Закрыть")
        row2.addStretch(1)
        row2.addWidget(self.btnDl)
        row2.addWidget(self.btnClose)
        lay.addLayout(row2)

        # Прогресс
        self.pb = QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)
        self.lab = QLabel("")
        lay.addWidget(self.pb)
        lay.addWidget(self.lab)

        # Сигналы
        self.btnBrowse.clicked.connect(self._choose_dir)
        self.btnDl.clicked.connect(self._start_download)
        self.btnClose.clicked.connect(self.reject)
        # Автоподстановка URL при выборе пресета
        self.tree.itemSelectionChanged.connect(self._fill_from_selected)

        self._worker: DownloadWorker|None = None
        
    def _autosize_columns(self):
        h = self.tree.header()
        h.setStretchLastSection(False)
        h.setSectionsMovable(True)
        h.setMinimumSectionSize(80)

        # 0: Название, 1: Тип, 2: Формат, 3: Примечание, 4: Размер
        # сначала поджимаем по содержимому
        for i in range(self.tree.columnCount()):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        # чуть «подпухлить» узкие колонки и ограничить сверхширокие
        QTimer.singleShot(0, lambda: self._finalize_columns())

    def _finalize_columns(self):
        # немного паддинга и ограничение максимальной ширины узких колонок
        for i in (0, 1, 2, 4):
            w = min(self.tree.sizeHintForColumn(i) + 24, 600)
            self.tree.setColumnWidth(i, w)

        # "Примечание" растягиваем на оставшееся место
        h = self.tree.header()
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        # чтобы пользователь мог потом руками двигать
        for i in range(self.tree.columnCount()):
            if i != 3:
                h.setSectionResizeMode(i, QHeaderView.Interactive)

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Куда сохранить", self.edPath.text() or self.default_dir)
        if d:
            self.edPath.setText(d)

    def _fill_from_selected(self):
        it = self.tree.currentItem()
        if not it:
            QMessageBox.information(self, "Модели", "Выберите пресет в списке.")
            return
        p = it.data(0, Qt.UserRole) or {}
        self.edUrl.setText(p.get("url",""))
        # Папку не трогаем — пользователь сам управляет

    def _start_download(self):
        url = (self.edUrl.text() or "").strip()
        if not url:
            QMessageBox.warning(self, "Модели", "Введите URL модели или выберите пресет.")
            return
        folder = (self.edPath.text() or "").strip() or self.default_dir
        os.makedirs(folder, exist_ok=True)
        # имя берём из URL
        fname = url.split("/")[-1].split("?")[0] or "model.gguf"
        dest = os.path.join(folder, fname)

        self.pb.setValue(0)
        self.lab.setText("Подключение...")
        self._worker = DownloadWorker(url, dest)
        self._worker.progress.connect(self.pb.setValue)
        self._worker.status.connect(self.lab.setText)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, ok: bool, msg: str):
        self.lab.setText(msg)
        if ok:
            QMessageBox.information(self, "Модели", "Загрузка завершена.")

            # после успешной загрузки попробуем обновить список моделей в родительском окне
            parent = self.parent()
            if parent is not None:
                for name in ("_rescan_models", "_scan_models", "_scan_models_ui", "_rebuild_models_list"):
                    if hasattr(parent, name):
                        try:
                            getattr(parent, name)()
                            break
                        except Exception as e:
                            print("[ModelHub] rescan error:", e)
        else:
            QMessageBox.critical(self, "Модели", msg)

# ================= Windows blur / PrintWindow =============

_HRESULT = getattr(wt, "HRESULT", ctypes.c_long)
class ACCENT_POLICY(ctypes.Structure):
    _fields_=[("AccentState",ctypes.c_int),("AccentFlags",ctypes.c_int),
              ("GradientColor",ctypes.c_uint),("AnimationId",ctypes.c_int)]
class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
    _fields_=[("Attribute",ctypes.c_int),("Data",ctypes.c_void_p),("SizeOfData",ctypes.c_size_t)]
WCA_ACCENT_POLICY = 19
ACCENT_DISABLED   = 0
ACCENT_ENABLE_ACRYLICBLURBEHIND = 4

user32 = ctypes.windll.user32
SetWindowCompositionAttribute = user32.SetWindowCompositionAttribute
SetWindowCompositionAttribute.argtypes=[wt.HWND, ctypes.POINTER(WINDOWCOMPOSITIONATTRIBDATA)]
SetWindowCompositionAttribute.restype=_HRESULT

def _set_accent(hwnd:int, state:int, tint_abgr:int=0):
    policy = ACCENT_POLICY()
    policy.AccentState = state
    policy.GradientColor = tint_abgr
    data = WINDOWCOMPOSITIONATTRIBDATA()
    data.Attribute = WCA_ACCENT_POLICY
    data.SizeOfData = ctypes.sizeof(policy)
    data.Data = ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p)
    try:
        SetWindowCompositionAttribute(wt.HWND(hwnd), ctypes.byref(data))
    except Exception as e:
        print("[ACRYLIC] error:", e)

def enable_acrylic(hwnd:int, tint_abgr:int=0x40101010):
    _set_accent(hwnd, ACCENT_ENABLE_ACRYLICBLURBEHIND, tint_abgr)

def disable_acrylic(hwnd:int):
    _set_accent(hwnd, ACCENT_DISABLED, 0)

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
        # применяем выбранный режим фона
        hwnd = int(self.winId())
        mode  = globals().get("PANEL_BG_MODE", "blur")
        alpha = int(globals().get("PANEL_BG_ALPHA", 64))
        alpha = max(0, min(255, alpha))

        if mode == "blur":
            # tint цвет — тёмно-серый, альфа задаётся из настроек
            tint = (alpha << 24) | 0x101010
            enable_acrylic(hwnd, tint_abgr=tint)
        else:
            # для solid/none акрил отключаем
            disable_acrylic(hwnd)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        ov = self.parent_overlay

        rect_full = self.rect()
        f = QFont(getattr(ov, 'font_family', 'Segoe UI'),
                  int(getattr(ov, 'font_size', 22)))
        f.setBold(bool(getattr(ov, 'font_bold', False)))
        p.setFont(f)

        text = self.text or ""

        # --- режим «подставлять перевод на место оригинального текста» ---
        if getattr(ov, "draw_over_original", False):
            # найдём регион, которому принадлежит эта панель
            region = None
            for r in getattr(ov, "regions", []):
                if getattr(r, "panel", None) is self:
                    region = r
                    break

            boxes = getattr(region, "text_boxes", None) if region is not None else None

            if boxes:
                # 1) Убираем строку с именем из текста (но не из OCR/перевода)
                raw_lines = text.split("\n")
                visible_text = text

                if len(raw_lines) >= 2:
                    raw_first = raw_lines[0].strip()

                    # сносим ведущие многоточия / точки / пробелы
                    cand = re.sub(r"^[.…\s]+", "", raw_first)

                    # убираем висящий двоеточие в конце
                    if cand.endswith(":"):
                        cand_core = cand[:-1].strip()
                    else:
                        cand_core = cand

                    if cand_core:
                        tokens = cand_core.split()

                        # считаем nameplate'ом:
                        #  - 1–3 коротких слова
                        #  - без пунктуации внутри
                        #  - каждое слово начинается с заглавной буквы
                        looks_like_label = (
                            1 <= len(tokens) <= 3
                            and not re.search(r"[.,!?;:]", cand_core)
                            and all(t and t[0].isupper() for t in tokens)
                        )

                        if looks_like_label:
                            visible_text = "\n".join(raw_lines[1:]).lstrip("\n")

                if not visible_text.strip():
                    return  # показывать нечего

                # 2) Объединяем ВСЕ боксы в одну область —
                # так перевод остаётся на том же месте, что и исходный английский блок
                boxes_sorted = sorted(boxes, key=lambda b: b[1])
                min_x = min(b[0] for b in boxes_sorted)
                min_y = min(b[1] for b in boxes_sorted)
                max_x = max(b[0] + b[2] for b in boxes_sorted)
                max_y = max(b[1] + b[3] for b in boxes_sorted)
                union_box = (min_x, min_y, max_x - min_x, max_y - min_y)

                def draw_block(norm_box, txt):
                    if not txt.strip():
                        return

                    nx, ny, nw, nh = norm_box

                    # переводим нормализованные координаты (0..1) в пиксели панели
                    x = rect_full.x() + int(nx * rect_full.width())
                    y = rect_full.y() + int(ny * rect_full.height())

                    fm = p.fontMetrics()
                    flags = Qt.TextWordWrap | Qt.AlignLeft

                    max_w = rect_full.width() - (x - rect_full.x()) - 10
                    if max_w <= 0:
                        return

                    # считаем размер текста в локальной системе координат
                    tmp_rect = fm.boundingRect(0, 0, max_w, 10_000, flags, txt)

                    text_rect = QRect(
                        x,
                        y,
                        tmp_rect.width(),
                        tmp_rect.height(),
                    )

                    # паддинги вокруг текста
                    pad_x, pad_y = 8, 4
                    bg_rect = text_rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)

                    # фон под строкой / блоком
                    mode_box = globals().get("BOX_BG_MODE", "solid")
                    alpha_box = int(globals().get("BOX_BG_ALPHA", 180))
                    alpha_box = max(0, min(255, alpha_box))

                    if mode_box == "solid" and alpha_box > 0:
                        p.setPen(Qt.NoPen)
                        p.setBrush(QColor(0, 0, 0, alpha_box))
                        p.drawRoundedRect(bg_rect, 4, 4)

                    # тень
                    p.setPen(QColor(0, 0, 0, 220))
                    p.drawText(text_rect.translated(1, 1), flags, txt)

                    # основной текст
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(text_rect, flags, txt)

                draw_block(union_box, visible_text)
                return  # всё нарисовали, базовый режим не нужен

        # --- обычный режим панели (как раньше) ---
        rect = rect_full.adjusted(self.margin, self.margin,
                                  -self.margin, -self.margin)

        mode = globals().get("PANEL_BG_MODE", "blur")
        alpha = int(globals().get("PANEL_BG_ALPHA", 64))
        alpha = max(0, min(255, alpha))

        if mode == "solid" and alpha > 0:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, alpha))
            p.drawRoundedRect(rect_full, 4, 4)

        # тень
        p.setPen(QColor(0, 0, 0, 220))
        p.drawText(
            rect.translated(1, 1),
            Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignVCenter,
            text,
        )
        # текст
        p.setPen(QColor(255, 255, 255))
        p.drawText(
            rect,
            Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignVCenter,
            text,
        )


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
    last_ocr_ts: float = 0.0

    # канон/анти-дубли
    last_ru_canon: str = ""
    last_sent_ru_canon: str = ""
    last_sent_ru_raw: str = ""
    pending_ru_raw: str = ""
    pending_ru_canon: str = ""
    pending_hits: int = 0
    is_frozen: bool = False
    
    last_en_canon: str = ""
    
    text_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list, repr=False)

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

                now = time.perf_counter()
                min_interval = 1.0 / max(0.1, float(MAX_OCR_FPS))  # защита от деления на 0
                elapsed = now - r.last_ocr_ts
                if elapsed < min_interval:
                    # доспим ровно до нужного FPS
                    rem_ms = int(max(1, (min_interval - elapsed) * 1000))
                    self.msleep(rem_ms)
                    continue

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
                now = time.perf_counter()

                r.last_ocr_ts = now

                # если включён режим «подставлять перевод на место оригинального текста» —
                # найдём bbox'ы текста и запомним их в нормализованном виде

                # LLM: одна картинка (регион) → перевод
                reg_b64  = _bgr_to_png_b64(frame, max_side=1024)
                full_b64 = None
                r.last_ocr_ts = time.perf_counter()

                en = ""
                ru = ""

                if getattr(self.overlay, "dual_mode", False) and LLMConfigDual is not None:
                    # ---------- DUAL: отдельно логируем OCR и перевод ----------
                    t0 = time.perf_counter()
                    en = extract_en_from_image(reg_b64, self.overlay.ocr_cfg_dual)
                    t1 = time.perf_counter()
                    print(f"[DUAL][OCR] region {idx}: {len(en) if en else 0} chars, {t1 - t0:.3f}s, text = {repr((en or '')[:200])}")

                    t2 = time.perf_counter()
                    ru = translate_en_to_ru_text(en, self.overlay.tr_cfg_dual) if en else ""
                    t3 = time.perf_counter()
                    print(f"[DUAL][TR ] region {idx}: {len(ru) if ru else 0} chars, {t3 - t2:.3f}s, text = {repr((ru or '')[:200])}")
                else:
                    # SOLO как раньше
                    t0 = time.perf_counter()
                    ru = vision_translate_from_images(reg_b64, full_b64, self.overlay.llm_cfg)
                    t1 = time.perf_counter()
                    print(f"[SOLO][VL ] region {idx}: {len(ru) if ru else 0} chars, {t1 - t0:.3f}s, text = {repr((ru or '')[:200])}")
                    
                if getattr(self.overlay, "draw_over_original", False):
                    try:
                        boxes_with_text = detect_text_boxes(frame)  # список (x,y,w,h,text)
                        print(f"[BOXES] region {idx}: {len(boxes_with_text)} boxes -> {boxes_with_text}")

                        boxes_px = _filter_boxes_by_llm_text(boxes_with_text, en)
                        if len(boxes_px) != len(boxes_with_text):
                            print(f"[BOXES-FIX] region {idx}: filtered overlay boxes; {len(boxes_with_text)} -> {len(boxes_px)}")

                        h, w = gray.shape[:2]
                        r.text_boxes = [
                            (x / float(w), y / float(h), bw / float(w), bh / float(h))
                            for (x, y, bw, bh) in boxes_px
                        ]
                    except Exception as e:
                        print("[OCR-BOXES] error:", e)
                        r.text_boxes = []

                # --- DUAL: вырезаем строку с именем из RU, если в EN оно есть ---
                name_line = None
                if getattr(self.overlay, "dual_mode", False) and en and ru:
                    try:
                        name_line, _body = _split_name_and_body(en)
                    except Exception:
                        name_line = None

                    if name_line:
                        ru_lines = ru.splitlines()
                        if len(ru_lines) >= 2:
                            ru_first = ru_lines[0].strip()

                            # Нормализация метки: убираем всё кроме букв/цифр
                            def _canon_label(s: str) -> str:
                                s = s or ""
                                return re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "", s).lower()

                            # первая строка RU выглядит как короткий неймплейт
                            looks_like_ru_label = (
                                ru_first
                                and len(ru_first) <= 32
                                and len(ru_first.split()) <= 3
                                and not any(ch in ru_first for ch in ".!?:")
                            )

                            if looks_like_ru_label:
                                en_label = _canon_label(name_line)
                                ru_label = _canon_label(ru_first)

                                # строки более-менее совпадают
                                if en_label and ru_label and (
                                    ru_label == en_label
                                    or en_label in ru_label
                                    or ru_label in en_label
                                ):
                                    # Считаем, что первая строка — реально имя → убираем её
                                    ru = "\n".join(ru_lines[1:]).lstrip("\n")
                
                if ru:
                    # канонизируем перевод
                    try:
                        ru_canon = canon_ru(ru)
                    except Exception:
                        ru_canon = (ru or "").strip().lower()

                    # канон исходного EN (есть только в DUAL, в SOLO en будет пустым)
                    try:
                        en_canon = canon_en(en) if en else ""
                    except Exception:
                        en_canon = ""

                    mode = "DUAL" if getattr(self.overlay, "dual_mode", False) else "SOLO"

                    # --- Анти-фликер по EN ---
                    # Если канон EN не изменился, считаем, что фраза та же,
                    # и игнорируем любые «перепридуманные» варианты перевода.
                    if en_canon and en_canon == (getattr(r, "last_en_canon", "") or ""):
                        print(f"[{mode}][SKIP] region {idx}: same EN canon → keep previous RU")
                        # НИЧЕГО не отправляем в панель и не обновляем last_sent_ru_*
                        # (на экране останется прошлый стабильный перевод)
                    else:
                        # EN изменился (или мы в SOLO и en пустой) — смотрим на RU-канон
                        if ru_canon == (r.last_sent_ru_canon or ""):
                            print(f"[{mode}][SKIP] region {idx}: same RU canon")
                        else:
                            # новый перевод — отправляем в панель и обновляем состояние
                            self.textReady.emit(idx, ru)
                            r.last_sent_ru_raw   = ru
                            r.last_sent_ru_canon = ru_canon
                            r.last_en_canon      = en_canon  # <--- важное обновление

                            r.pending_ru_raw = r.pending_ru_canon = ""
                            r.pending_hits = 0

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
        self.border_w = BORDER_WIDTH
        self.split_border_w = SPLIT_BORDER_WIDTH
        self.draw_over_original: bool = False
        self.dual_mode = False
        self.bg_mode  = PANEL_BG_MODE
        self.bg_alpha = PANEL_BG_ALPHA
        self.ocr_cfg_dual = None
        self.tr_cfg_dual  = None

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
        self.draw_over_original = DRAW_OVER_ORIGINAL

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
            pen=QPen(QColor(0,200,255,230) if r.selected else QColor(255,255,255,120))
            pen.setWidth(int(getattr(self, "border_w", BORDER_WIDTH)))
            p.setPen(pen)
            p.drawRect(r.display_rect); self._handles(p, r.display_rect, QColor(0,200,255,230))
            if r.split_mode:
                pen2=QPen(QColor(255,165,0,220)); pen2.setStyle(Qt.DashLine)
                pen2.setWidth(int(getattr(self, "split_border_w", SPLIT_BORDER_WIDTH)))
                p.setPen(pen2)
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
        try:
            _stop_llama_server2()
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
        self.hide()
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

def rebuild_server_cmd(exe, model, mmproj, *, host, port, ctx, batch, ubatch, parallel, ngl):
    cmd = [exe, "-m", model]
    if mmproj:
        cmd += ["--mmproj", mmproj]
    cmd += [
        "-ngl", str(int(ngl)),
        "--ctx-size", str(int(ctx)),
        "--batch-size", str(int(batch)),
        "--ubatch-size", str(int(ubatch)),
        "--parallel", str(int(parallel)),
        "--host", str(host),
        "--port", str(int(port)),
    ]
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
_LLAMA_PROC2 = None

def _spawn(cmd_list):
    # не используем shell, даём список аргументов
    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
             | getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
    return subprocess.Popen(cmd_list, shell=False, creationflags=flags)

def _ensure_llama_server(cmd_list, server_url):
    global _LLAMA_PROC
    if _ping_llama(server_url):
        print(f"[LLM] server ok @ {server_url}")
        return
    print("[LLM] spawn:", cmd_list)
    _LLAMA_PROC = _spawn(cmd_list)
    print(f"[LLM] llama-server started (pid={_LLAMA_PROC.pid})")
    for _ in range(120):
        time.sleep(0.5)
        if _ping_llama(server_url):
            print("[LLM] llama-server поднялся")
            return
        if _LLAMA_PROC.poll() is not None:
            print("[LLM] llama-server exited early code", _LLAMA_PROC.returncode)
            return

def _stop_llama_server():
    global _LLAMA_PROC
    if _LLAMA_PROC and _LLAMA_PROC.poll() is None:
        try:
            _LLAMA_PROC.terminate()
            try: _LLAMA_PROC.wait(timeout=3)
            except Exception: _LLAMA_PROC.kill()
        except Exception: pass
    _LLAMA_PROC = None

def _ensure_llama_server2(cmd_list, server_url):
    global _LLAMA_PROC2
    if _ping_llama(server_url):
        print(f"[LLM] server2 ok @ {server_url}")
        return
    print("[LLM] spawn#2:", cmd_list)
    _LLAMA_PROC2 = _spawn(cmd_list)
    print(f"[LLM] llama-server#2 started (pid={_LLAMA_PROC2.pid})")
    for _ in range(120):
        time.sleep(0.5)
        if _ping_llama(server_url):
            print("[LLM] llama-server#2 поднялся")
            return
        if _LLAMA_PROC2.poll() is not None:
            print("[LLM] llama-server#2 exited early code", _LLAMA_PROC2.returncode)
            return

def _stop_llama_server2():
    global _LLAMA_PROC2
    if _LLAMA_PROC2 and _LLAMA_PROC2.poll() is None:
        try:
            _LLAMA_PROC2.terminate()
            try: _LLAMA_PROC2.wait(timeout=3)
            except Exception: _LLAMA_PROC2.kill()
        except Exception: pass
    _LLAMA_PROC2 = None

# ====================== Настроечный GUI ===================

class ConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Translator — Настройки")
        self.setGeometry(120,120,980,760)

        # --- controls ---
        self.btnStart = QPushButton("Начать перевод")
        self.cbDual = QCheckBox("Двойной режим (OCR + Переводчик)")
        
        self.btnStart.clicked.connect(self._start)

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
        
        self.cbModelOCR  = QComboBox()   # DUAL: OCR-модель .gguf
        self.cbMmprojOCR = QComboBox()   # DUAL: mmproj для OCR
        self.cbModelTR   = QComboBox()   # DUAL: текстовая модель перевода

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
        self.spRetr = QDoubleSpinBox(); self.spRetr.setRange(0.0, 600.0); self.spRetr.setSingleStep(0.5); self.spRetr.setDecimals(1); self.spRetr.setValue(0.0)
        self.spBorder = QSpinBox(); self.spBorder.setRange(1, 16); self.spBorder.setValue(BORDER_WIDTH)
        self.spBorder.valueChanged.connect(lambda v: (setattr(self.overlay, "border_w", int(v)), self.overlay.update()))
        self.spSplit  = QSpinBox(); self.spSplit.setRange(1, 16); self.spSplit.setValue(SPLIT_BORDER_WIDTH)
        self.spSplit.valueChanged.connect(lambda v: (setattr(self.overlay, "split_border_w", int(v)), self.overlay.update()))
        self.cbSmooth = QCheckBox("Плавный вывод (smooth)"); self.cbSmooth.setChecked(RENDER_MODE=="smooth")
        self.cbOverlayBoxes = QCheckBox("Подставлять перевод на место оригинального текста")
        self.cbOverlayBoxes.setChecked(False)
        self.spHandle = QSpinBox(); self.spHandle.setRange(6, 30); self.spHandle.setValue(int(HANDLE))
        self.spHandle.valueChanged.connect(lambda v: (globals().__setitem__("HANDLE", int(v)), self.overlay.update()))

        # лор
        self.edLore  = QLineEdit(str(llm_cfg.lore_path or ""))
        self.btnLore = QPushButton("Выбрать…")
        self.edPB  = QLineEdit("")          # путь к phrasebook.txt
        self.btnPB = QPushButton("Выбрать…")
        

        # промпт
        self.btnPrompt  = QPushButton("Настроить промпт…")
        self.btnPreview = QPushButton("Предпросмотр system…")
        self.btnEditOCR = QPushButton("Настроить промпт (OCR)")
        self.btnPrevOCR = QPushButton("Предпросмотр (OCR)")
        self.btnEditOCR.clicked.connect(self._edit_ocr_prompt)
        self.btnPrevOCR.clicked.connect(self._preview_ocr)

        # --- layout: шапка + левое меню + стэк страниц ---

        top = QVBoxLayout(self)

        # основная область: слева навигация, справа стэк страниц
        main = QHBoxLayout()
        top.addLayout(main, 1)

        # =========== ЛЕВАЯ КОЛОНКА НАВИГАЦИИ ============

        def _make_nav_button(icon_path: str, tooltip: str) -> QToolButton:
            btn = QToolButton(self)
            # иконка
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(32, 32))  # размер самой картинки
            # текст нам не нужен, оставим только подсказку
            btn.setText("")                  # на всякий случай
            btn.setToolTip(tooltip)
            # поведение
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            # размеры кнопки (можешь подправить под свои иконки)
            btn.setFixedSize(64, 64)
            return btn

        # здесь подставишь свои реальные пути к картинкам
        self.btnNavHome      = _make_nav_button("_internal/icons/Home.png",      "Дом")
        self.btnNavCustomize = _make_nav_button("_internal/icons/UI.png",     "Кастомизация")
        self.btnNavDownload  = _make_nav_button("_internal/icons/Download.png",  "Скачать модели")
        self.btnNavModel     = _make_nav_button("_internal/icons/Settings.png",       "Настройки моделей")
        self.btnNavInfo      = _make_nav_button("_internal/icons/FAQ.png",      "Информация")

        navLayout = QVBoxLayout()
        navLayout.setContentsMargins(0, 0, 0, 0)
        navLayout.setSpacing(8)

        # верхняя группа кнопок
        navLayout.addWidget(self.btnNavHome)
        navLayout.addWidget(self.btnNavCustomize)
        navLayout.addWidget(self.btnNavDownload)
        navLayout.addWidget(self.btnNavModel)

        # растяжка, чтобы Info ушла в самый низ
        navLayout.addStretch()

        # кнопка Info прижата к низу
        navLayout.addWidget(self.btnNavInfo)

        # оборачиваем в отдельный виджет с фиксированной шириной
        navWidget = QWidget(self)
        navWidget.setLayout(navLayout)
        navWidget.setFixedWidth(64)  # подгони под размер своих иконок

        main.addWidget(navWidget)

        # вертикальная разделительная линия
        sep = QFrame(self)
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        main.addWidget(sep)

        # =========== ПРАВАЯ ОБЛАСТЬ: QStackedWidget ============
        self.stack = QStackedWidget(self)
        main.addWidget(self.stack, 1)

        # --------------------------------------------------------
        # СТРАНИЦА 0 — ДОМ: старт, окно захвата, лор и фразбук
        # --------------------------------------------------------

        pageHome = QWidget()
        homeLayout = QVBoxLayout(pageHome)
        homeLayout.setContentsMargins(0, 0, 0, 0)

        scrollHome = QScrollArea(pageHome)
        scrollHome.setWidgetResizable(True)
        innerHome = QWidget(scrollHome)
        formHome = QVBoxLayout(innerHome)
        
        # Начать перевод
        topRow = QHBoxLayout()
        topRow.addWidget(self.btnStart)
        topRow.addSpacing(12)
        topRow.addWidget(self.cbDual)
        topRow.addStretch()
        formHome.addLayout(topRow)

        # Окно для захвата
        gbWin = QGroupBox("Окно для захвата")
        lw = QHBoxLayout(gbWin)
        lw.addWidget(QLabel("Окно:"))
        lw.addWidget(self.cbWindow, 1)
        lw.addWidget(self.btnRefreshWin)
        formHome.addWidget(gbWin)

        # Выбор модели
        gbModels = QGroupBox("Модели")
        fmModels = QFormLayout(gbModels)

        # подписи запоминаем как атрибуты, чтобы можно было скрывать
        self.lblSoloModel   = QLabel("SOLO .gguf:")
        self.lblSoloMmproj  = QLabel("SOLO mmproj:")
        self.lblOcrModel    = QLabel("DUAL OCR .gguf:")
        self.lblOcrMmproj   = QLabel("DUAL OCR mmproj:")
        self.lblTrModel     = QLabel("DUAL TR .gguf:")

        # SOLO
        fmModels.addRow(self.lblSoloModel,  self.cbModel)
        fmModels.addRow(self.lblSoloMmproj, self.cbMmproj)

        # DUAL OCR
        fmModels.addRow(self.lblOcrModel,   self.cbModelOCR)
        fmModels.addRow(self.lblOcrMmproj,  self.cbMmprojOCR)

        # DUAL переводчик
        fmModels.addRow(self.lblTrModel,    self.cbModelTR)

        formHome.addWidget(gbModels)
        
        # Лор
        gbLore = QGroupBox("Лор (game_bible)")
        fl = QHBoxLayout(gbLore)
        fl.addWidget(QLabel("Файл лора:"))
        fl.addWidget(self.edLore, 1)
        fl.addWidget(self.btnLore)
        formHome.addWidget(gbLore)

        # Phrasebook
        gbPB = QGroupBox("Фразеологический словарик (phrasebook)")
        fpb = QHBoxLayout(gbPB)
        fpb.addWidget(QLabel("Файл phrasebook:"))
        fpb.addWidget(self.edPB, 1)
        fpb.addWidget(self.btnPB)
        formHome.addWidget(gbPB)

        formHome.addStretch()

        scrollHome.setWidget(innerHome)
        homeLayout.addWidget(scrollHome)
        self.stack.addWidget(pageHome)

        # --------------------------------------------------------
        # СТРАНИЦА 1 — КАСТОМИЗАЦИЯ UI: шрифт, фон, FPS, рамки
        # --------------------------------------------------------

        pageCustomize = QWidget()
        layCust = QVBoxLayout(pageCustomize)
        layCust.setContentsMargins(0, 0, 0, 0)

        scrollCust = QScrollArea(pageCustomize)
        scrollCust.setWidgetResizable(True)
        innerCust = QWidget(scrollCust)
        formCust = QVBoxLayout(innerCust)

        # Шрифт панели
        gbFont = QGroupBox("Шрифт панели")
        ff = QFormLayout(gbFont)
        ff.addRow("Гарнитура:", self.fontFamily)
        ff.addRow("Размер:", self.fontSize)
        ff.addRow("", self.fontBold)
        formCust.addWidget(gbFont)

        # Режим фона панели
        self.cbBgMode = QComboBox()
        self.cbBgMode.addItem("Размытие Windows (акрил)", "blur")
        self.cbBgMode.addItem("Чёрный фон", "solid")
        self.cbBgMode.addItem("Без фона (только текст)", "none")

        self.spBgAlpha = QSlider(Qt.Horizontal)
        self.spBgAlpha.setRange(1, 10)
        self.spBgAlpha.setSingleStep(1)
        self.spBgAlpha.setValue(5)
        self.spBgAlpha.setToolTip("1 — минимальное размытие/затемнение, 10 — максимальное")

        # Фон под строками
        self.cbBoxBgMode = QComboBox()
        self.cbBoxBgMode.addItem("Как у панели", "panel")
        self.cbBoxBgMode.addItem("Чёрный фон под строками", "solid")
        self.cbBoxBgMode.addItem("Без фона под строками", "none")

        self.spBoxBgAlpha = QSlider(Qt.Horizontal)
        self.spBoxBgAlpha.setRange(1, 10)
        self.spBoxBgAlpha.setSingleStep(1)
        self.spBoxBgAlpha.setValue(7)
        self.spBoxBgAlpha.setToolTip("1 — почти прозрачный, 10 — максимально тёмные плашки под строками")

        # Константы/поведение панели
        gbK = QGroupBox("Вид и поведение панели")
        fk = QFormLayout(gbK)
        fk.addRow("FPS захвата:", self.spFps)
        fk.addRow("Ширина рамки:", self.spBorder)
        fk.addRow("Ширина разделителя:", self.spSplit)
        fk.addRow("Плавный вывод:", self.cbSmooth)
        fk.addRow("Overlay по боксам:", self.cbOverlayBoxes)
        fk.addRow("Ручка рамки:", self.spHandle)
        fk.addRow("Фон панели:", self.cbBgMode)
        fk.addRow("Интенсивность фона:", self.spBgAlpha)
        fk.addRow("Фон под строками:", self.cbBoxBgMode)
        fk.addRow("Интенсивность под строками:", self.spBoxBgAlpha)
        formCust.addWidget(gbK)

        formCust.addStretch()
        scrollCust.setWidget(innerCust)
        layCust.addWidget(scrollCust)
        self.stack.addWidget(pageCustomize)

        # --------------------------------------------------------
        # СТРАНИЦА 2 — СКАЧАТЬ МОДЕЛИ
        # --------------------------------------------------------

        pageDownload = QWidget()
        layDown = QVBoxLayout(pageDownload)

        # Встраиваем менеджер моделей прямо во вкладку
        try:
            base = os.path.dirname(self.cbModel.currentText().strip() or "")
            if not base:
                base = str((BASE_DIR / "models" / "llm").resolve())
        except Exception:
            base = os.path.join(os.getcwd(), "models", "llm")

        # один экземпляр менеджера, просто как обычный виджет
        self.modelHub = ModelHubDialog(self, default_models_dir=base)
        # кнопка "Закрыть" во вкладке не нужна
        if hasattr(self.modelHub, "btnClose"):
            self.modelHub.btnClose.hide()

        layDown.addWidget(self.modelHub)
        self.stack.addWidget(pageDownload)

        # --------------------------------------------------------
        # СТРАНИЦА 3 — НАСТРОЙКА МОДЕЛЕЙ (SOLO / DUAL)
        # --------------------------------------------------------

        pageModel = QWidget()
        layModel = QVBoxLayout(pageModel)
        layModel.setContentsMargins(0, 0, 0, 0)

        scrollModel = QScrollArea(pageModel)
        scrollModel.setWidgetResizable(True)
        innerModel = QWidget(scrollModel)
        formModel = QVBoxLayout(innerModel)

        # OCR-сервер (DUAL)
        self.gbOCR = QGroupBox("OCR-сервер (vision)")
        self.edHost1 = QLineEdit("127.0.0.1")
        self.spPort1 = QSpinBox(); self.spPort1.setRange(1,65535); self.spPort1.setValue(8080)
        self.spCtx1  = QSpinBox(); self.spCtx1.setRange(1024,65536); self.spCtx1.setValue(8192)
        self.spBatch1= QSpinBox(); self.spBatch1.setRange(1,4096); self.spBatch1.setValue(256)
        self.spUB1   = QSpinBox(); self.spUB1.setRange(1,4096); self.spUB1.setValue(64)
        self.spPar1  = QSpinBox(); self.spPar1.setRange(1,16); self.spPar1.setValue(1)
        self.spNGL1  = QSpinBox(); self.spNGL1.setRange(0,999); self.spNGL1.setValue(999)
        self.spSeed1 = QSpinBox(); self.spSeed1.setRange(0, 2_000_000_000); self.spSeed1.setValue(0)
        self.spSlot1 = QSpinBox(); self.spSlot1.setRange(0,7); self.spSlot1.setValue(0)
        self.spMaxTok1 = QSpinBox()
        self.spMaxTok1.setRange(64, 16384)
        self.spMaxTok1.setSingleStep(64)
        self.spMaxTok1.setValue(512)

        f1 = QFormLayout(self.gbOCR)
        f1.addRow("host:", self.edHost1)
        f1.addRow("port:", self.spPort1)
        f1.addRow("max_tokens:", self.spMaxTok1)
        f1.addRow("ctx-size:", self.spCtx1)
        f1.addRow("batch-size:", self.spBatch1)
        f1.addRow("ubatch-size:", self.spUB1)
        f1.addRow("parallel:", self.spPar1)
        f1.addRow("ngl:", self.spNGL1)
        f1.addRow("seed:", self.spSeed1)
        f1.addRow("slot_id:", self.spSlot1)
        f1.addRow(self.btnEditOCR)
        f1.addRow(self.btnPrevOCR)
        f1.addRow(self.btnPrevOCR)
        formModel.addWidget(self.gbOCR)

        # Переводчик (DUAL)
        self.gbTR = QGroupBox("Перевод-сервер (text)")
        self.edHost2 = QLineEdit("127.0.0.1")
        self.spPort2 = QSpinBox(); self.spPort2.setRange(1,65535); self.spPort2.setValue(8081)
        self.spCtx2  = QSpinBox(); self.spCtx2.setRange(1024,65536); self.spCtx2.setValue(6000)
        self.spBatch2= QSpinBox(); self.spBatch2.setRange(1,4096); self.spBatch2.setValue(256)
        self.spUB2   = QSpinBox(); self.spUB2.setRange(1,4096); self.spUB2.setValue(64)
        self.spPar2  = QSpinBox(); self.spPar2.setRange(1,16); self.spPar2.setValue(1)
        self.spNGL2  = QSpinBox(); self.spNGL2.setRange(0,999); self.spNGL2.setValue(999)
        self.spSeed2 = QSpinBox(); self.spSeed2.setRange(0, 2_000_000_000); self.spSeed2.setValue(0)
        self.spSlot2 = QSpinBox(); self.spSlot2.setRange(0,7); self.spSlot2.setValue(1)
        self.spMaxTok2 = QSpinBox()
        self.spMaxTok2.setRange(64, 16384)
        self.spMaxTok2.setSingleStep(64)
        self.spMaxTok2.setValue(1024)
        
        self.spTO2 = QDoubleSpinBox();   self.spTO2.setRange(1.0, 600.0);  self.spTO2.setSingleStep(1.0);  self.spTO2.setDecimals(1);  self.spTO2.setValue(float(getattr(llm_cfg, 'timeout_s', 60.0)))
        self.spTemp2 = QDoubleSpinBox(); self.spTemp2.setRange(0.0, 2.0);  self.spTemp2.setSingleStep(0.05); self.spTemp2.setDecimals(2); self.spTemp2.setValue(float(getattr(llm_cfg, 'temp', 0.2)))
        self.spTopP2 = QDoubleSpinBox(); self.spTopP2.setRange(0.0, 1.0);  self.spTopP2.setSingleStep(0.05); self.spTopP2.setDecimals(2); self.spTopP2.setValue(float(getattr(llm_cfg, 'top_p', 0.9)))
        self.spTopK2 = QSpinBox();       self.spTopK2.setRange(0, 10000);  self.spTopK2.setValue(int(getattr(llm_cfg, 'top_k', 0)))
        self.spRP2   = QDoubleSpinBox(); self.spRP2.setRange(0.1, 2.0);    self.spRP2.setSingleStep(0.05);  self.spRP2.setDecimals(2);  self.spRP2.setValue(float(getattr(llm_cfg, 'repeat_penalty', 1.0)))

        self.cbDual.toggled.connect(self._reflow_mode_ui)

        self.btnEditTR  = QPushButton("Настроить промпт (Переводчик)")
        self.btnPrevTR  = QPushButton("Предпросмотр (Переводчик)")
        self.btnEditTR.clicked.connect(self._edit_tr_prompt)
        self.btnPrevTR.clicked.connect(self._preview_tr)

        f2 = QFormLayout(self.gbTR)
        f2.addRow("host:", self.edHost2)
        f2.addRow("port:", self.spPort2)
        f2.addRow("max_tokens:", self.spMaxTok2)
        f2.addRow("timeout, сек:", self.spTO2)
        f2.addRow("temperature:", self.spTemp2)
        f2.addRow("top_p:", self.spTopP2)
        f2.addRow("top_k:", self.spTopK2)
        f2.addRow("repeat_penalty:", self.spRP2)
        f2.addRow("ctx-size:", self.spCtx2)
        f2.addRow("batch-size:", self.spBatch2)
        f2.addRow("ubatch-size:", self.spUB2)
        f2.addRow("parallel:", self.spPar2)
        f2.addRow("ngl:", self.spNGL2)
        f2.addRow("seed:", self.spSeed2)
        f2.addRow("slot_id:", self.spSlot2)
        f2.addRow(self.btnEditTR)
        f2.addRow(self.btnPrevTR)
        formModel.addWidget(self.gbTR)

        # Настройки модели (SOLO)
        self.gbLLM = QGroupBox("Настройки модели (SOLO)")
        flm = QFormLayout(self.gbLLM)
        flm.addRow("max_tokens:", self.spMaxTok)
        flm.addRow("timeout, сек:", self.spTO)
        flm.addRow("temperature:", self.spTemp)
        flm.addRow("top_p:", self.spTopP)
        formModel.addWidget(self.gbLLM)

        # Доп. параметры генерации (SOLO)
        self.gbGen = QGroupBox("Доп. параметры генерации (SOLO)")
        fgen = QFormLayout(self.gbGen)
        fgen.addRow("top_k:", self.spTopK)
        fgen.addRow("repeat_penalty:", self.spRP)
        fgen.addRow("seed:", self.spSeed)
        fgen.addRow("slot_id:", self.spSlot)
        fgen.addRow("", self.cbCache)
        formModel.addWidget(self.gbGen)

        self.gbSrv = QGroupBox("Сервер (llama.cpp)")
        fsrv = QFormLayout(self.gbSrv)
        fsrv.addRow("ctx-size:", self.spCtx)
        fsrv.addRow("batch-size:", self.spBatch)
        fsrv.addRow("ubatch-size:", self.spUBatch)
        fsrv.addRow("parallel:", self.spPar)
        fsrv.addRow("ngl:", self.spNGL)

        rowSrv = QWidget()
        h = QHBoxLayout(rowSrv)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("host:"))
        h.addWidget(self.edHost)
        h.addSpacing(8)
        h.addWidget(QLabel("port:"))
        h.addWidget(self.spPort)
        fsrv.addRow(rowSrv)

        formModel.addWidget(self.gbSrv)

        # Кнопки промпта (SOLO)
        rowPrompt = QHBoxLayout()
        rowPrompt.addWidget(self.btnPrompt)
        rowPrompt.addWidget(self.btnPreview)
        formModel.addLayout(rowPrompt)

        formModel.addStretch()
        scrollModel.setWidget(innerModel)
        layModel.addWidget(scrollModel)
        self.stack.addWidget(pageModel)

        # --------------------------------------------------------
        # СТРАНИЦА 4 — ИНФО
        # --------------------------------------------------------

        pageInfo = QWidget()
        layInfo = QVBoxLayout(pageInfo)
        infoLabel = QLabel()
        infoLabel.setTextFormat(Qt.RichText)
        infoLabel.setOpenExternalLinks(True)
        infoLabel.setWordWrap(True)
        infoLabel.setText("АИ переводчик by IgoRexa. <br>"
                            "Версия 0.0.4 <br>"
                            "GitHub: "
                            "<a href='https://github.com/igorexa225/AIGameTranslater'>"
                            "https://github.com/igorexa225/AIGameTranslater"
                            "</a>")
        layInfo.addWidget(infoLabel)
        infoLabel.setStyleSheet("font-size: 14pt;")
        layInfo.addStretch()
        self.stack.addWidget(pageInfo)

        # по умолчанию показываем Дом
        self.btnNavHome.setChecked(True)
        self.stack.setCurrentIndex(0)

        # навигация
        self.btnNavHome.clicked.connect(lambda: self._switch_page(0))
        self.btnNavCustomize.clicked.connect(lambda: self._switch_page(1))
        self.btnNavDownload.clicked.connect(lambda: self._switch_page(2))
        self.btnNavModel.clicked.connect(lambda: self._switch_page(3))
        self.btnNavInfo.clicked.connect(lambda: self._switch_page(4))
        
        self.overlay = Overlay(on_quit=self._back_from_overlay, start_worker_immediately=False)
        self.overlay.show()
        self.raise_()  # держим окно настроек сверху

        self._rescan_models()
        self._fill_windows()
        self._load_prefs_into_ui()
        self._reflow_mode_ui()
    
    def _open_model_hub(self):
        # пытаемся угадать папку моделей
        base = ""
        try:
            base = os.path.dirname(self.cbModel.currentText().strip() or "")
            if not base:
                base = str((BASE_DIR / "models" / "llm").resolve())
        except Exception:
            base = os.path.join(os.getcwd(), "models", "llm")

        dlg = ModelHubDialog(self, default_models_dir=base)
        dlg.exec()

        # после закрытия — рескан моделей (если метод есть)
        for name in ("_rescan_models", "_scan_models", "_scan_models_ui", "_rebuild_models_list"):
            if hasattr(self, name):
                try:
                    getattr(self, name)()
                    break
                except Exception as e:
                    print("[ModelHub] rescan error:", e)
    
    def _switch_page(self, index: int):
        """Переключение вкладок слева."""
        if hasattr(self, "stack"):
            self.stack.setCurrentIndex(index)
        # подсветка выбранной кнопки
        btns = [
            getattr(self, "btnNavHome", None),
            getattr(self, "btnNavCustomize", None),
            getattr(self, "btnNavDownload", None),
            getattr(self, "btnNavModel", None),
            getattr(self, "btnNavInfo", None),
        ]
        for i, b in enumerate(btns):
            if b is not None:
                b.setChecked(i == index)
    
    def _reflow_mode_ui(self, *_):
        """Переключение SOLO <-> DUAL: что показывать в UI."""
        dual = self.cbDual.isChecked()

        # --- Вкладка "Дом": какие модели видны ---

        # SOLO видно только в одиночном режиме
        for w in (self.lblSoloModel, self.cbModel,
                  self.lblSoloMmproj, self.cbMmproj):
            w.setVisible(not dual)

        # DUAL видно только в двойном режиме
        for w in (self.lblOcrModel, self.cbModelOCR,
                  self.lblOcrMmproj, self.cbMmprojOCR,
                  self.lblTrModel, self.cbModelTR):
            w.setVisible(dual)

        # --- Вкладка "LLM": какие группы настроек активны ---

        # SOLO-группы видны только в одиночном режиме
        for w in (getattr(self, "gbLLM", None),
                  getattr(self, "gbGen", None),
                  getattr(self, "gbSrv", None),
                  getattr(self, "btnPrompt", None),
                  getattr(self, "btnPreview", None)):
            if w is not None:
                w.setVisible(not dual)

        # DUAL-группы (OCR / TR) видны только в двойном режиме
        for w in (getattr(self, "gbOCR", None),
                  getattr(self, "gbTR", None),
                  getattr(self, "btnEditOCR", None),
                  getattr(self, "btnPrevOCR", None),
                  getattr(self, "btnEditTR", None),
                  getattr(self, "btnPrevTR", None)):
            if w is not None:
                w.setVisible(dual)
    
    def _load_prefs_into_ui(self):
        p = _load_prefs()
        dual = bool(p.get("dual_enabled", False))
        self.cbDual.setChecked(dual)
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
        pb = p.get("phrasebook_path")
        if pb: self.edPB.setText(pb)
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
        # OCR
        if (m := p.get("dual_ocr_model","")):
            i=self.cbModelOCR.findText(m);    self.cbModelOCR.setCurrentIndex(max(i,0))
        if (mm := p.get("dual_ocr_mmproj","")):
            i=self.cbMmprojOCR.findText(mm);  self.cbMmprojOCR.setCurrentIndex(max(i,0))
        self.edHost1.setText(p.get("dual_ocr_host","127.0.0.1"))
        try: self.spPort1.setValue(int(p.get("dual_ocr_port",8080)))
        except: pass
        try: self.spMaxTok1.setValue(int(p.get("dual_ocr_max_tokens", 512)))
        except: pass
        self.spCtx1.setValue(int(p.get("dual_ocr_ctx",8192)))
        self.spBatch1.setValue(int(p.get("dual_ocr_batch",256)))
        self.spUB1.setValue(int(p.get("dual_ocr_ubatch",64)))
        self.spPar1.setValue(int(p.get("dual_ocr_parallel",1)))
        self.spNGL1.setValue(int(p.get("dual_ocr_ngl",999)))
        self.spSeed1.setValue(int(p.get("dual_ocr_seed",0)))
        self.spSlot1.setValue(int(p.get("dual_ocr_slot",0)))
        self._ocr_prompt_override = p.get("dual_ocr_system_override","")
        # TR
        if (m := p.get("dual_tr_model","")):
            i=self.cbModelTR.findText(m);     self.cbModelTR.setCurrentIndex(max(i,0))
        self.edHost2.setText(p.get("dual_tr_host","127.0.0.1"))
        try: self.spPort2.setValue(int(p.get("dual_tr_port",8081)))  
        except: pass
        try: self.spMaxTok2.setValue(int(p.get("dual_tr_max_tokens", 1024)))
        except: pass
        self.spCtx2.setValue(int(p.get("dual_tr_ctx",6000)))
        self.spBatch2.setValue(int(p.get("dual_tr_batch",256)))
        self.spUB2.setValue(int(p.get("dual_tr_ubatch",64)))
        self.spPar2.setValue(int(p.get("dual_tr_parallel",1)))
        self.spNGL2.setValue(int(p.get("dual_tr_ngl",999)))
        self.spSeed2.setValue(int(p.get("dual_tr_seed",0)))
        self.spSlot2.setValue(int(p.get("dual_tr_slot",1)))
        self.spTO2.setValue(float(p.get("dual_tr_timeout_s", p.get("timeout_s", 30.0))))
        self.spTemp2.setValue(float(p.get("dual_tr_temp",      p.get("temp", 0.2))))
        self.spTopP2.setValue(float(p.get("dual_tr_top_p",     p.get("top_p", 0.9))))
        self.spTopK2.setValue(int(p.get("dual_tr_top_k",       p.get("top_k", 0))))
        self.spRP2.setValue(float(p.get("dual_tr_repeat_penalty", p.get("repeat_penalty", 1.0))))
        self._tr_prompt_override = p.get("dual_tr_system_override","")
        # константы
        self.spFps.setValue(float(p.get("fps", 1.0)))
        self.cbSmooth.setChecked(p.get("render_mode", "smooth") == "smooth")
        self.cbOverlayBoxes.setChecked(bool(p.get("draw_over_original", False)))
        self.spHandle.setValue(int(p.get("handle", 15)))
        self.spBorder.setValue(int(p.get("border_w", BORDER_WIDTH)))
        self.spSplit.setValue(int(p.get("split_border_w", SPLIT_BORDER_WIDTH)))
        
        bg_mode = p.get("bg_mode", PANEL_BG_MODE)
        idx = self.cbBgMode.findData(bg_mode)
        if idx < 0:
            idx = 0
        self.cbBgMode.setCurrentIndex(idx)

        # из сохранённого alpha (0..255) получаем уровень 1..10
        alpha = int(p.get("bg_alpha", PANEL_BG_ALPHA))
        alpha = max(0, min(255, alpha))
        level = 1 if alpha == 0 else max(1, min(10, round(alpha / 255.0 * 10)))
        self.spBgAlpha.setValue(level)

        # сразу применяем фон
        self._apply_bg_settings()
        self.cbBgMode.currentIndexChanged.connect(lambda _ : self._apply_bg_settings())
        self.spBgAlpha.valueChanged.connect(lambda _ : self._apply_bg_settings())
        
        globals()["PANEL_BG_MODE"]  = self.cbBgMode.currentData()
        globals()["PANEL_BG_ALPHA"] = int(self.spBgAlpha.value())
        
        box_mode = p.get("box_bg_mode", BOX_BG_MODE)
        idx2 = self.cbBoxBgMode.findData(box_mode)
        if idx2 < 0:
            idx2 = 0
        self.cbBoxBgMode.setCurrentIndex(idx2)

        alpha_box = int(p.get("box_bg_alpha", BOX_BG_ALPHA))
        alpha_box = max(0, min(255, alpha_box))
        level_box = 1 if alpha_box == 0 else max(1, min(10, round(alpha_box / 255.0 * 10)))
        self.spBoxBgAlpha.setValue(level_box)

        # сразу применяем фон под строками
        self._apply_box_bg_settings()
        self.cbBoxBgMode.currentIndexChanged.connect(lambda _ : self._apply_box_bg_settings())
        self.spBoxBgAlpha.valueChanged.connect(lambda _ : self._apply_box_bg_settings())

        self.overlay.bg_mode  = PANEL_BG_MODE
        self.overlay.bg_alpha = PANEL_BG_ALPHA

        try:
            for r in getattr(self.overlay, "regions", []):
                if getattr(r, "panel", None):
                    r.panel.apply_acrylic()
        except Exception as e:
            print("[UI] apply bg prefs error:", e)
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
                
    def _apply_bg_settings(self):
        """Применить текущие настройки фона ко всем панелям сразу."""
        global PANEL_BG_MODE, PANEL_BG_ALPHA

        mode = self.cbBgMode.currentData()
        level = int(self.spBgAlpha.value())  # 1..10

        # Маппинг 1–10 → 0..255 (линейно)
        level = max(1, min(10, level))
        alpha = int(round(level / 10.0 * 255))

        PANEL_BG_MODE = mode
        PANEL_BG_ALPHA = alpha

        self.overlay.bg_mode = PANEL_BG_MODE
        self.overlay.bg_alpha = PANEL_BG_ALPHA

        # Перекинуть настройки на все RegionPanel
        try:
            for r in getattr(self.overlay, "regions", []):
                panel = getattr(r, "panel", None)
                if panel is not None:
                    panel.apply_acrylic()
                    panel.update()
        except Exception as e:
            print("[UI] apply bg settings error:", e)
            
    def _apply_box_bg_settings(self):
        """Применить настройки фона под строками ко всем панелям."""
        global BOX_BG_MODE, BOX_BG_ALPHA

        mode = self.cbBoxBgMode.currentData()
        level = int(self.spBoxBgAlpha.value())  # 1..10

        level = max(1, min(10, level))
        alpha = int(round(level / 10.0 * 255))

        BOX_BG_MODE = mode
        BOX_BG_ALPHA = alpha

        # Перерисовать все RegionPanel, чтобы фон под строками обновился
        try:
            for r in getattr(self.overlay, "regions", []):
                panel = getattr(r, "panel", None)
                if panel is not None:
                    panel.update()
        except Exception as e:
            print("[UI] apply box-bg settings error:", e)
            
    def _collect_prefs(self) -> dict:
        prefs = {
            "font_family": self.fontFamily.currentFont().family(),
            "font_size": int(self.fontSize.value()),
            "font_bold": bool(self.fontBold.isChecked()),
            "model_path": self.cbModel.currentText().strip(),
            "mmproj_path": self.cbMmproj.currentText().strip(),
            "lore_path": self.edLore.text().strip(),
            "phrasebook_path": self.edPB.text().strip(),
            "max_tokens": int(self.spMaxTok.value()),
            "timeout_s": float(self.spTO.value()),
            "temp": float(self.spTemp.value()),
            "top_p": float(self.spTopP.value()),
            "fps": float(self.spFps.value()),
            "border_w": int(self.spBorder.value()),
            "split_border_w": int(self.spSplit.value()),
            "render_mode": "smooth" if self.cbSmooth.isChecked() else "instant",
            "draw_over_original": bool(self.cbOverlayBoxes.isChecked()),
            "handle": int(self.spHandle.value()),
            "bg_mode": self.cbBgMode.currentData(),
            "bg_alpha": int(round(max(1, min(10, self.spBgAlpha.value())) / 10.0 * 255)),
            "box_bg_mode": self.cbBoxBgMode.currentData(),
            "box_bg_alpha": int(round(max(1, min(10, self.spBoxBgAlpha.value())) / 10.0 * 255)),
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
            "port": int(self.spPort.value()),
        }

        # dual-mode
        prefs["dual_enabled"] = bool(self.cbDual.isChecked())
        prefs["dual_ocr_model"]  = self.cbModelOCR.currentText().strip()
        prefs["dual_ocr_mmproj"] = self.cbMmprojOCR.currentText().strip()
        prefs["dual_ocr_host"]   = self.edHost1.text().strip()
        prefs["dual_ocr_max_tokens"] = int(self.spMaxTok1.value())
        prefs["dual_ocr_port"]   = int(self.spPort1.value())
        prefs["dual_ocr_ctx"]    = int(self.spCtx1.value())
        prefs["dual_ocr_batch"]  = int(self.spBatch1.value())
        prefs["dual_ocr_ubatch"] = int(self.spUB1.value())
        prefs["dual_ocr_parallel"]=int(self.spPar1.value())
        prefs["dual_ocr_ngl"]    = int(self.spNGL1.value())
        prefs["dual_ocr_seed"]   = int(self.spSeed1.value())
        prefs["dual_ocr_slot"]   = int(self.spSlot1.value())
        prefs["dual_ocr_system_override"] = getattr(self, "_ocr_prompt_override", "")

        prefs["dual_tr_model"]  = self.cbModelTR.currentText().strip()
        prefs["dual_tr_host"]   = self.edHost2.text().strip()
        prefs["dual_tr_port"]   = int(self.spPort2.value())
        prefs["dual_tr_max_tokens"]  = int(self.spMaxTok2.value())
        prefs["dual_tr_ctx"]    = int(self.spCtx2.value())
        prefs["dual_tr_batch"]  = int(self.spBatch2.value())
        prefs["dual_tr_ubatch"] = int(self.spUB2.value())
        prefs["dual_tr_parallel"]=int(self.spPar2.value())
        prefs["dual_tr_ngl"]    = int(self.spNGL2.value())
        prefs["dual_tr_seed"]   = int(self.spSeed2.value())
        prefs["dual_tr_slot"]   = int(self.spSlot2.value())
        prefs["dual_tr_timeout_s"]       = float(self.spTO2.value())
        prefs["dual_tr_temp"]            = float(self.spTemp2.value())
        prefs["dual_tr_top_p"]           = float(self.spTopP2.value())
        prefs["dual_tr_top_k"]           = int(self.spTopK2.value())
        prefs["dual_tr_repeat_penalty"]  = float(self.spRP2.value())
        prefs["dual_tr_system_override"] = getattr(self, "_tr_prompt_override", "")

        # регионы
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
        _stop_llama_server2()
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
        for p in models: self.cbModelOCR.addItem(p)
        for p in projs:  self.cbMmprojOCR.addItem(p)
        for p in models: self.cbModelTR.addItem(p)

    def _browse_lore(self):
        path,_ = QFileDialog.getOpenFileName(self,"Выберите файл лора", str(BASE_DIR), "Text (*.txt);;All (*.*)")
        if path:
            try:
                p = Path(path)
                try: rel = p.relative_to(BASE_DIR); self.edLore.setText(str(rel))
                except Exception: self.edLore.setText(str(p))
            except Exception:
                self.edLore.setText(path)
        
    def _browse_phrasebook(self):
        path,_ = QFileDialog.getOpenFileName(self, "Выберите phrasebook", str(BASE_DIR), "Text (*.txt);;All (*.*)")
        if path:
            try:
                p = Path(path)
                try:
                    rel = p.relative_to(BASE_DIR)
                    self.edPB.setText(str(rel))
                except Exception:
                    self.edPB.setText(str(p))
            except Exception:
                self.edPB.setText(path)

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
        pb_path = (self.edPB.text() or "").strip()
        if pb_path and not os.path.isabs(pb_path):
            pb_path = str((BASE_DIR / pb_path).resolve())
        llm_cfg.phrasebook_path = pb_path
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
        
    def _edit_ocr_prompt(self):
        txt = getattr(self, "_ocr_prompt_override", "")
        dlg = QDialog(self); dlg.setWindowTitle("System (OCR) — редактирование")
        lay = QVBoxLayout(dlg); te=QPlainTextEdit(txt); lay.addWidget(te)
        btn=QPushButton("Сохранить"); lay.addWidget(btn)
        def _save():
            self._ocr_prompt_override = te.toPlainText().strip()
            p = _load_prefs(); p["dual_ocr_system_override"]=self._ocr_prompt_override; _save_prefs(p)
            QMessageBox.information(dlg,"OCR","Сохранено"); dlg.close()
        btn.clicked.connect(_save); dlg.exec()

    def _preview_ocr(self):
        from llm_adapter_dual import _OCR_SYSTEM_DEFAULT
        base = getattr(self, "_ocr_prompt_override", "") or _OCR_SYSTEM_DEFAULT
        QMessageBox.information(self,"OCR system",base)

    def _edit_tr_prompt(self):
        txt = getattr(self, "_tr_prompt_override", "")
        dlg = QDialog(self); dlg.setWindowTitle("System (Переводчик) — редактирование")
        lay = QVBoxLayout(dlg); te=QPlainTextEdit(txt); lay.addWidget(te)
        btn=QPushButton("Сохранить"); lay.addWidget(btn)
        def _save():
            self._tr_prompt_override = te.toPlainText().strip()
            p = _load_prefs(); p["dual_tr_system_override"]=self._tr_prompt_override; _save_prefs(p)
            try: reset_tr_system_cache()  # чтобы предпросмотр видел новые тексты
            except Exception: pass
            QMessageBox.information(dlg,"TR","Сохранено"); dlg.close()
        btn.clicked.connect(_save); dlg.exec()

    def _preview_tr(self):
        # соберём system так же, как в адаптере (с LORE/PHRASEBOOK)
        try:
            from llm_adapter_dual import _ensure_tr_system
            lp = (self.edLore.text() or "").strip()
            if lp and not os.path.isabs(lp): lp = str((BASE_DIR / lp).resolve())
            pb = (self.edPB.text() or "").strip()
            if pb and not os.path.isabs(pb): pb = str((BASE_DIR / pb).resolve())
            cfg = LLMConfigDual(
                server="http://127.0.0.1:8081",
                model=self.cbModelTR.currentText().strip(),
                system_override=(getattr(self,"_tr_prompt_override","") or None),
                lore_path=(lp or None),
                phrasebook_path=(pb or None),
            )
            reset_tr_system_cache()
            text = _ensure_tr_system(cfg)
        except Exception as e:
            text = f"(error: {e})"
        dlg = QDialog(self); dlg.setWindowTitle("System (Переводчик) — предпросмотр")
        lay = QVBoxLayout(dlg); te=QPlainTextEdit(text); te.setReadOnly(True); lay.addWidget(te)
        btn=QPushButton("Закрыть"); lay.addWidget(btn); btn.clicked.connect(dlg.close)
        dlg.exec()
    
    def _start(self):
        # применяем настройки
        global MAX_OCR_FPS, RENDER_MODE, HANDLE, SERVER_EXE, DRAW_OVER_ORIGINAL, PANEL_BG_MODE, PANEL_BG_ALPHA
        
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
        dual = self.cbDual.isChecked()

        MAX_OCR_FPS    = float(self.spFps.value())
        RENDER_MODE    = "smooth" if self.cbSmooth.isChecked() else "instant"
        HANDLE         = int(self.spHandle.value())
        DRAW_OVER_ORIGINAL = bool(self.cbOverlayBoxes.isChecked())
        self._apply_bg_settings()
        try:
            for r in getattr(self.overlay, "regions", []):
                if getattr(r, "panel", None):
                    r.panel.apply_acrylic()
        except Exception as e:
            print("[UI] reapply acrylic on start error:", e)
        self._apply_box_bg_settings()
        
        self.overlay.border_w = int(self.spBorder.value())
        self.overlay.split_border_w = int(self.spSplit.value())
        self.overlay.draw_over_original = bool(self.cbOverlayBoxes.isChecked())

        # применим лор
        lore_path = (self.edLore.text() or "").strip()
        if lore_path and not os.path.isabs(lore_path):
            lore_path = str((BASE_DIR / lore_path).resolve())
        llm_cfg.lore_path = lore_path
        
        pb_path = (self.edPB.text() or "").strip()
        if pb_path and not os.path.isabs(pb_path):
            pb_path = str((BASE_DIR / pb_path).resolve())
        llm_cfg.phrasebook_path = pb_path
        
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

        dual = self.cbDual.isChecked()
        _stop_llama_server()
        _stop_llama_server2()

        if dual and LLMConfigDual is not None:
            # -------- DUAL MODE: OCR + TR --------
            host1 = (self.edHost1.text() or "127.0.0.1").strip()
            port1 = int(self.spPort1.value())
            host2 = (self.edHost2.text() or "127.0.0.1").strip()
            port2 = int(self.spPort2.value())

            ocr_model   = self.cbModelOCR.currentText().strip()
            ocr_mmproj  = self.cbMmprojOCR.currentText().strip()
            tr_model    = self.cbModelTR.currentText().strip()

            cmd1 = rebuild_server_cmd(
                SERVER_EXE, ocr_model, ocr_mmproj,
                host=host1, port=port1,
                ctx=int(self.spCtx1.value()),
                batch=int(self.spBatch1.value()),
                ubatch=int(self.spUB1.value()),
                parallel=int(self.spPar1.value()),
                ngl=int(self.spNGL1.value()),
            )
            cmd2 = rebuild_server_cmd(
                SERVER_EXE, tr_model, "",          # <- переводчик БЕЗ mmproj
                host=host2, port=port2,
                ctx=int(self.spCtx2.value()),
                batch=int(self.spBatch2.value()),
                ubatch=int(self.spUB2.value()),
                parallel=int(self.spPar2.value()),
                ngl=int(self.spNGL2.value()),
            )

            print("[LLM] spawn OCR:", cmd1)
            print("[LLM] spawn TR :", cmd2)

            _ensure_llama_server(cmd1, f"http://{host1}:{port1}")
            _ensure_llama_server2(cmd2, f"http://{host2}:{port2}")

            # конфиги для воркера
            self.overlay.dual_mode = True
            self.overlay.ocr_cfg_dual = LLMConfigDual(
                server=f"http://{host1}:{port1}",
                model=ocr_model,
                timeout_s=float(self.spTO.value()),
                max_tokens=int(self.spMaxTok1.value()),
                slot_id=int(self.spSlot1.value()), seed=int(self.spSeed1.value()),
                system_override=(getattr(self,"_ocr_prompt_override","") or None),
            )
            self.overlay.tr_cfg_dual = LLMConfigDual(
                server=f"http://{host2}:{port2}",
                model=tr_model,
                timeout_s=float(self.spTO2.value()),
                max_tokens=int(self.spMaxTok2.value()),
                temp=float(self.spTemp2.value()),
                top_p=float(self.spTopP2.value()),
                top_k=int(self.spTopK2.value()),
                repeat_penalty=float(self.spRP2.value()),
                slot_id=int(self.spSlot2.value()),
                seed=int(self.spSeed2.value()),
                system_override=(getattr(self, "_tr_prompt_override", "") or None),
                lore_path=(llm_cfg.lore_path or None),
                phrasebook_path=(llm_cfg.phrasebook_path or None),
            )

            # прогрев (опционально)
            try: preload_prompt_cache_ocr(self.overlay.ocr_cfg_dual)
            except Exception as e: print("[DUAL] OCR warmup error:", e)
            try: preload_prompt_cache_tr(self.overlay.tr_cfg_dual)
            except Exception as e: print("[DUAL] TR warmup error:", e)

        else:
            # -------- SINGLE MODE ("всё в одном") --------
            host = (self.edHost.text() or "127.0.0.1").strip()
            port = int(self.spPort.value())
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

            print("[LLM] model:", model_path)
            print("[LLM] mmproj:", mmproj_path)

            llm_cfg.server = f"http://{host}:{port}"
            _ensure_llama_server(cmd, llm_cfg.server)

            # прогрев single
            try:
                if llm_cfg.enabled and llm_cfg.use_prompt_cache:
                    preload_prompt_cache(llm_cfg)
            except Exception as e:
                print("[LLM] warmup error:", e)

            self.overlay.dual_mode = False
        
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
            self.hide()
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
