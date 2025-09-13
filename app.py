from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPixmap, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QToolBar,
    QWidget,
    QVBoxLayout,
    QStyle,
    QSpinBox,
    QWidgetAction,
    QComboBox,
    QCheckBox,
)

from u2net.infer import U2NetBackgroundRemover
from u2net.downloader import ensure_default_weights


APP_NAME = "ClearCut - Background Remover"


class Worker(QThread):
    done = pyqtSignal(object, object, str)
    failed = pyqtSignal(str)

    def __init__(self, image_path: Path, target_size: int, feather: int, erode: int, dilate: int,
                 conservative: bool, fg_thresh: float, bg_thresh: float,
                 decon_strength: float, decon_band: int):
        super().__init__()
        self.image_path = image_path
        self.target_size = target_size
        self.feather = feather
        self.erode = erode
        self.dilate = dilate
        self.conservative = conservative
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.decon_strength = decon_strength
        self.decon_band = decon_band

    def run(self):
        try:
            remover = U2NetBackgroundRemover()
            img = Image.open(self.image_path)
            orig, cutout = remover.remove_bg(
                img,
                target_size=self.target_size,
                feather=self.feather,
                erode=self.erode,
                dilate=self.dilate,
                conservative=self.conservative,
                fg_thresh=self.fg_thresh,
                bg_thresh=self.bg_thresh,
                decontaminate_strength=self.decon_strength,
                decontaminate_band=self.decon_band,
            )
            self.done.emit(orig, cutout, str(self.image_path))
        except Exception as e:
            # Print full traceback to console for debugging
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 650)
        self._current_image: Optional[Image.Image] = None
        self._result_image: Optional[Image.Image] = None
        self._current_path: Optional[Path] = None
        self._hq: bool = True
        self._feather: int = 12
        self._edge: int = 0  # negative=expand (dilate), positive=shrink (erode)
        self._anti_halo: int = 70  # percent 0..100 -> strength 0..1 (stronger default)
        self._anti_halo_band: int = 12
        self._model_pref: str = "auto"  # 'auto' | 'u2netp' | 'u2net'
        self._keep_details: bool = True   # conservative refinement

        self._create_ui()

    def _create_ui(self):
        # Toolbar
        tb = QToolBar("Main")
        self.addToolBar(tb)

        open_act = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Bild öffnen", self)
        open_act.triggered.connect(self.on_open)
        tb.addAction(open_act)

        proc_act = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Hintergrund entfernen", self)
        proc_act.triggered.connect(self.on_process)
        tb.addAction(proc_act)

        save_act = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Als PNG speichern", self)
        save_act.triggered.connect(self.on_save)
        tb.addAction(save_act)

        # High Quality toggle
        self.hq_act = QAction("High Quality", self)
        self.hq_act.setCheckable(True)
        self.hq_act.setChecked(self._hq)
        self.hq_act.toggled.connect(self.on_toggle_hq)
        tb.addAction(self.hq_act)

        # Feather spinbox
        feather_label_act = QWidgetAction(self)
        feather_label = QLabel("Feather:")
        feather_label.setStyleSheet("QLabel{padding-left:8px;padding-right:4px}")
        feather_label_act.setDefaultWidget(feather_label)
        tb.addAction(feather_label_act)

        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(0, 40)
        self.feather_spin.setSingleStep(2)
        self.feather_spin.setValue(self._feather)
        self.feather_spin.valueChanged.connect(self.on_change_feather)
        feather_spin_act = QWidgetAction(self)
        feather_spin_act.setDefaultWidget(self.feather_spin)
        tb.addAction(feather_spin_act)

        # Edge adjust spinbox (shrink/expand mask)
        edge_label_act = QWidgetAction(self)
        edge_label = QLabel("Edge:")
        edge_label.setToolTip("Negativ = erweitern (dilate), Positiv = verkleinern (erode)")
        edge_label.setStyleSheet("QLabel{padding-left:8px;padding-right:4px}")
        edge_label_act.setDefaultWidget(edge_label)
        tb.addAction(edge_label_act)

        self.edge_spin = QSpinBox()
        self.edge_spin.setRange(-50, 50)
        self.edge_spin.setSingleStep(1)
        self.edge_spin.setValue(self._edge)
        self.edge_spin.valueChanged.connect(self.on_change_edge)
        edge_spin_act = QWidgetAction(self)
        edge_spin_act.setDefaultWidget(self.edge_spin)
        tb.addAction(edge_spin_act)

        # Model selection combo
        model_label_act = QWidgetAction(self)
        model_label = QLabel("Modell:")
        model_label.setStyleSheet("QLabel{padding-left:8px;padding-right:4px}")
        model_label_act.setDefaultWidget(model_label)
        tb.addAction(model_label_act)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Auto", "Klein (u2netp)", "Groß (u2net)"])
        self.model_combo.currentIndexChanged.connect(self.on_change_model)
        model_combo_act = QWidgetAction(self)
        model_combo_act.setDefaultWidget(self.model_combo)
        tb.addAction(model_combo_act)

        # Details behalten toggle (conservative refinement)
        details_act = QWidgetAction(self)
        details_check = QCheckBox("Details behalten")
        details_check.setChecked(self._keep_details)
        details_check.stateChanged.connect(self.on_toggle_details)
        details_act.setDefaultWidget(details_check)
        tb.addAction(details_act)

        # Anti-Halo strength (percent)
        ah_label_act = QWidgetAction(self)
        ah_label = QLabel("Anti-Halo:")
        ah_label.setStyleSheet("QLabel{padding-left:8px;padding-right:4px}")
        ah_label_act.setDefaultWidget(ah_label)
        tb.addAction(ah_label_act)

        self.anti_halo_spin = QSpinBox()
        self.anti_halo_spin.setRange(0, 100)
        self.anti_halo_spin.setSingleStep(5)
        self.anti_halo_spin.setValue(self._anti_halo)
        self.anti_halo_spin.valueChanged.connect(self.on_change_anti_halo)
        ah_spin_act = QWidgetAction(self)
        ah_spin_act.setDefaultWidget(self.anti_halo_spin)
        tb.addAction(ah_spin_act)

        # Central split view
        self.left_label = QLabel("Original")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_label.setStyleSheet("QLabel { background: #222; color: #ccc; border: 1px solid #444; }")

        self.right_label = QLabel("Ohne Hintergrund")
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_label.setStyleSheet("QLabel { background: #222; color: #ccc; border: 1px solid #444; }")

        splitter = QSplitter()
        splitter.addWidget(self.left_label)
        splitter.addWidget(self.right_label)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.addWidget(splitter)
        self.setCentralWidget(wrapper)

        # Menu
        file_menu = self.menuBar().addMenu("Datei")
        file_menu.addAction(open_act)
        file_menu.addAction(save_act)
        act_quit = QAction("Beenden", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        action_menu = self.menuBar().addMenu("Aktionen")
        action_menu.addAction(proc_act)
        action_menu.addAction(self.hq_act)

        help_menu = self.menuBar().addMenu("Hilfe")
        about_act = QAction("Über", self)
        about_act.triggered.connect(self.on_about)
        help_menu.addAction(about_act)

        # Prepare weights check in background
        self.progress = QProgressDialog("Initialisiere Modell...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.show()
        QApplication.processEvents()
        try:
            ensure_default_weights(lambda stage, dl, total: None)
        finally:
            self.progress.close()

    def on_about(self):
        QMessageBox.information(self, "Über", "ClearCut – Lokale Bild-Hintergrundentfernung mit U²-Net")

    def on_open(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Bild öffnen", str(Path.home()), "Bilder (*.png *.jpg *.jpeg)")
        if not fn:
            return
        try:
            img = Image.open(fn)
            self._current_image = img
            self._current_path = Path(fn)
            self._result_image = None
            self._update_previews()
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Konnte Bild nicht laden: {e}")

    def on_process(self):
        if self._current_image is None and self._current_path is None:
            QMessageBox.information(self, "Hinweis", "Bitte zuerst ein Bild öffnen.")
            return
        # Always reload from disk to avoid PIL file handles issues
        self.progress = QProgressDialog("Hintergrund wird entfernt...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.show()

        # High Quality: slightly more aggressive anti-halo but safe defaults
        if self._hq:
            target_size = 384
            fg_thresh, bg_thresh = 0.72, 0.25
            decon_strength, decon_band = self._anti_halo / 100.0, self._anti_halo_band
        else:
            target_size = 320
            fg_thresh, bg_thresh = 0.65, 0.25
            decon_strength, decon_band = self._anti_halo / 100.0, self._anti_halo_band
        # Edge adjustment: positive -> erode, negative -> dilate
        erode = self._edge if self._edge > 0 else 0
        dilate = -self._edge if self._edge < 0 else 0
        self.worker = Worker(
            self._current_path,
            target_size,
            self._feather,
            erode,
            dilate,
            self._keep_details,  # conservative mapping protects body details
            fg_thresh,
            bg_thresh,
            decon_strength,
            decon_band,
        )
        self.worker.done.connect(self.on_processed)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_processed(self, orig: Image.Image, cutout: Image.Image, path: str):
        self.progress.close()
        self._current_image = orig
        self._result_image = cutout
        self._current_path = Path(path)
        self._update_previews()

    def on_failed(self, msg: str):
        self.progress.close()
        QMessageBox.critical(self, "Fehler", f"Verarbeitung fehlgeschlagen: {msg}")

    def _update_previews(self):
        def to_qpix(img: Image.Image, maxw=1000, maxh=1000):
            from PyQt6.QtGui import QImage
            w, h = img.size
            scale = min(maxw / w, maxh / h, 1.0)
            if scale != 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                w, h = img.size
            rgba = img.convert("RGBA")
            data = rgba.tobytes("raw", "RGBA")
            qimage = QImage(data, w, h, QImage.Format.Format_RGBA8888)
            return QPixmap.fromImage(qimage)

        if self._current_image is not None:
            pm = to_qpix(self._current_image)
            self.left_label.setPixmap(pm)
        else:
            self.left_label.setText("Original")

        if self._result_image is not None:
            pm = to_qpix(self._result_image)
            self.right_label.setPixmap(pm)
        else:
            self.right_label.setText("Ohne Hintergrund")

    def on_save(self):
        if self._result_image is None:
            QMessageBox.information(self, "Hinweis", "Bitte erst Hintergrund entfernen.")
            return
        default = (self._current_path.parent / (self._current_path.stem + "_noBG.png")) if self._current_path else Path.home() / "output.png"
        fn, _ = QFileDialog.getSaveFileName(self, "Als PNG speichern", str(default), "PNG (*.png)")
        if not fn:
            return
        try:
            self._result_image.save(fn, format="PNG")
            QMessageBox.information(self, "Gespeichert", f"Gespeichert: {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Konnte nicht speichern: {e}")

    def on_toggle_hq(self, checked: bool):
        self._hq = checked

    def on_change_feather(self, val: int):
        self._feather = int(val)

    def on_change_edge(self, val: int):
        self._edge = int(val)

    def on_change_anti_halo(self, val: int):
        self._anti_halo = int(val)

    def on_change_model(self, idx: int):
        mapping = {0: "auto", 1: "u2netp", 2: "u2net"}
        self._model_pref = mapping.get(idx, "auto")

    def on_toggle_details(self, state: int):
        self._keep_details = (state != 0)


def main():
    # Set DPI rounding policy before creating the app (prevents warning on Windows)
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
