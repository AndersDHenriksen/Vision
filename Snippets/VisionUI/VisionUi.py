import sys
import configparser
from pathlib import Path
import numpy as np
import cv2
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from Vision import FlowTools as ft, CameraTools, QtTools, FileTools, VisionTools as vt
from resources import visionui_resources as resources
try:
    import pyi_splash
    pyi_splash.close()
except ImportError:
    pass

VERSION = '0.0.1'
NAME = 'VisionUI'


class ConfigFile:

    def __init__(self, logger):
        self.path = Path(f'{NAME}_config.ini')
        self.config = configparser.ConfigParser()
        self.logger = logger
        self.image_folder = Path()
        self.n_image_folders_to_keep = 10000
        self.dark_mode = True
        self.SquareRealSize = 10
        if not self.path.exists():
            self.logger.warning("Config file not present.")
            self.save_file()
        self.load_file()

    def save_file(self):
        self.config['General'] = {"Dark_Mode": self.dark_mode,
                                  'Image_Folder': str(self.image_folder.absolute()),
                                  'N_Image_Folders_To_Keep': self.n_image_folders_to_keep,
                                  'Calibration_Square_Side_Length_Mm': self.SquareRealSize}
        with open(self.path, "w") as configfile:
            self.config.write(configfile)

    def load_file(self):
        self.config.read(self.path)
        self.dark_mode = self.config['General']['Dark_Mode'].lower() == 'true'
        self.image_folder = Path(self.config['General']['Image_Folder'])
        self.n_image_folders_to_keep = int(self.config['General']['N_Image_Folders_To_Keep'])
        self.SquareRealSize = float(self.config['General']['Calibration_Square_Side_Length_Mm'])
        self.logger.info(f"Config loaded from {self.path}")


class VisionUI(qtw.QMainWindow):
    done_signal = qtc.pyqtSignal(tuple)

    def __init__(self):
        super().__init__()

        self.cam = None
        self.last_save_path, self.last_load_path = qtc.QDir.homePath(), qtc.QDir.homePath()
        self.update_image_timer = qtc.QTimer(interval=200)
        self.update_image_timer.timeout.connect(self.update_image)

        self.icon_run = self.style().standardIcon(qtw.QStyle.SP_MediaPlay)
        self.icon_stop = self.style().standardIcon(qtw.QStyle.SP_MediaStop)
        icon_pc = self.style().standardIcon(qtw.QStyle.SP_ComputerIcon)
        icon_cancel = self.style().standardIcon(qtw.QStyle.SP_DialogCancelButton)
        icon_disc = self.style().standardIcon(qtw.QStyle.SP_DriveHDIcon)
        icon_trash = self.style().standardIcon(qtw.QStyle.SP_TrashIcon)
        icon_open = self.style().standardIcon(qtw.QStyle.SP_DialogOpenButton)
        icon_reload = self.style().standardIcon(qtw.QStyle.SP_BrowserReload)

        self.setWindowTitle(f'{NAME} {VERSION}')
        self.setWindowIcon(qtg.QIcon(":/icons/Camera_icon.ico"))

        main_column = qtw.QWidget()
        self.setGeometry(50, 50, 1500, 900)
        main_column.setLayout(qtw.QVBoxLayout())
        self.setCentralWidget(main_column)

        # Button row
        button_row = qtw.QWidget()
        main_column.layout().addWidget(button_row)
        button_row.setLayout(qtw.QHBoxLayout())

        pixmap_logo = qtg.QPixmap(':/images/Customer_logo.png')
        pixmap_logo = pixmap_logo.scaledToHeight(52, qtc.Qt.SmoothTransformation)
        customer_logo = qtw.QLabel(pixmap=pixmap_logo)
        button_row.layout().addWidget(customer_logo)

        button_row.layout().addSpacing(20)
        self.live_button = qtw.QPushButton(" &Live", clicked=self.run, shortcut=qtg.QKeySequence('Ctrl+l'), icon=self.icon_run, checkable=True)
        self.live_button.setToolTip("Ctrl+l. Initialize Live View")
        self.save_button = qtw.QPushButton("📷  &Save", clicked=self.save_image, shortcut=qtg.QKeySequence('Ctrl+s'))
        self.save_button.setToolTip("Ctrl+s. Save only the currently displayed image")
        self.compute_button = qtw.QPushButton(" &Analyze", clicked=self.analyze, shortcut=qtg.QKeySequence('Ctrl+a'), icon=icon_pc)
        self.compute_button.setToolTip("Ctrl+a. Analyze current camera image")
        self.calib_button = qtw.QPushButton("📏 Calibra&te", clicked=self.calibrate, shortcut=qtg.QKeySequence('Ctrl+t'))
        self.calib_button.setToolTip("Ctrl+t. Calculate the image to pixel factor")

        buttons = [self.live_button, self.save_button, self.compute_button, self.calib_button]

        [button_row.layout().addWidget(q) for q in buttons]
        button_row.layout().addSpacing(20)

        self.load_button = qtw.QPushButton(" &Open raw ", clicked=self.read_raws, shortcut=qtg.QKeySequence('Ctrl+o'), icon=icon_open)
        self.load_button.setToolTip("Ctrl+o. Load all *raw.png files from folders")
        self.export_button = qtw.QPushButton(" Export", clicked=self.export, icon=icon_disc)
        self.export_button.setToolTip("Saves current table to a flat file, and all images from the current table to a folder next to the .csv")
        self.delete_button = qtw.QPushButton(" Delete", clicked=self.delete, icon=icon_cancel)
        self.delete_button.setToolTip("Delete the highlighted row from the datatable")
        self.clear_button = qtw.QPushButton(" Clear", clicked=self.clear, icon=icon_trash)
        self.clear_button.setToolTip("Delete all data from the table")
        buttons2 = [self.load_button, self.export_button, self.delete_button, self.clear_button]

        [button_row.layout().addWidget(q) for q in buttons2]
        [q.setEnabled(False) for q in buttons + [self.export_button, self.delete_button, self.clear_button]]
        [b.setSizePolicy(qtw.QSizePolicy.MinimumExpanding, qtw.QSizePolicy.Preferred) for b in buttons + buttons2]

        button_row.layout().addStretch(3)
        self.operator = qtw.QLineEdit(placeholderText='Operator Initials')
        self.serial = qtw.QLineEdit(placeholderText='Serial Number')
        self.comment = qtw.QLineEdit(placeholderText='Comment')
        self.text_buttons = [self.operator, self.serial, self.comment]
        [button_row.layout().addWidget(w) for w in self.text_buttons]
        [w.setAlignment(qtc.Qt.AlignCenter) for w in self.text_buttons]
        [w.setFixedSize(150, 30) for w in self.text_buttons]

        pixmap_logo2 = qtg.QPixmap(':/images/ProInvent_logo.png')
        pixmap_logo2 = pixmap_logo2.scaledToHeight(52, qtc.Qt.SmoothTransformation)
        pro_invent_logo = qtw.QLabel(pixmap=pixmap_logo2)
        button_row.layout().addSpacing(20)
        button_row.layout().addWidget(pro_invent_logo)

        # Image field
        image_table_row = qtw.QWidget()
        main_column.layout().addWidget(image_table_row)
        image_table_row.setLayout(qtw.QHBoxLayout())

        self.image_view = QtTools.ImageViewer(self)
        self.image_view.setFixedSize(900, 780)
        image_table_row.layout().addWidget(self.image_view)

        # Table field
        headers = ['Timestamp', 'Operator', 'Serial', 'Result', 'Comment']

        self.table = qtw.QTableView()
        self.table_model = QtTools.TableModel(self.table, headers, resize_columns=True)
        self.table.clicked.connect(self.table_selected)
        self.image_data = []
        image_table_row.layout().addWidget(self.table)
        self.spinner = QtTools.QSpinner(self.table)

        # Text field
        self.log_output = qtw.QTextEdit(readOnly=True)
        main_column.layout().addWidget(self.log_output)
        self.log_output.setMaximumHeight(200)
        self.log_output.setMinimumHeight(80)
        # add_stdout_to_textfield(sys.stdout, self.log_output)
        self.logger = QtTools.SetupLogger(self.log_output, f'{NAME}.log').logger
        self.config = ConfigFile(self.logger)
        self.calibrator = CameraTools.CheckerboardCalibrator(self.config.path)
        self.done_signal.connect(self.analyze_done)
        self.show()
        self.init_camera()
        self.compute_button.setFocus()

    def init_camera(self):
        try:
            self.cam = CameraTools.CameraWrapper()
            self.cam.grab()  # test acquire
            self.logger.info("Camera connected")
            self.run(True)
            [button.setEnabled(True) for button in [self.live_button, self.save_button, self.compute_button, self.calib_button]]
        except Exception as e:
            self.logger.warning(f"Cannot connect to camera. Retrying in 5 secs. Error: {e}")
            qtc.QTimer.singleShot(5000, self.init_camera)

    def run(self, state):
        self.live_button.setIcon(self.icon_stop if state else self.icon_run)
        self.live_button.setChecked(state)
        if state:
            self.update_image_timer.start()
            self.table.clearSelection()
            self.update_table()
        else:
            self.update_image_timer.stop()

    def analyze(self):
        if not self.live_button.isChecked():
            self.logger.info("Starting live-view")
            return self.run(True)
        self.run(False)

        image = self.cam.grab()
        # image = cv2.imread(r"", cv2.IMREAD_GRAYSCALE)
        self.analyze_image(image)

    def calibrate(self):
        if not self.live_button.isChecked():
            self.logger.info("Starting live-view")
            return self.run(True)
        self.run(False)

        response = qtw.QMessageBox.warning(self, 'Calibration', 'You are about to calibrate the system. This will overwrite the current calibration value.\nIs the calibration target inserted?',
                                           qtw.QMessageBox.Yes | qtw.QMessageBox.Abort)
        if response == qtw.QMessageBox.Abort:
            return

        self.spinner.start()
        image = self.cam.grab()
        image_overlay = self.calibrator.calibrate_from_checkerboard(image, True, self.config.SquareRealSize)
        self.calibrator.export_to_config_file(self.config.path)
        self.logger.info(f"New calibration value: {self.calibrator.PixelsPerMm:.3f} px/mm")
        self.update_image(image_overlay)
        self.spinner.stop()

    @QtTools.OwnThread
    def analyze_image(self, image=None, image_path=None):
        self.spinner.start()
        self.logger.info(f"Analyzing {image_path or 'image'} ...")
        timestamp = ft.get_timestamp('-')
        try:
            if image_path:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                timestamp = image_path.stem.rstrip(' - raw').rstrip('raw')
            result_metrics, image_overlay = [0], image  # Change this from dummy result
        except Exception as e:
            self.logger.error(f"Analysis crashed. Error: {e}")
        else:
            self.done_signal.emit((timestamp, result_metrics, image, image_overlay))
        self.spinner.stop()
        # qtw.QApplication.beep()
        # qtc.QTimer.singleShot(5000, lambda: self.run(True))

    def analyze_done(self, results):
        timestamp, result_metrics, image_raw, image_overlay = results
        self.add_measurement(timestamp, result_metrics, image_raw, image_overlay)
        self.update_image(image_overlay)

    def update_image(self, image=None):
        if image is None:
            image = self.cam.grab()
        self.image_view.setImage(image)

    def add_measurement(self, timestamp, result, image_raw, image_overlay):
        self.image_data.insert(0, [timestamp, image_raw, image_overlay])
        data_list = [timestamp, self.operator.text(), self.serial.text(), *result, self.comment.text()]
        self.table_model._data.insert(0, data_list)
        self.update_table()

    def update_table(self):
        self.table_model.update_table()
        self.export_button.setEnabled(self.table_model.rowCount())
        self.clear_button.setEnabled(self.table_model.rowCount())
        self.delete_button.setEnabled(False)

    def delete(self):
        row = self.table.currentIndex().row()
        if row >= 0:
            self.image_data.pop(row)
            self.table_model._data.pop(row)
            self.update_table()

    def clear(self):
        self.image_data = []
        self.table_model._data = []
        self.update_table()

    def table_selected(self, item):
        self.delete_button.setEnabled(True)
        self.live_button.setChecked(False)
        self.run(False)
        self.update_image(self.image_data[item.row()][2])

    def save_image(self):
        image = self.camera.grab()
        filename, _ = qtw.QFileDialog.getSaveFileName(self, "Select the file to save to…", self.last_save_path, 'PNG File (*.png)')
        if filename == '':
            return
        self.last_save_path = str(Path(filename).parent)
        cv2.imwrite(filename, image)

    def export(self):
        filename = self.table_model.export()
        if filename is None:
            return
        image_folder = Path(filename).parent / Path(filename).stem
        image_folder.mkdir(exist_ok=True)
        print(f"Starting image export to {image_folder}, please wait ...")
        for row in range(len(self.image_data)):
            overlay, result = self.generate_overlay_image(row)
            img_name = f"{result.serial or self.image_data[row][0]}"
            # if result.comment:
            #     img_name += f" - {result.comment}"
            img_name = FileTools.make_filename_safe(img_name)
            cv2.imwrite(str(image_folder / f"{img_name} - raw.png"), cv2.cvtColor(self.image_data[row][1], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(image_folder / f"{img_name} - overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("Export Done")

    def read_raws(self):
        self.run(False)
        load_path = qtw.QFileDialog.getExistingDirectory(self, "Select a folder with *raw.png files", self.last_load_path)
        if not load_path:
            return
        self.last_load_path = load_path
        raw_paths = sorted(Path(self.last_load_path).rglob("*raw.png"))
        self.logger.info(f"Found {len(raw_paths)} images to analyze. Please wait ...")
        for image_path in raw_paths:
            self.analyze_image(image_path=image_path)

    def closeEvent(self, event):
        if self.cam:
            self.cam.stop()
        event.accept()


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    ui = VisionUI()
    if ui.config.dark_mode:
        QtTools.QDarkPalette(app=app)
    else:
        app.setStyle('Fusion')
    sys.exit(app.exec())
