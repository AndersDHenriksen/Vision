import sys
import logging
import functools
from pathlib import Path
from logging.handlers import RotatingFileHandler
import numpy as np

from PySide6 import QtWidgets as qtw
from PySide6 import QtGui as qtg
from PySide6 import QtCore as qtc

from .QtSpinner import WaitingSpinner


WHITE =     qtg.QColor(255, 255, 255)
BLACK =     qtg.QColor(0, 0, 0)
RED =       qtg.QColor(255, 0, 0)
PRIMARY =   qtg.QColor(53, 53, 53)
SECONDARY = qtg.QColor(35, 35, 35)
TERTIARY =  qtg.QColor(42, 130, 218)
GREY =      qtg.QColor(128, 128, 128)
DARKGREY =  qtg.QColor(75, 75, 75)


class QDarkPalette(qtg.QPalette):
    """Dark palette for a Qt application meant to be used with the Fusion theme."""
    def __init__(self, *__args, app=None):
        super().__init__(*__args)

        # Set all the colors based on the constants in globals
        self.setColor(qtg.QPalette.Window,          PRIMARY)
        self.setColor(qtg.QPalette.WindowText,      WHITE)
        self.setColor(qtg.QPalette.Base,            SECONDARY)
        self.setColor(qtg.QPalette.AlternateBase,   PRIMARY)
        self.setColor(qtg.QPalette.ToolTipBase,     WHITE)
        self.setColor(qtg.QPalette.ToolTipText,     WHITE)
        self.setColor(qtg.QPalette.Text,            WHITE)
        self.setColor(qtg.QPalette.Button,          PRIMARY)
        self.setColor(qtg.QPalette.ButtonText,      WHITE)
        self.setColor(qtg.QPalette.BrightText,      RED)
        self.setColor(qtg.QPalette.Link,            TERTIARY)
        self.setColor(qtg.QPalette.Highlight,       TERTIARY)
        self.setColor(qtg.QPalette.HighlightedText, BLACK)
        self.setColor(qtg.QPalette.Disabled, qtg.QPalette.Text,       GREY)
        self.setColor(qtg.QPalette.Disabled, qtg.QPalette.ButtonText, GREY)
        self.setColor(qtg.QPalette.Disabled, qtg.QPalette.WindowText, GREY)
        self.setColor(qtg.QPalette.Disabled, qtg.QPalette.Button,     DARKGREY)
        self.setColor(qtg.QPalette.Disabled, qtg.QPalette.Base,       PRIMARY)

        if app:
            self.set_app(app)

    @staticmethod
    def set_stylesheet(app):
        css_rgb = lambda color:  "rgb({}, {}, {})".format(*color.getRgb())
        """Static method to set the tooltip stylesheet to a `QtWidgets.QApplication`."""
        app.setStyleSheet("QToolTip {{"
                          "color: {white};"
                          "background-color: {tertiary};"
                          "border: 1px solid {white};"
                          "}}".format(white=css_rgb(WHITE), tertiary=css_rgb(TERTIARY)))

    def set_app(self, app):
        """Set the Fusion theme and this palette to a `QtWidgets.QApplication`."""
        app.setStyle("Fusion")
        app.setPalette(self)
        self.set_stylesheet(app)


class QLed(qtw.QWidget):
    def __init__(self, parent=None, **kwargs):
        qtw.QWidget.__init__(self, parent, **kwargs)
        self.colour_dict = {-1: qtc.Qt.red, 0: qtc.Qt.gray, 1: qtc.Qt.green}
        self.current_color = self.colour_dict[0]
        self.sizeHint = lambda: qtc.QSize(50, 50)

    def setState(self, state):
        assert state in self.colour_dict.keys()
        self.current_color = self.colour_dict[state]
        self.update()

    def paintEvent(self, event):
        painter = qtg.QPainter(self)
        painter.setPen(qtg.QPen(self.parent().palette().button().color(), 2, qtc.Qt.SolidLine))
        painter.setRenderHint(qtg.QPainter.Antialiasing, True)

        radialGradient = qtg.QRadialGradient(qtc.QPoint(25, 25), 50)
        radialGradient.setColorAt(0.1, self.current_color)
        radialGradient.setColorAt(0.7, qtg.QColor(self.current_color).darker())
        painter.setBrush(qtg.QBrush(radialGradient))
        painter.drawEllipse(3, 3, 44, 44)


class SetupLogger(qtc.QObject):

    def __init__(self, log_q_text_edit=None, log_file_name=None):
        super().__init__()
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S",
                            format='%(asctime)s.%(msecs)03d | %(levelname)5s | %(message)s')
        self.logger = logging.getLogger()
        self.log_out = log_q_text_edit
        self.log_out.setFont(qtg.QFontDatabase.systemFont(qtg.QFontDatabase.FixedFont))
        self._text_field_stream = TextFieldStream(log_q_text_edit)

        # Set up log handler
        handlers = []
        if log_file_name is not None:
            handlers.append(RotatingFileHandler(log_file_name, maxBytes=5 * 1024 * 1024, backupCount=1))
        if log_q_text_edit is not None:
            handlers.append(self._text_field_stream)
        for handler in handlers:
            self.logger.addHandler(handler)
            self.logger.handlers[-1].setFormatter(self.logger.handlers[0].formatter)


class TextFieldStream(qtc.QObject, logging.Handler):

    def __init__(self, log_out):
        qtc.QObject.__init__(self)
        logging.Handler.__init__(self)
        self.log_out = log_out

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_out.setText(self.log_out.toPlainText() + msg + "\n")
            self.log_out.moveCursor(qtg.QTextCursor.End)
        except Exception:
            self.handleError(record)


class TagMonitor(qtc.QObject):  # Designed with EthernetIP in mind

    def __init__(self, tag, comm, python_class, interval_ms=50):
        super(TagMonitor, self).__init__()
        self.tag = tag
        self.comm = comm
        self.python_class = python_class
        self.change_signal = qtc.Signal(python_class)
        self.last_value = python_class(self.comm.read(self.tag))
        self.timer = qtc.QTimer()
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.check_tag)
        self.timer.start()

    def check_tag(self):
        value = self.python_class(self.comm.read(self.tag))
        if value != self.last_value:
            self.last_value = value
            self.change_signal.emit(value)


class Worker(qtc.QObject):
    start_signal = qtc.Signal(tuple, dict)

    def __init__(self, func, done_func=None, exception_func=print):
        super().__init__()
        self.thread = qtc.QThread()
        self.moveToThread(self.thread)
        self.thread.start()
        self.func = func
        self.result = None
        self.done_func = done_func
        self.exception_func = exception_func
        self.start_signal.connect(self._run)

    def __call__(self, *args, **kwargs):
        self.start_signal.emit(args, kwargs)

    @qtc.Slot(tuple, dict)
    def _run(self, args, kwargs):
        try:
            self.result = self.func(*args, **kwargs)
            if self.done_func:
                self.done_func(self.result)
        except Exception as e:
            if self.exception_func:
                self.exception_func(e)


class OwnThread(qtc.QObject):
    start_signal = qtc.Signal(tuple, dict)
    done_signal = qtc.Signal()

    def __init__(self, func, done_slot=None):
        super().__init__()
        self.orig_func = func
        self.instance_func = None
        self.result = None
        if done_slot is not None:
            self.done_signal.connect(done_slot)
        functools.update_wrapper(self, func)
        self.thread = qtc.QThread()
        self.moveToThread(self.thread)
        self.thread.start()
        self.start_signal.connect(self._run)

    def __call__(self, *args, **kwargs):
        self.result = self.start_signal.emit(args, kwargs)

    def __get__(self, instance, owner):
        # return functools.partial(self.__call__, instance)  # Alternative if no need to access func.done_signal
        self.instance_func = self.instance_func or instance
        return self

    @qtc.Slot(tuple, dict)
    def _run(self, args, kwargs):
        self.orig_func(self.instance_func, *args, **kwargs)
        self.done_signal.emit()


class QTimer(qtc.QTimer):  # Timer that can be started/stopped from all threads
    _signal_start = qtc.Signal()
    _signal_stop = qtc.Signal()

    def __init__(self, *args, timeout_call=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal_start.connect(self._start_slot)
        self._signal_stop.connect(self._stop_slot)
        if timeout_call:
            self.connect(timeout_call)

    @qtc.Slot()
    def _start_slot(self):
        super().start()

    @qtc.Slot()
    def _stop_slot(self):
        super().stop()

    def start(self):
        self._signal_start.emit()

    def stop(self):
        self._signal_stop.emit()


class QSpinner(WaitingSpinner):  # Spinner with new default values and that can be started/stopped from all threads
    _signal_start = qtc.Signal()
    _signal_stop = qtc.Signal()

    def __init__(self, parent, center_on_parent=True, disable_parent_when_spinning=True,
                 modality=qtc.Qt.NonModal, roundness=100., fade=80., lines=20,
                 line_length=50, line_width=4, radius=50, speed=1, color=(50, 50, 255)):
        super().__init__(parent, center_on_parent, disable_parent_when_spinning, modality, roundness, fade,
                         lines, line_length, line_width, radius, speed, color=qtg.QColor(*color))
        self._signal_start.connect(self._start_slot)
        self._signal_stop.connect(self._stop_slot)

    @qtc.Slot()
    def _start_slot(self):
        super().start()

    @qtc.Slot()
    def _stop_slot(self):
        super().stop()

    def start(self):
        self._signal_start.emit()

    def stop(self):
        self._signal_stop.emit()


class TableModel(qtc.QAbstractTableModel):
    _signal_update = qtc.Signal()

    def __init__(self, table_view, header_list, numbering='ascending', n_digits=2, resize_columns=False, data=None):
        assert numbering in [None, 'ascending', 'descending']
        super(TableModel, self).__init__()
        self._data = data or []
        self.n_digits = n_digits
        self.header_list = header_list
        self.numbering = numbering
        self.last_save_path = qtc.QDir.homePath()
        self.table_view = table_view
        self.table_view.setModel(self)
        self.resize_columns = resize_columns
        self.centered_columns = []
        self.min_column_width = self.table_view.columnWidth(0)
        self._signal_update.connect(self._update_table_slot)

    def data(self, index, role):
        if role == qtc.Qt.DisplayRole:
            data_point = self._data[index.row()][index.column()]
            return data_point if isinstance(data_point, str) else f"{data_point:.{self.n_digits}f}"
        elif role == qtc.Qt.TextAlignmentRole and index.column() in self.centered_columns:
            return qtc.Qt.AlignCenter

    def setData(self, index, value, role):
        if role == qtc.Qt.EditRole:
            try:
                self._data[index.row()][index.column()] = float(value)
            except ValueError:
                self._data[index.row()][index.column()] = value
            return True

    def rowCount(self, index=None):
        return len(self._data) if self._data else 0

    def columnCount(self, index=None):
        return len(self.header_list)

    def flags(self, index):
        return qtc.Qt.ItemIsSelectable | qtc.Qt.ItemIsEnabled | qtc.Qt.ItemIsEditable

    def headerData(self, section, orientation, role):
        if role == qtc.Qt.DisplayRole:
            if orientation == qtc.Qt.Horizontal:
                return self.header_list[section]
            if orientation == qtc.Qt.Vertical and self.numbering:
                return str(section + 1) if self.numbering == 'ascending' else str(self.rowCount() - section)

    def update_table(self):
        self._signal_update.emit()

    @qtc.Slot()
    def _update_table_slot(self):
        self.layoutChanged.emit()
        if self.resize_columns:
            self.table_view.resizeColumnsToContents()
            self.table_view.horizontalHeader().setMinimumSectionSize(self.min_column_width)

    def export(self, filename=None, allow_excel_export=True):
        extensions = 'CSV File (*.csv);;Excel File (*.xlsx)' if allow_excel_export else 'CSV File (*.csv)'
        if not isinstance(filename, (str, Path)):
            filename, _ = qtw.QFileDialog.getSaveFileName(self.table_view, "Select the file to save to…", self.last_save_path, extensions)
        if filename == '':
            return
        self.last_save_path = str(Path(filename).parent)
        if filename[-3:] == 'csv':
            import numpy as np
            np.savetxt(filename, self._data, fmt='%s', delimiter=', ', header=", ".join(self.header_list))
        elif filename[-4:] == 'xlsx':
            from openpyxl import Workbook
            wb = Workbook()
            ws1 = wb.active
            [ws1.append(d) for d in [self.header_list] + self._data]
            wb.save(filename)
        return filename


class ImageViewer(qtw.QGraphicsView):
    imageClicked = qtc.Signal(qtc.QPoint)
    _signal_update = qtc.Signal(bool)

    def __init__(self, parent, use_fast_zoom=False):
        super(ImageViewer, self).__init__(parent)
        self.media = None
        self.use_fast_zoom = use_fast_zoom
        self._zoom = 0
        self._max_zoom = 0
        self._scene = qtw.QGraphicsScene(self)
        self._image = qtw.QGraphicsPixmapItem()
        self._scene.addItem(self._image)
        self.setScene(self._scene)
        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(qtg.QBrush(qtg.QColor(30, 30, 30)))
        self.setFrameShape(qtw.QFrame.NoFrame)
        self._signal_update.connect(self._update_scale_slot)

    def _update_scale_quality(self, do_zoom_in=True):
        if self._zoom > self._max_zoom + do_zoom_in:
            return self.scale(1.25, 1.25) if do_zoom_in else self.scale(0.8, 0.8)

        center_point = self.mapToScene(self.size().width()//2, self.size().height()//2) * (1.25 if do_zoom_in else 0.8)
        resize_to = qtc.QSizeF(self.size() * 1.25 ** self._zoom).toSize()
        self._image.setPixmap(self.media.scaled(resize_to, qtc.Qt.KeepAspectRatio, qtc.Qt.SmoothTransformation))

        rect = qtc.QRectF(self._image.pixmap().rect())
        self.setSceneRect(rect)
        self.centerOn(center_point)

    def _update_scale_fast(self, do_zoom_in=True):
        if self._zoom > 0:
            return self.scale(1.25, 1.25) if do_zoom_in else self.scale(0.8, 0.8)

        self.resetTransform()
        self._image.setPixmap(self.media)
        factor = min(self.size().width() / self.media.size().width(), self.size().height() / self.media.size().height())
        self.scale(factor, factor)

    @qtc.Slot(bool)
    def _update_scale_slot(self, do_zoom_in=True):
        if self.media is None:
            return
        self._update_scale_fast(do_zoom_in) if self.use_fast_zoom else self._update_scale_quality(do_zoom_in)

    def _update_scale(self, do_zoom_in=True):
        self._signal_update.emit(do_zoom_in)

    def setImage(self, pixmap=None):
        self._zoom = 0
        if isinstance(pixmap, np.ndarray):
            h, w, *_ = pixmap.shape
            if pixmap.ndim == 2:  # if problem here consider the slower: https://pypi.org/project/qimage2ndarray/
                # pixmap = qtg.QImage(pixmap.data, w, h, bytesPerLine=w, format=qtg.QImage.Format_Grayscale8)
                pixmap = qtg.QImage(pixmap.data.tobytes(), w, h, w, qtg.QImage.Format_Grayscale8)
            else:
                # pixmap = qtg.QImage(pixmap.data, w, h, bytesPerLine=3 * w, format=qtg.QImage.Format_RGB888)
                pixmap = qtg.QImage(pixmap.data.tobytes(), w, h, 3 * w, qtg.QImage.Format_RGB888)
        if isinstance(pixmap, qtg.QImage):
            pixmap = qtg.QPixmap.fromImage(pixmap)

        self.media = pixmap
        self.setDragMode(qtw.QGraphicsView.NoDrag if pixmap is None or pixmap.isNull() else qtw.QGraphicsView.ScrollHandDrag)
        self._max_zoom = round(max(np.log(self.media.size().width() / self.size().width())/np.log(1.25),
                                   np.log(self.media.size().height() / self.size().height()) / np.log(1.25)))
        self.resetTransform()
        self._update_scale()

    def wheelEvent(self, event):
        do_zoom_in = event.angleDelta().y() > 0
        self._zoom += 1 if do_zoom_in else -1
        if self.media is None or self._zoom < 0:
            self._zoom = 0
            return
        # self.centerOn(self.mapToScene(event.position().x(), event.position().y()))  # Zoom on cursor
        self._update_scale(do_zoom_in)

    def mousePressEvent(self, event):
        if self._image.isUnderMouse():
            self.imageClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(ImageViewer, self).mousePressEvent(event)
