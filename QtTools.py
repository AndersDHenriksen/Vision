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
        self.log_out.setStyleSheet('font-family: Monospace;')  # below command should be enough unless stylesheet is set
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
            self.log_out.setPlainText(self.log_out.toPlainText() + msg + "\n")
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
        if numbering is None:
            self.table_view.verticalHeader().hide()

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

    def __init__(self, parent, scaling_method='normal'):
        assert scaling_method in ['fast', 'normal', 'slow']
        super(ImageViewer, self).__init__(parent)
        self.media = None
        self._min_scale = 0
        self._current_scale = 0
        self._scene = qtw.QGraphicsScene(self)
        self._image = qtw.QGraphicsPixmapItem()
        self.scaling_method = scaling_method
        if scaling_method in ['normal', 'slow']:
            self._image.setTransformationMode(qtc.Qt.SmoothTransformation)
        self._scene.addItem(self._image)
        self.setScene(self._scene)
        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)  # TODO is this needed?
        self.setResizeAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(qtg.QBrush(qtg.QColor(30, 30, 30)))
        self.setFrameShape(qtw.QFrame.NoFrame)

    def reset_zoom(self):
        self.resetTransform()
        self._min_scale = min(self.size().width() / self.media.size().width(), self.size().height() / self.media.size().height())
        self.scale(self._min_scale, self._min_scale)

    def setImage(self, pixmap=None):
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
        self._image.setPixmap(self.media)
        self.change_drag_mode(rubber_instead_of_scrolling=True)
        self.reset_zoom()

    def change_drag_mode(self, rubber_instead_of_scrolling):
        is_media_present = self.media is None or self.media.isNull()
        if rubber_instead_of_scrolling:
            self.setDragMode(qtw.QGraphicsView.NoDrag if is_media_present else qtw.QGraphicsView.RubberBandDrag)
        else:
            self.setDragMode(qtw.QGraphicsView.NoDrag if is_media_present else qtw.QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        s = 1.25
        if event.angleDelta().y() < 0:  # zoom out
            s = max(0.8, self._min_scale / self.transform().m11())
        # self.centerOn(self.mapToScene(event.position().x(), event.position().y()))  # Zoom on cursor
        self.scale(s, s)

    def paintEvent(self, event):
        if self.scaling_method == 'slow':
            if not self.media:
                return
            if self._current_scale < 1 and self.viewportTransform().m11() >= 1:
                self._current_scale = 1
                self._image.setPixmap(self.media)
            # Down-sample image using best but expensive method, then upscale to keep scaling and view correct
            elif self.viewportTransform().m11() < 1 and self.viewportTransform().m11() != self._current_scale:
                self._current_scale = self.viewportTransform().m11()
                scaled_size = self._image.pixmap().size() * self.viewportTransform().m11()
                scaled_img = self.media.scaled(scaled_size, qtc.Qt.KeepAspectRatio, qtc.Qt.SmoothTransformation)
                self._image.setPixmap(scaled_img.scaled(self.media.size(), qtc.Qt.KeepAspectRatio, qtc.Qt.FastTransformation))
        super(ImageViewer, self).paintEvent(event)

    def mousePressEvent(self, event):
        if self._image.isUnderMouse():
            self.imageClicked.emit(self.mapToScene(event.pos()).toPoint())
        if event.button() == qtc.Qt.MouseButton.MiddleButton:
            self.change_drag_mode(rubber_instead_of_scrolling=False)
            event = qtg.QMouseEvent(qtc.QEvent.MouseButtonRelease, qtc.QPointF(event.pos()), qtc.Qt.LeftButton, event.buttons(), qtc.Qt.KeyboardModifiers())
        super(ImageViewer, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rubberBandRect().size():
            self.fitInView(self.mapToScene(self.rubberBandRect()).boundingRect(), qtc.Qt.KeepAspectRatio)
        if event.button() == qtc.Qt.MouseButton.MiddleButton:
            self.change_drag_mode(rubber_instead_of_scrolling=True)
        super(ImageViewer, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.reset_zoom()
        super(ImageViewer, self).mouseDoubleClickEvent(event)


class TitleBar(qtw.QWidget):
    def __init__(self, main_window, title, pixmap_logo=None):
        super().__init__()
        self.main_window = main_window
        self.setFixedHeight(125)

        stack_layout = qtw.QStackedLayout(self)
        stack_layout.setStackingMode(qtw.QStackedLayout.StackAll)
        title_widget = qtw.QWidget()
        image_button_widget = qtw.QWidget()
        image_button_widget.setStyleSheet("background: transparent;")
        stack_layout.addWidget(image_button_widget)
        stack_layout.addWidget(title_widget)

        title_layout = qtw.QHBoxLayout(title_widget)
        self.title = qtw.QLabel(title, parent=title_widget)
        self.title.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet('font-size: 48px; font-weight: bold; text-align: center;')  # background-color: yellow;
        title_layout.addWidget(self.title)

        layout = qtw.QHBoxLayout(image_button_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if pixmap_logo:
            pixmap_logo = pixmap_logo.scaledToHeight(100, qtc.Qt.SmoothTransformation)
            logo = qtw.QLabel(pixmap=pixmap_logo)
            layout.addWidget(logo)

        layout.addStretch()
        self.minimum_button = qtw.QPushButton(self, text='🗕', clicked=self.main_window.showMinimized)
        self.maximum_button = qtw.QPushButton(self, text='🗖', clicked=self._maximize)
        self.full_screen_button = qtw.QPushButton(self, text='⛶', clicked=self._full_screen)
        self.close_button = qtw.QPushButton(self, text='🗙', clicked=self.main_window.close)
        self.buttons = [self.minimum_button, self.maximum_button, self.full_screen_button, self.close_button]
        for b in self.buttons:
            layout.addWidget(b, alignment=qtc.Qt.AlignmentFlag.AlignTop)
        self.apply_button_stylesheets()

    def apply_button_stylesheets(self):
        common_btn_style = f"QPushButton {{ background-color: transparent; border: 0;width: 50;height: 50; font-size: 16px}}"
        for b in [self.minimum_button, self.maximum_button, self.full_screen_button]:
            b.setStyleSheet(f"{common_btn_style}QPushButton:hover {{background-color: #ddd;}}QPushButton:pressed {{background-color: #aaa;}}")
        self.close_button.setStyleSheet(f"{common_btn_style}QPushButton:hover {{background-color: #f00;color: white;}}QPushButton:pressed {{background-color: #f44;color: white;}}")

    def _maximize(self):
        if self.main_window.isMaximized():
            self.main_window.showNormal()
            self.maximum_button.setText('🗖')
        else:
            self.main_window.showMaximized()
            self.maximum_button.setText('🗗')

    def _full_screen(self):
        if self.main_window.isFullScreen():
            self.main_window.showNormal()
        else:
            self.main_window.showFullScreen()


class FramelessMainWindow(qtw.QMainWindow):
    def __init__(self, title='Vision Inspection', pixmap_logo=None):
        super().__init__()
        self._inside_margin = False
        self._top = False
        self._bottom = False
        self._left = False
        self._right = False
        self._margin = 4
        self.setMouseTracking(True)
        self.setWindowFlags(self.windowFlags() | qtc.Qt.FramelessWindowHint)

        self.main_widget = qtw.QWidget()
        self.main_layout = qtw.QVBoxLayout(self.main_widget)
        self.main_widget.setMouseTracking(True)  # Needed to propagate the mouseMove event to FramelessMainWindow
        self.setCentralWidget(self.main_widget)
        self.title_bar = TitleBar(self, title, pixmap_logo)
        self.main_layout.addWidget(self.title_bar)

    def check_mouse_in_margin(self, p):
        if self.isMaximized() or self.isFullScreen():
            return

        self._inside_margin = True
        self._left = p.x() <= self._margin
        self._top = p.y() <= self._margin
        self._right = self.width() - p.x() <= self._margin
        self._bottom = self.height() - p.y() <= self._margin

        if (self._top and self._left) or (self._bottom and self._right):
            self.setCursor(qtc.Qt.SizeFDiagCursor)
        elif (self._top and self._right) or (self._bottom and self._left):
            self.setCursor(qtc.Qt.SizeBDiagCursor)
        elif self._left or self._right:
            self.setCursor(qtc.Qt.SizeHorCursor)
        elif self._top or self._bottom:
            self.setCursor(qtc.Qt.SizeVerCursor)
        else:
            self._inside_margin = False
            self.unsetCursor()

    def mousePressEvent(self, e):
        if e.button() == qtc.Qt.LeftButton:
            if self._inside_margin:
                self._resize()
            elif not (self.isMaximized() or self.isFullScreen()):
                window = self.window().windowHandle()
                window.startSystemMove()
        return super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        self.check_mouse_in_margin(e.position())
        return super().mouseMoveEvent(e)

    def _resize(self):
        window = self.window().windowHandle()
        edges = qtc.Qt.Edge(0)
        for b, e in zip([self._top, self._left, self._right, self._bottom],
                        [qtc.Qt.Edge.TopEdge, qtc.Qt.Edge.LeftEdge, qtc.Qt.Edge.RightEdge, qtc.Qt.Edge.BottomEdge]):
            if b:
                edges |= e
        window.startSystemResize(edges)

    def set_frame_color(self, color):
        if isinstance(color, str):
            color = qtg.QColor(color)
        p = qtg.QPalette()
        b = qtg.QBrush(color)
        p.setBrush(qtg.QPalette.Window, b)
        self.setPalette(p)

    def get_frame_color(self):
        return self.palette().color(qtg.QPalette.Window)


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    pixmap_logo = qtg.QPixmap("resources/ProInvent_Logo_Transparent.png")
    window = FramelessMainWindow(pixmap_logo=pixmap_logo)
    window.main_layout.addWidget(qtw.QTextEdit())
    window.setGeometry(300, 300, 1200, 300)
    window.show()
    sys.exit(app.exec())
