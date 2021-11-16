import sys
import logging
from logging.handlers import RotatingFileHandler

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc


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


class SetupLogger(qtc.QObject):

    def __init__(self, log_q_text_edit=None, log_file_name=None):
        super().__init__()
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s | %(levelname)5s | %(message)s')
        self.logger = logging.getLogger()
        self.log_out = log_q_text_edit
        self.log_out.setFont(qtg.QFontDatabase.systemFont(qtg.QFontDatabase.FixedFont))
        self._text_field_stream = TextFieldStream()

        # Set up log handler
        handlers = []
        if log_file_name is not None:
            handlers.append(RotatingFileHandler(log_file_name, maxBytes=5 * 1024 * 1024, backupCount=1))
        if log_q_text_edit is not None:
            handlers.append(self._text_field_stream)
        for handler in handlers:
            self.logger.addHandler(handler)
            self.logger.handlers[-1].setFormatter(self.logger.handlers[0].formatter)
        # self._text_field_stream.text_arrived.connect(self.log_append_text)
        self._text_field_stream.text_arrived.connect(lambda msg: self.log_append_text(msg))  # Lambda needed when no reference is kept

    @qtc.pyqtSlot(str)
    def log_append_text(self, msg):
        self.log_out.setText(self.log_out.toPlainText() + msg + "\n")
        self.log_out.moveCursor(qtg.QTextCursor.End)


class TextFieldStream(qtc.QObject, logging.Handler):
    text_arrived = qtc.pyqtSignal(str)

    def __init__(self):
        qtc.QObject.__init__(self)
        logging.Handler.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.text_arrived.emit(msg)
        except Exception:
            self.handleError(record)


class TagMonitor(qtc.QObject):  # Designed with EthernetIP in mind

    def __init__(self, tag, comm, python_class, interval_ms=50):
        super(TagMonitor, self).__init__()
        self.tag = tag
        self.comm = comm
        self.python_class = python_class
        self.change_signal = qtc.pyqtSignal(python_class)
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
    start_signal = qtc.pyqtSignal(tuple, dict)

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

    @qtc.pyqtSlot(tuple, dict)
    def _run(self, args, kwargs):
        try:
            self.result = self.func(*args, **kwargs)
            if self.done_func:
                self.done_func(self.result)
        except Exception as e:
            if self.exception_func:
                self.exception_func(e)
