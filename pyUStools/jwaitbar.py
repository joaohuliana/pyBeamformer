from PyQt6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QApplication
from PyQt6.QtGui import QGuiApplication
class ProgressBarWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()

        # Create a QProgressBar
        self.progress_bar = QProgressBar(self)
        vbox.addWidget(self.progress_bar)

        self.setLayout(vbox)

        self.setGeometry(300, 500, 300, 80)


    def center_on_screen(self):
        # Get the geometry of the screen
        primary_screen = QGuiApplication.primaryScreen()
        screen_rect = primary_screen.availableGeometry()

        # Center the window on the screen
        self.move((screen_rect.width() - self.width()) // 2, (screen_rect.height() - self.height()) // 2)

    def set_title(self, title):
        self.setWindowTitle(title)
    def set_progress_value(self, value):
        self.progress_bar.setValue(value)

    def close_window(self):
        self.close()