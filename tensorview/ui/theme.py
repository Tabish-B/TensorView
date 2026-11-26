"""Desktop theme and reusable text and button styling."""

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton

BG = "#10141E"
SURFACE = "#161C29"
PLOT = "#121824"
BORDER = "#2A3447"
TEXT = "#E8EFFA"
MUTED = "#95A5BF"
SUBTLE = "#7889A5"
ACCENT = "#55DEC5"
YELLOW = "#FFC98B"
BLUE = "#819DFF"


def label(text="", role="", wrap=False) -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(wrap)
    if role:
        widget.setObjectName(role)
    return widget


def button(text, role="", tooltip=None) -> QPushButton:
    widget = QPushButton(text)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if role:
        widget.setObjectName(role)
    if tooltip:
        widget.setToolTip(tooltip)
    return widget


class ComboBox(QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(MUTED if self.isEnabled() else SUBTLE), 1.4))
        x, y = self.width() - 11, self.height() / 2
        painter.drawPolyline(QPolygonF([QPointF(x - 3, y - 1), QPointF(x, y + 2), QPointF(x + 3, y - 1)]))


def mono() -> QFont:
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)
    app.setStyleSheet(STYLES)


STYLES = f"""
QWidget {{ color: {TEXT}; selection-background-color: #284759; selection-color: #F0FFFC; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget#workspace {{ background: {BG}; }}
QFrame#header {{ background: #141A26; border-bottom: 1px solid {BORDER}; }}
QFrame#sidebar {{ background: #141A26; border-right: 1px solid {BORDER}; }}
QFrame#footer {{ background: #141A26; border-top: 1px solid {BORDER}; }}
QFrame#panel, QFrame#metric {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#divider {{ border: 0; background: {BORDER}; min-height: 1px; max-height: 1px; }}
QLabel#brand {{ font-weight: 800; font-size: 18px; color: #F3F7FE; }}
QLabel#caption {{ color: {MUTED}; font-size: 11px; }}
QLabel#section {{ color: {MUTED}; font-size: 11px; font-weight: 750; letter-spacing: 1px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#subtle {{ color: {SUBTLE}; font-size: 11px; }}
QLabel#title {{ font-size: 26px; font-weight: 750; }}
QLabel#detailTitle {{ font-size: 22px; font-weight: 750; }}
QLabel#heading {{ font-size: 14px; font-weight: 700; }}
QLabel#metricValue {{ font-size: 17px; font-weight: 700; }}
QLabel#accent {{ color: {ACCENT}; }}
QLabel#error {{ color: #FFA7B1; }}
QPushButton {{
    background: #20293A; border: 1px solid #354258; border-radius: 6px;
    color: {TEXT}; padding: 6px 12px; min-height: 19px; font-weight: 620;
}}
QPushButton:hover {{ background: #2A364C; border-color: #52637E; }}
QPushButton:pressed {{ background: #172132; }}
QPushButton:focus, QComboBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #637189; background: #192130; border-color: #263146; }}
QPushButton#primary {{ color: #0D2927; background: {ACCENT}; border-color: {ACCENT}; font-weight: 800; }}
QPushButton#primary:hover {{ background: #83F3DC; }}
QPushButton#primary:disabled {{ background: #244B4C; color: #A2BBB9; border-color: #315959; }}
QPushButton#flat {{ background: transparent; border-color: transparent; color: {MUTED}; }}
QPushButton#flat:hover {{ background: #29364A; color: {TEXT}; }}
QPushButton#nav {{
    background: transparent; border-color: transparent; color: {MUTED}; min-height: 24px;
    padding-left: 14px; padding-right: 14px;
}}
QPushButton#nav:hover {{ background: #243044; color: {TEXT}; }}
QPushButton#nav:checked {{ background: #223343; border-color: #344B5E; color: {ACCENT}; }}
QLineEdit, QComboBox {{
    background: #101725; color: {TEXT}; border: 1px solid #303D53; border-radius: 6px;
    padding: 6px 8px; min-height: 17px;
}}
QComboBox {{ padding-right: 18px;
}}
QLineEdit {{ placeholder-text-color: #899BB6; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border: 0;
}}
QComboBox::down-arrow {{ width: 0; height: 0; }}
QComboBox QAbstractItemView {{ background: #1B2433; border: 1px solid {BORDER}; outline: 0; }}
QComboBox:disabled, QLineEdit:disabled {{ color: {SUBTLE}; }}
QTreeWidget, QListWidget, QTableView, QPlainTextEdit {{
    background: transparent; alternate-background-color: #192131; border: 0; outline: 0;
    color: {TEXT}; gridline-color: {BORDER};
}}
QTreeWidget::item, QListWidget::item {{ min-height: 26px; border-radius: 3px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: #253247; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: #273F50; color: #D9FFFA; }}
QHeaderView {{ background: {SURFACE}; }}
QHeaderView::section {{
    color: {MUTED}; background: #1B2435; padding: 8px;
    border: 0; border-bottom: 1px solid {BORDER}; font-weight: 650;
}}
QTabWidget::pane {{ border: 0; border-top: 1px solid {BORDER}; background: transparent; }}
QTabBar::tab {{ padding: 8px 12px; color: {MUTED}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom-color: {ACCENT}; font-weight: 700; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QMenuBar {{ background: #141A26; padding: 1px 4px; }}
QMenuBar::item {{ padding: 4px 7px; color: {MUTED}; }}
QMenuBar::item:selected {{ background: #29364A; color: {TEXT}; }}
QMenu {{ background: #1A2333; border: 1px solid {BORDER}; padding: 5px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 3px; }}
QMenu::item:selected {{ background: #2E4159; }}
QMenu::item:disabled {{ color: {SUBTLE}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QToolTip {{ background: #243349; color: #F6FCFF; border: 1px solid #465872; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 1px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 1px; }}
QScrollBar::handle {{ background: #394B64; border-radius: 4px; min-height: 21px; min-width: 21px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; width: 10px; }}
QProgressBar {{ background: #283447; border: 0; border-radius: 3px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
"""
