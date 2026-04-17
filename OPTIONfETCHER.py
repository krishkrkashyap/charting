import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QDateEdit,
    QFrame,
    QGroupBox,
    QSpinBox,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QColorDialog,
    QCheckBox,
    QSlider,
    QToolButton,
    QButtonGroup,
    QScrollArea,
    QSplitter,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QTabWidget,
    QGridLayout,
    QSizePolicy,
    QDoubleSpinBox,
    QFileDialog,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtCore import (
    Qt,
    QDate,
    pyqtSignal,
    QTimer,
    QThread,
    pyqtSlot,
    QObject,
    QSize,
)
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QShortcut, QAction, QFont
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
import json
import os
from datetime import datetime, timedelta

from api_call import ApiCall


# ============================================================
# THEME MANAGER
# ============================================================


class ThemeManager:
    """Manages application themes (light/dark)"""

    THEMES = {
        "dark": {
            "name": "Dark",
            "colors": {
                "background": "#131722",
                "surface": "#1e222d",
                "surface_light": "#2a2e39",
                "text": "#d1d4dc",
                "text_secondary": "#737780",
                "primary": "#2196F3",
                "success": "#26a69a",
                "danger": "#ef5350",
                "warning": "#FF9800",
                "border": "#363c4e",
                "grid": "#1e222d",
            },
        },
        "light": {
            "name": "Light",
            "colors": {
                "background": "#ffffff",
                "surface": "#f5f5f5",
                "surface_light": "#e0e0e0",
                "text": "#1a1a1a",
                "text_secondary": "#666666",
                "primary": "#1976D2",
                "success": "#4CAF50",
                "danger": "#f44336",
                "warning": "#FF9800",
                "border": "#cccccc",
                "grid": "#e0e0e0",
            },
        },
    }

    def __init__(self, theme="dark"):
        self._current_theme = theme
        self._theme_changed_callbacks = []

    def get_theme_colors(self, theme=None):
        """Get colors for a theme"""
        if theme is None:
            theme = self._current_theme
        return self.THEMES.get(theme, self.THEMES["dark"])["colors"]

    def get_chart_colors(self, theme=None):
        """Get chart-specific colors (for JavaScript)"""
        if theme is None:
            theme = self._current_theme

        colors = self.THEMES.get(theme, self.THEMES["dark"])["colors"]

        if theme == "dark":
            return {
                "bg": colors["background"],
                "text": colors["text"],
                "grid": colors["grid"],
                "border": colors["border"],
                "up": colors["success"],
                "down": colors["danger"],
            }
        else:
            return {
                "bg": "#ffffff",
                "text": "#1a1a1a",
                "grid": "#e0e0e0",
                "border": "#cccccc",
                "up": "#4CAF50",
                "down": "#f44336",
            }

    def set_theme(self, theme):
        """Set the current theme"""
        if theme in self.THEMES:
            self._current_theme = theme
            for callback in self._theme_changed_callbacks:
                callback(theme)

    def get_theme(self):
        return self._current_theme

    def toggle_theme(self):
        """Toggle between dark and light themes"""
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self.set_theme(new_theme)
        return new_theme

    def on_theme_changed(self, callback):
        """Register a callback for theme changes"""
        self._theme_changed_callbacks.append(callback)


# ============================================================
# INDICATOR PARAMETER DIALOG
# ============================================================


class IndicatorParameterDialog(QDialog):
    """Dialog for configuring indicator parameters"""

    def __init__(self, ind_type, parent=None):
        super().__init__(parent)
        self.ind_type = ind_type
        self.params = {}
        self.setWindowTitle(f"Configure {ind_type}")
        self.setModal(True)
        self.setFixedSize(380, 350)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel(f"📊 {ind_type} Parameters")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        layout.addWidget(title)

        # Create parameter inputs based on indicator type
        self.param_widgets = {}

        if ind_type in ["SMA", "EMA", "WMA", "DEMA", "TEMA", "VWMA"]:
            # Moving averages - period input
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(2, 500)
            self.period_spin.setValue(20)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin

        elif ind_type == "BB":
            # Bollinger Bands - period and stdDev
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(2, 500)
            self.period_spin.setValue(20)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Std Dev:"))
            self.stddev_spin = QDoubleSpinBox()
            self.stddev_spin.setRange(0.1, 10)
            self.stddev_spin.setValue(2)
            self.stddev_spin.setDecimals(2)
            self.stddev_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.stddev_spin)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin
            self.param_widgets["stdDev"] = self.stddev_spin

        elif ind_type == "KC":
            # Keltner Channel
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(2, 500)
            self.period_spin.setValue(20)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("ATR Period:"))
            self.atr_spin = QSpinBox()
            self.atr_spin.setRange(1, 100)
            self.atr_spin.setValue(10)
            self.atr_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.atr_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Multiplier:"))
            self.mult_spin = QDoubleSpinBox()
            self.mult_spin.setRange(0.1, 10)
            self.mult_spin.setValue(1.5)
            self.mult_spin.setDecimals(2)
            self.mult_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.mult_spin)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin
            self.param_widgets["atrPeriod"] = self.atr_spin
            self.param_widgets["multiplier"] = self.mult_spin

        elif ind_type == "DC":
            # Donchian Channel
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(2, 500)
            self.period_spin.setValue(20)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin

        elif ind_type == "SUPERTREND":
            # Supertrend
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(1, 100)
            self.period_spin.setValue(10)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Multiplier:"))
            self.mult_spin = QDoubleSpinBox()
            self.mult_spin.setRange(0.1, 10)
            self.mult_spin.setValue(3)
            self.mult_spin.setDecimals(1)
            self.mult_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.mult_spin)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin
            self.param_widgets["multiplier"] = self.mult_spin

        elif ind_type == "RSI":
            # RSI
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(2, 100)
            self.period_spin.setValue(14)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin

        elif ind_type == "MACD":
            # MACD
            row = QHBoxLayout()
            row.addWidget(QLabel("Fast Period:"))
            self.fast_spin = QSpinBox()
            self.fast_spin.setRange(2, 100)
            self.fast_spin.setValue(12)
            self.fast_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.fast_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Slow Period:"))
            self.slow_spin = QSpinBox()
            self.slow_spin.setRange(2, 200)
            self.slow_spin.setValue(26)
            self.slow_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.slow_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Signal Period:"))
            self.signal_spin = QSpinBox()
            self.signal_spin.setRange(2, 100)
            self.signal_spin.setValue(9)
            self.signal_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.signal_spin)
            layout.addLayout(row)

            self.param_widgets["fastPeriod"] = self.fast_spin
            self.param_widgets["slowPeriod"] = self.slow_spin
            self.param_widgets["signalPeriod"] = self.signal_spin

        elif ind_type == "STOCH":
            # Stochastic
            row = QHBoxLayout()
            row.addWidget(QLabel("%K Period:"))
            self.k_spin = QSpinBox()
            self.k_spin.setRange(2, 100)
            self.k_spin.setValue(14)
            self.k_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.k_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("%D Period:"))
            self.d_spin = QSpinBox()
            self.d_spin.setRange(2, 100)
            self.d_spin.setValue(3)
            self.d_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.d_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Smooth:"))
            self.smooth_spin = QSpinBox()
            self.smooth_spin.setRange(1, 50)
            self.smooth_spin.setValue(3)
            self.smooth_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.smooth_spin)
            layout.addLayout(row)

            self.param_widgets["kPeriod"] = self.k_spin
            self.param_widgets["dPeriod"] = self.d_spin
            self.param_widgets["smooth"] = self.smooth_spin

        elif ind_type == "ATR":
            # Average True Range
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(1, 100)
            self.period_spin.setValue(14)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin

        elif ind_type == "ADX":
            # Average Directional Index
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(1, 100)
            self.period_spin.setValue(14)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin

        elif ind_type == "OBV":
            # On Balance Volume - no parameters needed
            info = QLabel("OBV uses default settings (no parameters required).")
            info.setStyleSheet("color: #737780;")
            layout.addWidget(info)

        elif ind_type == "AD":
            # Accumulation/Distribution - no parameters needed
            info = QLabel(
                "Accumulation/Distribution uses default settings (no parameters required)."
            )
            info.setStyleSheet("color: #737780;")
            layout.addWidget(info)

        elif ind_type == "MOMENTUM":
            # Momentum
            row = QHBoxLayout()
            row.addWidget(QLabel("Period:"))
            self.period_spin = QSpinBox()
            self.period_spin.setRange(1, 100)
            self.period_spin.setValue(10)
            self.period_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.period_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Source:"))
            self.source_combo = QComboBox()
            self.source_combo.addItems(
                ["Close", "Open", "High", "Low", "HLC3", "OHLC4", "HL2"]
            )
            self.source_combo.setCurrentText("Close")
            self.source_combo.currentTextChanged.connect(self._update_params)
            row.addWidget(self.source_combo)
            layout.addLayout(row)

            self.param_widgets["period"] = self.period_spin
            self.param_widgets["source"] = self.source_combo

        elif ind_type == "STOCHRSI":
            # Stochastic RSI
            row = QHBoxLayout()
            row.addWidget(QLabel("RSI Period:"))
            self.rsi_spin = QSpinBox()
            self.rsi_spin.setRange(2, 100)
            self.rsi_spin.setValue(14)
            self.rsi_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.rsi_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("%K Period:"))
            self.k_spin = QSpinBox()
            self.k_spin.setRange(2, 100)
            self.k_spin.setValue(3)
            self.k_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.k_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("%D Period:"))
            self.d_spin = QSpinBox()
            self.d_spin.setRange(2, 100)
            self.d_spin.setValue(3)
            self.d_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.d_spin)
            layout.addLayout(row)

            self.param_widgets["rsiPeriod"] = self.rsi_spin
            self.param_widgets["kPeriod"] = self.k_spin
            self.param_widgets["dPeriod"] = self.d_spin

        elif ind_type == "ICHIMOKU":
            # Ichimoku Cloud
            row = QHBoxLayout()
            row.addWidget(QLabel("Tenkan Period:"))
            self.tenkan_spin = QSpinBox()
            self.tenkan_spin.setRange(1, 100)
            self.tenkan_spin.setValue(9)
            self.tenkan_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.tenkan_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Kijun Period:"))
            self.kijun_spin = QSpinBox()
            self.kijun_spin.setRange(1, 100)
            self.kijun_spin.setValue(26)
            self.kijun_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.kijun_spin)
            layout.addLayout(row)

            row = QHBoxLayout()
            row.addWidget(QLabel("Senkou Span B:"))
            self.senkou_spin = QSpinBox()
            self.senkou_spin.setRange(1, 100)
            self.senkou_spin.setValue(52)
            self.senkou_spin.valueChanged.connect(self._update_params)
            row.addWidget(self.senkou_spin)
            layout.addLayout(row)

            self.param_widgets["tenkan"] = self.tenkan_spin
            self.param_widgets["kijun"] = self.kijun_spin
            self.param_widgets["senkouSpanB"] = self.senkou_spin

        elif ind_type == "HEIKINASHI":
            # Heikin-Ashi - no parameters needed
            info = QLabel(
                "Heikin-Ashi converts candlestick data.\nNo parameters required."
            )
            info.setStyleSheet("color: #737780;")
            layout.addWidget(info)

        else:
            # Default - no parameters
            info = QLabel("This indicator uses default settings.")
            info.setStyleSheet("color: #737780;")
            layout.addWidget(info)

        # Preview
        self.preview_label = QLabel("Parameters: {}")
        self.preview_label.setStyleSheet(
            "color: #26a69a; font-family: monospace; font-size: 12px;"
        )
        layout.addWidget(self.preview_label)

        layout.addStretch()

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Set dialog style
        self.setStyleSheet("""
            QDialog { background-color: #1e222d; }
            QLabel { color: #d1d4dc; }
            QSpinBox, QDoubleSpinBox {
                background-color: #2a2e39;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 5px;
                color: #d1d4dc;
            }
        """)

        self._update_params()

    def _update_params(self):
        """Update the params dict and preview"""
        self.params = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                self.params[name] = widget.value()
            else:
                self.params[name] = widget.value()

        param_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        self.preview_label.setText(
            f"Parameters: {param_str if param_str else 'Default'}"
        )

    def get_params(self):
        """Return the configured parameters"""
        return self.params


class CustomPythonIndicatorDialog(QDialog):
    """Dialog for writing custom Python indicator scripts"""

    def __init__(self, parent=None, initial_code=None, initial_pane="New Pane"):
        super().__init__(parent)
        self.setWindowTitle("⚡ Custom Python Indicator")
        self.setMinimumSize(750, 600)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header / Help
        help_box = QGroupBox("Development Guide")
        help_box.setStyleSheet(
            "color: #d1d4dc; font-weight: bold; background-color: #1e222d; border: 1px solid #363c4e;"
        )
        help_layout = QVBoxLayout(help_box)

        help_text = QLabel(
            "<b>Available Variables:</b><br/>"
            "• <code>df</code>: Pandas DataFrame with <code>'open', 'high', 'low', 'close', 'volume'</code><br/>"
            "• <code>talib</code>, <code>np</code>, <code>pd</code>: Standard libraries<br/><br/>"
            "<b>Output Requirement:</b><br/>"
            "• Assign result to <code>result</code> variable.<br/>"
            "• <b>Single Line:</b> <code>result = talib.SMA(df['close'], 14)</code><br/>"
            "• <b>Multi-Line:</b> <code>result = {'Upper': u, 'Lower': l}</code> (Dictionary of series)"
        )
        help_text.setStyleSheet("font-weight: normal; color: #737780; font-size: 13px;")
        help_layout.addWidget(help_text)
        layout.addWidget(help_box)

        # Code Editor
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Consolas", 11))

        default_code = initial_code or (
            "# Pandas 'df' is available. Assign output to 'result'\n"
            "import talib\n\n"
            "# Example: 14-period RSI\n"
            "result = talib.RSI(df['close'], timeperiod=14)"
        )
        self.code_editor.setPlainText(default_code)

        self.code_editor.setStyleSheet("""
            QTextEdit {
                background-color: #131722;
                color: #d1d4dc;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.code_editor)

        # Options
        options_layout = QHBoxLayout()

        self.pane_label = QLabel("Plotting Mode:")
        self.pane_label.setStyleSheet("color: #d1d4dc;")
        options_layout.addWidget(self.pane_label)

        self.pane_combo = QComboBox()
        self.pane_combo.addItems(["Overlay (On Main Chart)", "New Pane (Oscillator)"])
        # Set default based on input
        if initial_pane == "Main Chart":
            self.pane_combo.setCurrentIndex(0)
        else:
            self.pane_combo.setCurrentIndex(1)

        self.pane_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2e39;
                color: #d1d4dc;
                border: 1px solid #363c4e;
                padding: 5px;
                min-width: 250px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e222d;
                color: #d1d4dc;
                selection-background-color: #2196F3;
            }
        """)
        options_layout.addWidget(self.pane_combo)
        options_layout.addStretch()

        layout.addLayout(options_layout)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        buttons.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #d1d4dc;
                border: none;
                padding: 8px 30px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #363c4e;
            }
        """)

        layout.addWidget(buttons)
        self.setStyleSheet("background-color: #131722;")

    def get_data(self):
        return {
            "code": self.code_editor.toPlainText(),
            "pane": "Main Chart" if self.pane_combo.currentIndex() == 0 else "New Pane",
        }


# ============================================================
# WORKER CLASSES FOR BACKGROUND TASKS
# ============================================================


class ApiWorker(QObject):
    """Worker to fetch data in background thread"""

    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._params = {}
        self._is_running = False

    def set_params(self, **kwargs):
        self._params = kwargs

    @pyqtSlot()
    def fetch_data(self):
        if self._is_running:
            return
        self._is_running = True

        try:
            data_type = self._params.get("data_type")
            symbol = self._params.get("symbol")
            start = self._params.get("start")
            end = self._params.get("end")
            freq = self._params.get("freq")

            self.progress.emit(f"Fetching {data_type} data for {symbol}...")

            if data_type == "Options":
                strike = self._params.get("strike")
                side = self._params.get("side")
                expiry = self._params.get("expiry")

                self.progress.emit(f"Strike={strike}, Side={side}, Expiry={expiry}")

                # Determine exchange
                exchange = self._params.get("exchange")

                if exchange == "MCX":
                    # MCX options API call
                    df = ApiCall(data_type="Options", exchange="MCX").get_data(
                        symbol=[symbol],
                        start_date=start,
                        end_date=end,
                        start_expiry=expiry,
                        strike_price=[strike],
                        option_type=side,
                        resample_bool=True,
                        resample_freq=freq,
                        label="left",
                        time_filter=False,
                    )
                elif symbol == "SENSEX":
                    api = ApiCall(data_type="Options", exchange="BSE")
                    df = api.get_data(
                        symbol=[symbol],
                        start_date=start,
                        end_date=end,
                        start_expiry=expiry,
                        strike_price=[strike],
                        option_type=side,
                        resample_bool=True,
                        resample_freq=freq,
                        label="left",
                    )
                elif symbol == "SPY":
                    api = ApiCall(data_type="Options", exchange="NYSE")
                    df = api.get_data(
                        symbol=[symbol],
                        start_date=start,
                        end_date=end,
                        start_expiry=expiry,
                        strike_price=[strike],
                        option_type=side,
                        resample_bool=True,
                        resample_freq=freq,
                        label="left",
                    )
                else:
                    api = ApiCall(data_type="Options")
                    df = api.get_data(
                        symbol=[symbol],
                        start_date=start,
                        end_date=end,
                        start_expiry=expiry,
                        strike_price=[strike],
                        option_type=side,
                        resample_bool=True,
                        resample_freq=freq,
                        label="left",
                    )

                expiry_dt = pd.to_datetime(expiry, dayfirst=True)
                ticker = f"{symbol}{expiry_dt.strftime('%y%b%d').upper()}{strike}{side}"

            else:
                self.progress.emit(f"Fetching Spot data...")

                exchange = self._params.get("exchange")

                if exchange == "MCX":
                    # MCX doesn't have spot data in the same way
                    self.error.emit("Spot data not available for MCX exchange")
                    df = None
                elif symbol == "SENSEX":
                    df = ApiCall(data_type="Equity", exchange="BSE").get_data(
                        symbol=[symbol],
                        start_date=start,
                        end_date=end,
                        resample_bool=True,
                        resample_freq=freq,
                        label="left",
                    )
                elif symbol == "SPY":
                    df = ApiCall(data_type="Equity", exchange="NYSE").get_data(
                        symbol=[symbol],
                        start_date=start,
                        end_date=end,
                        resample_bool=True,
                        resample_freq=freq,
                        label="left",
                    )
                else:
                    df = ApiCall(data_type="Equity").get_data(
                        symbol=[symbol],
                        start_date=start,
                        end_date=end,
                        resample_bool=True,
                        resample_freq=freq,
                        label="left",
                    )

                ticker = symbol

            if df is None or df.empty:
                self.error.emit("No data returned from API")
            else:
                self.progress.emit(f"Received {len(df)} rows")
                self.finished.emit(df, ticker)

        except Exception as e:
            import traceback

            self.error.emit(f"API Error: {str(e)}")
            traceback.print_exc()
        finally:
            self._is_running = False


class ExpiryWorker(QObject):
    """Worker to fetch expiry dates"""

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._symbol = None
        self._expiry_type = None
        self._exchange = None
        self._is_running = False

    def set_params(self, symbol, expiry_type, exchange=None):
        self._symbol = symbol
        self._expiry_type = expiry_type
        self._exchange = exchange

    @pyqtSlot()
    def fetch_expiry_dates(self):
        if self._is_running:
            return
        self._is_running = True

        try:
            # For MCX, always use 'O' segment (monthly options)
            if self._exchange == "MCX":
                segment = "O"
            else:
                segment = "F" if self._expiry_type == "Monthly" else "O"

            # Determine exchange for API call
            if self._symbol == "SENSEX":
                api_exchange = "BSE"
            elif self._exchange == "MCX":
                api_exchange = "MCX"
            elif self._symbol == "SPY":
                api_exchange = "NYSE"
            else:
                api_exchange = None  # Default NSE

            if api_exchange:
                expiry_data = ApiCall(
                    data_type="ExpiryDates", exchange=api_exchange
                ).get_data(symbol=self._symbol, segment=segment)
            else:
                expiry_data = ApiCall(data_type="ExpiryDates").get_data(
                    symbol=self._symbol, segment=segment
                )

            if expiry_data is not None and not expiry_data.empty:
                expiry_data = expiry_data[
                    expiry_data["Date"] >= pd.to_datetime("2012-01-01")
                ]
                expiry_list = sorted(
                    pd.to_datetime(expiry_data["Date"], errors="coerce")
                    .dropna()
                    .tolist()
                )
                self.finished.emit(expiry_list)
            else:
                self.finished.emit([])

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([])
        finally:
            self._is_running = False


# ============================================================
# THREAD MANAGER
# ============================================================


class ThreadManager:
    """Manages background threads to prevent premature destruction"""

    def __init__(self):
        self._threads = []

    def create_thread(
        self, worker, started_callback, finished_callbacks=None, error_callback=None
    ):
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(started_callback)

        if finished_callbacks:
            for signal, slot in finished_callbacks:
                signal.connect(slot)

        if error_callback and hasattr(worker, "error"):
            worker.error.connect(error_callback)

        def cleanup():
            thread.quit()
            thread.wait()
            QTimer.singleShot(100, lambda: self._remove_thread(thread))

        if hasattr(worker, "finished"):
            worker.finished.connect(cleanup)
        if hasattr(worker, "error"):
            worker.error.connect(cleanup)

        self._threads.append((thread, worker))
        thread.start()

        return thread, worker

    def _remove_thread(self, thread):
        self._threads = [(t, w) for t, w in self._threads if t != thread]

    def stop_all(self):
        for thread, worker in self._threads:
            if thread.isRunning():
                thread.quit()
                thread.wait(2000)
                if thread.isRunning():
                    thread.terminate()
        self._threads.clear()


# ============================================================
# GO TO DIALOG
# ============================================================


class GoToDialog(QDialog):
    """Dialog to jump to specific date/time"""

    goto_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Go To Date/Time")
        self.setModal(True)
        self.setFixedWidth(350)

        layout = QVBoxLayout(self)

        info = QLabel("Jump to a specific date and time:")
        info.setStyleSheet("color: #737780; margin-bottom: 10px;")
        layout.addWidget(info)

        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Date:"))
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        date_layout.addWidget(self.date_input)
        layout.addLayout(date_layout)

        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time (HH:MM):"))
        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("09:15 (optional)")
        time_layout.addWidget(self.time_input)
        layout.addLayout(time_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setStyleSheet("""
            QDialog { background-color: #1e222d; }
            QLabel { color: #d1d4dc; }
            QLineEdit, QDateEdit {
                background-color: #2a2e39; border: 1px solid #363c4e;
                border-radius: 4px; padding: 6px; color: #d1d4dc;
            }
        """)

    def accept(self):
        date = self.date_input.date().toPyDate()
        time_str = self.time_input.text().strip()

        if time_str:
            try:
                time_parts = time_str.split(":")
                hour = int(time_parts[0])
                minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                dt = datetime(date.year, date.month, date.day, hour, minute)
            except Exception as e:
                logging.error(f"Error parsing time: {e}")
                dt = datetime(date.year, date.month, date.day, 9, 15)
        else:
            dt = datetime(date.year, date.month, date.day, 9, 15)

        timestamp = int(dt.timestamp())
        self.goto_requested.emit(str(timestamp))
        super().accept()


# ============================================================
# ABOUT DIALOG
# ============================================================


class AboutDialog(QDialog):
    """Comprehensive About Dialog with documentation"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📖 About - TradingView Pro Chart")
        self.setModal(True)
        self.setMinimumSize(700, 600)
        self.resize(800, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #1976D2, stop:1 #2196F3);
                padding: 20px;
            }
        """)
        header_layout = QVBoxLayout(header)

        title = QLabel("📈 TradingView Pro Chart")
        title.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)

        version = QLabel("Version 2.0.0 | Professional Edition")
        version.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 12px;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(version)

        layout.addWidget(header)

        # Tab Widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #1e222d;
            }
            QTabBar::tab {
                background-color: #2a2e39;
                color: #737780;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #1e222d;
                color: #2196F3;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #363c4e;
            }
        """)

        # Overview Tab
        tabs.addTab(self._create_overview_tab(), "🏠 Overview")

        # Shortcuts Tab
        tabs.addTab(self._create_shortcuts_tab(), "⌨️ Shortcuts")

        # Features Tab
        tabs.addTab(self._create_features_tab(), "✨ Features")

        # Use Cases Tab
        tabs.addTab(self._create_usecases_tab(), "📊 Use Cases")

        # Developer Tab
        tabs.addTab(self._create_developer_tab(), "👨‍💻 Developer")

        layout.addWidget(tabs)

        # Footer
        footer = QFrame()
        footer.setStyleSheet("background-color: #131722; padding: 10px;")
        footer_layout = QHBoxLayout(footer)

        copyright_label = QLabel(
            "© 2026 All Rights Reserved | Junomoneta Finsol Pvt Ltd"
        )
        copyright_label.setStyleSheet("color: #737780; font-size: 11px;")
        footer_layout.addWidget(copyright_label)

        footer_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 30px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        close_btn.clicked.connect(self.close)
        footer_layout.addWidget(close_btn)

        layout.addWidget(footer)

        self.setStyleSheet("""
            QDialog {
                background-color: #131722;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QLabel {
                color: #d1d4dc;
            }
        """)

    def _create_scroll_content(self, content_widget):
        """Wrap content in a scroll area"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _create_overview_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        intro = QLabel("""
<h2 style="color: #2196F3;">Welcome to TradingView Pro Chart</h2>
<p style="font-size: 14px; line-height: 1.8;">
TradingView Pro Chart is a powerful, professional-grade charting application designed 
specifically for financial market analysis. Built with cutting-edge technology, it provides 
traders and analysts with the tools they need to make informed decisions.
</p>

<h3 style="color: #26a69a; margin-top: 20px;">🎯 Key Highlights</h3>
<ul style="font-size: 13px; line-height: 2;">
<li><b>Options Analysis:</b> Comprehensive options chain data with strike selection</li>
<li><b>Drawing Tools:</b> Professional drawing tools including trend lines, horizontal/vertical lines</li>
<li><b>Technical Indicators:</b> Built-in indicators like SMA, EMA, Bollinger Bands, VWAP</li>
<li><b>Measure Tool:</b> Calculate price differences, percentage changes, and bar counts</li>
<li><b>Customizable:</b> Adjust colors, scales, and display preferences</li>
</ul>

<h3 style="color: #26a69a; margin-top: 20px;">📈 Supported Instruments</h3>
<table style="font-size: 13px; margin-top: 10px;">
<tr><td style="padding: 8px; color: #2196F3;"><b>Spot/Equity:</b></td><td style="padding: 8px;">NIFTY 50, BANK NIFTY, SENSEX</td></tr>
<tr><td style="padding: 8px; color: #2196F3;"><b>Options:</b></td><td style="padding: 8px;">NIFTY, BANKNIFTY, SENSEX, COPPER, CRUDEOIL, CRUDEOILM, GOLD, GOLDM, NATGASMINI, NATURALGAS, NICKEL, SILVER, SILVERM, ZINC, ZINCMINI</td></tr>
<tr><td style="padding: 8px; color: #2196F3;"><b>Expiry Types:</b></td><td style="padding: 8px;">Weekly & Monthly</td></tr>
<tr><td style="padding: 8px; color: #2196F3;"><b>Timeframes:</b></td><td style="padding: 8px;">1min, 5min, 15min, 30min, 1h, 1D</td></tr>
</table>
        """)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        layout.addStretch()

        return self._create_scroll_content(widget)

    def _create_shortcuts_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        title = QLabel("<h2 style='color: #2196F3;'>⌨️ Keyboard Shortcuts</h2>")
        title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(title)

        shortcuts_data = [
            (
                "Drawing Tools",
                [
                    ("H", "Horizontal Line", "Draw a horizontal price level line"),
                    ("V", "Vertical Line", "Draw a vertical time marker"),
                    ("T", "Trend Line", "Draw a trend line (2 clicks)"),
                    ("M", "Measure Tool", "Measure price & time difference"),
                    ("Delete", "Delete Selected", "Remove selected drawing"),
                    ("Escape", "Cancel Drawing", "Cancel current drawing operation"),
                ],
            ),
            (
                "Chart Navigation",
                [
                    ("F", "Fit Content", "Auto-fit all data in view"),
                    ("Ctrl+G", "Go To Date", "Jump to specific date/time"),
                    ("Ctrl+S", "Switch Chart", "Switch to the next chart"),
                ],
            ),
            (
                "Indicators & Drawings",
                [
                    ("Ctrl+D", "Clear Drawings", "Remove all drawings from chart"),
                    ("Ctrl+I", "Clear Indicators", "Remove all indicators from chart"),
                ],
            ),
        ]

        for category, shortcuts in shortcuts_data:
            cat_label = QLabel(
                f"<h3 style='color: #26a69a; margin-top: 15px;'>{category}</h3>"
            )
            cat_label.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(cat_label)

            grid = QGridLayout()
            grid.setSpacing(10)

            for i, (key, action, desc) in enumerate(shortcuts):
                key_label = QLabel(
                    f"<span style='background-color: #2a2e39; padding: 5px 12px; border-radius: 4px; font-family: monospace; font-weight: bold; color: #FF9800;'>{key}</span>"
                )
                key_label.setTextFormat(Qt.TextFormat.RichText)
                grid.addWidget(key_label, i, 0)

                action_label = QLabel(f"<b style='color: #d1d4dc;'>{action}</b>")
                action_label.setTextFormat(Qt.TextFormat.RichText)
                grid.addWidget(action_label, i, 1)

                desc_label = QLabel(f"<span style='color: #737780;'>{desc}</span>")
                desc_label.setTextFormat(Qt.TextFormat.RichText)
                grid.addWidget(desc_label, i, 2)

            layout.addLayout(grid)

        # Tips section
        tips = QLabel("""
<h3 style="color: #26a69a; margin-top: 25px;">💡 Pro Tips</h3>
<ul style="font-size: 13px; line-height: 2; color: #d1d4dc;">
<li>Enable <b>Magnet Mode</b> to snap drawings to candle close prices</li>
<li>Click on a drawing to select it, then press <b>Delete</b> to remove</li>
<li>Use <b>Scroll Wheel</b> to zoom in/out on the chart</li>
<li>Click and drag to pan the chart left/right</li>
<li>Double-click on price scale to auto-fit vertical range</li>
</ul>
        """)
        tips.setTextFormat(Qt.TextFormat.RichText)
        tips.setWordWrap(True)
        layout.addWidget(tips)

        layout.addStretch()

        return self._create_scroll_content(widget)

    def _create_features_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        content = QLabel("""
<h2 style="color: #2196F3;">✨ Features & Capabilities</h2>

<h3 style="color: #26a69a; margin-top: 20px;">📊 Data Management</h3>
<ul style="font-size: 13px; line-height: 2;">
<li><b>Multiple Data Types:</b> Switch between Spot and Options data seamlessly</li>
<li><b>Flexible Date Range:</b> Select custom start and end dates for analysis</li>
<li><b>Auto Expiry Loading:</b> Automatically loads available expiry dates for options</li>
<li><b>Multiple Timeframes:</b> From 1-minute to daily candles</li>
</ul>

<h3 style="color: #26a69a; margin-top: 20px;">🎨 Drawing Tools</h3>
<table style="font-size: 13px; width: 100%;">
<tr>
<td style="padding: 10px; vertical-align: top; width: 50%;">
<p><b style="color: #2196F3;">━ Horizontal Line</b></p>
<p style="color: #737780;">Mark important price levels, support/resistance zones</p>
</td>
<td style="padding: 10px; vertical-align: top; width: 50%;">
<p><b style="color: #FF9800;">┃ Vertical Line</b></p>
<p style="color: #737780;">Mark significant time events, market opens</p>
</td>
</tr>
<tr>
<td style="padding: 10px; vertical-align: top;">
<p><b style="color: #9C27B0;">📈 Trend Line</b></p>
<p style="color: #737780;">Connect two points to identify trend direction</p>
</td>
<td style="padding: 10px; vertical-align: top;">
<p><b style="color: #4CAF50;">📏 Measure Tool</b></p>
<p style="color: #737780;">Calculate price change, percentage, and bar count</p>
</td>
</tr>
</table>

<h3 style="color: #26a69a; margin-top: 20px;">📈 Technical Indicators</h3>
<table style="font-size: 13px; width: 100%;">
<tr>
<td style="padding: 10px;"><b style="color: #2196F3;">SMA</b></td>
<td style="padding: 10px;">Simple Moving Average - Customizable period</td>
</tr>
<tr>
<td style="padding: 10px;"><b style="color: #FF9800;">EMA</b></td>
<td style="padding: 10px;">Exponential Moving Average - Faster response to price</td>
</tr>
<tr>
<td style="padding: 10px;"><b style="color: #9C27B0;">Bollinger Bands</b></td>
<td style="padding: 10px;">Volatility bands with customizable deviation</td>
</tr>
<tr>
<td style="padding: 10px;"><b style="color: #4CAF50;">VWAP</b></td>
<td style="padding: 10px;">Volume Weighted Average Price</td>
</tr>
</table>

<h3 style="color: #26a69a; margin-top: 20px;">⚙️ Chart Settings</h3>
<ul style="font-size: 13px; line-height: 2;">
<li><b>Magnet Mode:</b> Snap crosshair and drawings to candle prices</li>
<li><b>Grid Toggle:</b> Show/hide chart grid lines</li>
<li><b>Scale Mode:</b> Switch between Normal and Logarithmic scale</li>
<li><b>Fit Content:</b> Auto-adjust view to show all data</li>
<li><b>Go To Date:</b> Jump to specific date and time</li>
</ul>
        """)
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setWordWrap(True)
        layout.addWidget(content)

        layout.addStretch()

        return self._create_scroll_content(widget)

    def _create_usecases_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        content = QLabel("""
<h2 style="color: #2196F3;">📊 Use Cases & Workflows</h2>

<h3 style="color: #26a69a; margin-top: 20px;">1️⃣ Options Trading Analysis</h3>
<div style="background-color: #1e222d; padding: 15px; border-radius: 8px; margin: 10px 0;">
<p style="font-size: 13px; line-height: 1.8;">
<b>Workflow:</b><br>
1. Select "Options" as data type<br>
2. Choose symbol (NIFTY/BANKNIFTY/SENSEX)<br>
3. Select expiry type (Weekly/Monthly)<br>
4. Set date range for analysis<br>
5. Enter strike price and select CE/PE<br>
6. Click "Fetch & Plot Data"<br>
7. Add indicators like SMA/EMA for trend analysis<br>
8. Use horizontal lines to mark key price levels
</p>
</div>

<h3 style="color: #26a69a; margin-top: 20px;">2️⃣ Index Trend Analysis</h3>
<div style="background-color: #1e222d; padding: 15px; border-radius: 8px; margin: 10px 0;">
<p style="font-size: 13px; line-height: 1.8;">
<b>Workflow:</b><br>
1. Select "Spot" as data type<br>
2. Choose index (NIFTY_50/NIFTY_BANK/SENSEX)<br>
3. Set appropriate timeframe (15min for intraday, 1D for swing)<br>
4. Add multiple EMAs (9, 21, 50) for trend confirmation<br>
5. Use trend lines to identify support/resistance<br>
6. Apply Bollinger Bands for volatility analysis
</p>
</div>

<h3 style="color: #26a69a; margin-top: 20px;">3️⃣ Support & Resistance Identification</h3>
<div style="background-color: #1e222d; padding: 15px; border-radius: 8px; margin: 10px 0;">
<p style="font-size: 13px; line-height: 1.8;">
<b>Workflow:</b><br>
1. Load historical data with wider date range<br>
2. Use "Fit Content" (F) to see full picture<br>
3. Enable Magnet mode for precise level marking<br>
4. Draw horizontal lines at key swing highs/lows<br>
5. Use Measure Tool to calculate risk:reward ratios
</p>
</div>

<h3 style="color: #26a69a; margin-top: 20px;">4️⃣ Price Action Measurement</h3>
<div style="background-color: #1e222d; padding: 15px; border-radius: 8px; margin: 10px 0;">
<p style="font-size: 13px; line-height: 1.8;">
<b>Workflow:</b><br>
1. Identify a price move you want to measure<br>
2. Press 'M' or click Measure Tool<br>
3. Click on starting point (e.g., swing low)<br>
4. Move to ending point (e.g., swing high)<br>
5. View points change, percentage move, and bars elapsed<br>
6. Click to finish measurement
</p>
</div>

<h3 style="color: #26a69a; margin-top: 20px;">5️⃣ Multi-Timeframe Analysis</h3>
<div style="background-color: #1e222d; padding: 15px; border-radius: 8px; margin: 10px 0;">
<p style="font-size: 13px; line-height: 1.8;">
<b>Workflow:</b><br>
1. Start with higher timeframe (1D) for trend direction<br>
2. Mark key levels using horizontal lines<br>
3. Switch to lower timeframe (15min/5min)<br>
4. Observe price behavior at marked levels<br>
5. Use VWAP for intraday fair value reference
</p>
</div>
        """)
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setWordWrap(True)
        layout.addWidget(content)

        layout.addStretch()

        return self._create_scroll_content(widget)

    def _create_developer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Developer Card
        dev_card = QFrame()
        dev_card.setStyleSheet("""
            QFrame {
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        card_layout = QVBoxLayout(dev_card)

        name = QLabel("Mr. Krish Kumar")
        name.setStyleSheet("color: #2196F3; font-size: 28px; font-weight: bold;")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(name)

        role = QLabel("Research Analyst")
        role.setStyleSheet("color: #26a69a; font-size: 16px; font-weight: bold;")
        role.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(role)

        emp_id = QLabel("Employee ID: JM1710")
        emp_id.setStyleSheet("color: #737780; font-size: 13px;")
        emp_id.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(emp_id)

        layout.addWidget(dev_card)

        # About Developer
        about_dev = QLabel("""
<h3 style="color: #2196F3; margin-top: 20px;">About the Developer</h3>
<p style="font-size: 13px; line-height: 1.8; color: #d1d4dc;">
Krish is a dedicated Research Analyst specializing in quantitative analysis and 
financial market research. With a strong background in data analysis and software 
development, Krish combines domain expertise with technical skills to create 
powerful tools for market analysis.
</p>

<h3 style="color: #2196F3; margin-top: 20px;">About the Product</h3>
<p style="font-size: 13px; line-height: 1.8; color: #d1d4dc;">
This application was created by Krish Kumar, a Research Analyst at Junomoneta Finsol Private Limited. 
The purpose of this application is to overcome the issues faced by backtesters while backtesting 
their models—issues that existing platforms do not address.

The application comes with built-in SPOT and options data for NIFTY, BANKNIFTY, and SENSEX, 
allowing users to test their strategies on historical data across different indices. 
Not only can you backtest your strategies, but you can also apply built-in indicators 
(EMA, Bollinger Bands, etc.) without limitations, unlike other platforms such as TradingView.

We hope you like the product, and we are always open to listening to your feedback and 
implementing it to improve the product and benefit the company.</p>

<h3 style="color: #2196F3; margin-top: 20px;">Job Responsibilities</h3>
<ul style="font-size: 13px; line-height: 2; color: #d1d4dc;">
<li>Conduct quantitative research on financial instruments</li>
<li>Develop analytical tools and charting applications</li>
<li>Analyze market data and identify trading patterns</li>
<li>Create technical indicators and backtesting frameworks</li>
<li>Generate insights for trading strategies</li>
</ul>

<h3 style="color: #2196F3; margin-top: 20px;">Technical Stack</h3>
<p style="font-size: 13px; color: #737780;">
<span style="background-color: #2a2e39; padding: 4px 10px; border-radius: 4px; margin-right: 5px;">Python</span>
<span style="background-color: #2a2e39; padding: 4px 10px; border-radius: 4px; margin-right: 5px;">PyQt6</span>
<span style="background-color: #2a2e39; padding: 4px 10px; border-radius: 4px; margin-right: 5px;">JavaScript</span>
<span style="background-color: #2a2e39; padding: 4px 10px; border-radius: 4px; margin-right: 5px;">Lightweight Charts</span>
<span style="background-color: #2a2e39; padding: 4px 10px; border-radius: 4px; margin-right: 5px;">Pandas</span>
<span style="background-color: #2a2e39; padding: 4px 10px; border-radius: 4px;">NumPy</span>
</p>

<h3 style="color: #2196F3; margin-top: 25px;">Project Information</h3>
<table style="font-size: 13px; margin-top: 10px;">
<tr><td style="padding: 8px; color: #737780;">Application:</td><td style="padding: 8px; color: #d1d4dc;">TradingView Pro Chart</td></tr>
<tr><td style="padding: 8px; color: #737780;">Version:</td><td style="padding: 8px; color: #d1d4dc;">2.0.0</td></tr>
<tr><td style="padding: 8px; color: #737780;">Framework:</td><td style="padding: 8px; color: #d1d4dc;">PyQt6 + Lightweight Charts</td></tr>
<tr><td style="padding: 8px; color: #737780;">Last Updated:</td><td style="padding: 8px; color: #d1d4dc;">12-FEB-2026</td></tr>
</table>
        """)
        about_dev.setTextFormat(Qt.TextFormat.RichText)
        about_dev.setWordWrap(True)
        layout.addWidget(about_dev)

        layout.addStretch()

        return self._create_scroll_content(widget)


# ============================================================
# CUSTOM WEB PAGE
# ============================================================


class WebEnginePage(QWebEnginePage):
    """Custom page to capture JavaScript console messages"""

    js_message = pyqtSignal(str)

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        try:
            level_map = {
                QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel: "Info",
                QWebEnginePage.JavaScriptConsoleMessageLevel.WarningMessageLevel: "Warning",
                QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel: "Error",
            }
            level_str = level_map.get(level, "Log")
        except Exception as e:
            logging.error(f"Error parsing console message level: {e}")
            level_str = "Log"

        msg = f"[JS {level_str}] {message}"
        logging.info(msg)
        self.js_message.emit(msg)


# ============================================================
# NEXTSTEP DIALOG - Custom Data Upload
# ============================================================


class NextSTEPDialog(QDialog):
    """Dialog for uploading and plotting custom Excel/CSV data"""

    data_loaded = pyqtSignal(pd.DataFrame, str)  # Emits DataFrame and ticker

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📁 NextSTEP - Upload Custom Data")
        self.setModal(True)
        self.setMinimumSize(500, 700)
        self.resize(550, 700)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("📁 Upload Custom Data (Excel/CSV)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #9C27B0;")
        layout.addWidget(title)

        # File selection
        file_row = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select Excel or CSV file...")
        self.file_path.setReadOnly(True)
        file_row.addWidget(self.file_path)

        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                border: 1px solid #7B1FA2;
                border-radius: 4px;
                padding: 8px 15px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #AB47BC;
            }
        """)
        browse_btn.clicked.connect(self.browse_file)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Column mapping
        col_group = QGroupBox("Column Mapping")
        col_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #363c4e;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                color: #d1d4dc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        col_layout = QGridLayout(col_group)
        col_layout.setSpacing(10)

        # Column name selectors (QComboBox)
        col_layout.addWidget(QLabel("Datetime Column:"), 0, 0)
        self.datetime_col = QComboBox()
        self.datetime_col.setPlaceholderText("Select column")
        col_layout.addWidget(self.datetime_col, 0, 1)

        col_layout.addWidget(QLabel("Open Column:"), 1, 0)
        self.open_col = QComboBox()
        self.open_col.setPlaceholderText("Select column")
        col_layout.addWidget(self.open_col, 1, 1)

        col_layout.addWidget(QLabel("High Column:"), 2, 0)
        self.high_col = QComboBox()
        self.high_col.setPlaceholderText("Select column")
        col_layout.addWidget(self.high_col, 2, 1)

        col_layout.addWidget(QLabel("Low Column:"), 3, 0)
        self.low_col = QComboBox()
        self.low_col.setPlaceholderText("Select column")
        col_layout.addWidget(self.low_col, 3, 1)

        col_layout.addWidget(QLabel("Close Column:"), 4, 0)
        self.close_col = QComboBox()
        self.close_col.setPlaceholderText("Select column")
        col_layout.addWidget(self.close_col, 4, 1)

        layout.addWidget(col_group)

        # Ticker name
        ticker_row = QHBoxLayout()
        ticker_row.addWidget(QLabel("Ticker Name:"))
        self.ticker_name = QLineEdit()
        self.ticker_name.setPlaceholderText("e.g., MYDATA, CUSTOM1")
        ticker_row.addWidget(self.ticker_name)
        layout.addLayout(ticker_row)

        # Resample options
        resample_group = QGroupBox("Resample Data (Optional)")
        resample_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #363c4e;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                color: #d1d4dc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        resample_layout = QHBoxLayout(resample_group)
        resample_layout.setSpacing(10)

        resample_layout.addWidget(QLabel("Target Timeframe:"))
        self.resample_freq = QComboBox()
        self.resample_freq.addItems(
            [
                "No Resampling",
                "1min",
                "2min",
                "3min",
                "5min",
                "10min",
                "15min",
                "30min",
                "45min",
                "60min",
                "1h",
                "2h",
                "3h",
                "4h",
                "1D",
                "1W",
                "1M",
            ]
        )
        self.resample_freq.setCurrentIndex(0)
        self.resample_freq.setToolTip("Select target timeframe for resampling")
        resample_layout.addWidget(self.resample_freq)

        self.resample_btn = QPushButton("Apply Resample")
        self.resample_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                border: 1px solid #F57C00;
                border-radius: 4px;
                padding: 8px 15px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FFB74D;
            }
            QPushButton:disabled {
                background-color: #424242;
                border-color: #616161;
            }
        """)
        self.resample_btn.setToolTip("Resample data to selected timeframe")
        self.resample_btn.clicked.connect(self.apply_resample)
        self.resample_btn.setEnabled(False)
        resample_layout.addWidget(self.resample_btn)

        layout.addWidget(resample_group)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #737780; font-style: italic;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Style the OK button
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("📊 Plot Data")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #26a69a;
                border: 1px solid #1e8a82;
                border-radius: 4px;
                padding: 10px 20px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)
        layout.addWidget(buttons)

        # Set dialog style
        self.setStyleSheet("""
            QDialog { background-color: #1e222d; }
            QLabel { color: #d1d4dc; }
            QLineEdit {
                background-color: #2a2e39;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 8px;
                color: #d1d4dc;
            }
            QGroupBox { color: #d1d4dc; }
        """)

        # Store original data
        self._df = None

    def browse_file(self):
        """Open file dialog to select Excel/CSV file"""

        file_filter = "Excel/CSV Files (*.xlsx *.xls *.csv);;Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", file_filter
        )

        if file_path:
            self.file_path.setText(file_path)
            self.load_file_preview(file_path)

    def load_file_preview(self, file_path):
        """Load file and show preview of columns"""
        try:
            # Load file based on extension
            if file_path.endswith(".csv"):
                self._df = pd.read_csv(file_path)
            else:
                self._df = pd.read_excel(file_path)

            # Get column names
            cols = self._df.columns.tolist()

            # Populate combo boxes with column names
            self._populate_column_combos(cols)

            # Auto-detect and select columns
            self._auto_detect_columns(cols)

            # Enable resample button
            self.resample_btn.setEnabled(True)

            # Show preview
            preview = f"Loaded: {len(self._df)} rows, {len(cols)} columns"
            self.status_label.setText(preview)
            self.status_label.setStyleSheet("color: #26a69a;")

        except Exception as e:
            self.status_label.setText(f"Error loading file: {str(e)}")
            self.status_label.setStyleSheet("color: #ef5350;")
            self._df = None
            self.resample_btn.setEnabled(False)

    def _populate_column_combos(self, cols):
        """Populate all column combo boxes with available columns"""
        combo_boxes = [
            self.datetime_col,
            self.open_col,
            self.high_col,
            self.low_col,
            self.close_col,
        ]

        for combo in combo_boxes:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")  # Empty option
            combo.addItems(cols)
            combo.blockSignals(False)

    def _auto_detect_columns(self, cols):
        """Auto-detect and select column names based on common patterns"""
        col_lower = {c.lower(): c for c in cols}

        # Datetime columns
        datetime_patterns = ["datetime", "date", "time", "timestamp", "dt"]
        self._select_combo_by_text(
            self.datetime_col, datetime_patterns, cols, col_lower
        )

        # OHLC columns - find best match
        ohlc_map = {
            "open": ["open", "open_price"],
            "high": ["high", "high_price"],
            "low": ["low", "low_price"],
            "close": ["close", "close_price", "c"],
        }

        combo_map = {
            "open": self.open_col,
            "high": self.high_col,
            "low": self.low_col,
            "close": self.close_col,
        }

        for field, patterns in ohlc_map.items():
            self._select_combo_by_text(combo_map[field], patterns, cols, col_lower)

    def _select_combo_by_text(self, combo, patterns, cols, col_lower):
        """Helper to select combo box item by matching pattern"""
        for pattern in patterns:
            if pattern in col_lower:
                combo.setCurrentText(col_lower[pattern])
                return
        # If no match found, try case-insensitive search
        for pattern in patterns:
            for col in cols:
                if col.lower() == pattern:
                    combo.setCurrentText(col)
                    return

    def apply_resample(self):
        """Resample data to selected timeframe"""
        if self._df is None:
            return

        try:
            # Get resample frequency
            freq = self.resample_freq.currentText()

            if freq == "No Resampling":
                self.status_label.setText(
                    "Resampling cancelled (No Resampling selected)"
                )
                self.status_label.setStyleSheet("color: #737780;")
                return

            # Check if already resampled (columns are already standard OHLC)
            if all(
                col in self._df.columns
                for col in ["datetime", "open", "high", "low", "close"]
            ):
                df = self._df.copy()
            else:
                # Get column names from combo boxes
                datetime_col = self.datetime_col.currentText().strip()
                open_col = self.open_col.currentText().strip()
                high_col = self.high_col.currentText().strip()
                low_col = self.low_col.currentText().strip()
                close_col = self.close_col.currentText().strip()

                # Validate columns exist
                for col in [datetime_col, open_col, high_col, low_col, close_col]:
                    if col and col not in self._df.columns:
                        self.status_label.setText(f"Column not found: {col}")
                        self.status_label.setStyleSheet("color: #ef5350;")
                        return

                # Process data - rename columns to standard names
                df = self._df.copy()

                if datetime_col:
                    df = df.rename(columns={datetime_col: "datetime"})
                if open_col:
                    df = df.rename(columns={open_col: "open"})
                if high_col:
                    df = df.rename(columns={high_col: "high"})
                if low_col:
                    df = df.rename(columns={low_col: "low"})
                if close_col:
                    df = df.rename(columns={close_col: "close"})

            # Ensure datetime is converted
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime"])

            # Ensure OHLC are numeric
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop rows with NaN in OHLC
            df = df.dropna(subset=["open", "high", "low", "close"])

            if df.empty:
                self.status_label.setText("No valid data to resample")
                self.status_label.setStyleSheet("color: #ef5350;")
                return

            # Set datetime as index
            df = df.set_index("datetime").sort_index()

            # Convert pandas freq to offset alias
            freq_map = {
                "1min": "1min",
                "2min": "2min",
                "3min": "3min",
                "5min": "5min",
                "10min": "10min",
                "15min": "15min",
                "30min": "30min",
                "45min": "45min",
                "60min": "60min",
                "1h": "1H",
                "2h": "2H",
                "3h": "3H",
                "4h": "4H",
                "1D": "1D",
                "1W": "1W",
                "1M": "1M",
            }

            pandas_freq = freq_map.get(freq, "5min")

            # Resample OHLC data
            df_resampled = (
                df[["open", "high", "low", "close"]]
                .resample(pandas_freq)
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            )

            # Drop NaN rows
            df_resampled = df_resampled.dropna()

            # Reset index to make datetime a column again
            df_resampled = df_resampled.reset_index()

            # Update the stored dataframe
            self._df = df_resampled

            # Update preview
            preview = f"Resampled: {len(df_resampled)} rows ({freq})"
            self.status_label.setText(preview)
            self.status_label.setStyleSheet("color: #FF9800;")

        except Exception as e:
            import traceback

            self.status_label.setText(f"Resample error: {str(e)}")
            self.status_label.setStyleSheet("color: #ef5350;")

    def accept(self):
        """Validate and emit data"""
        if self._df is None:
            self.status_label.setText("Please select a file first")
            self.status_label.setStyleSheet("color: #ef5350;")
            return

        # Get column names from combo boxes
        datetime_col = self.datetime_col.currentText().strip()
        open_col = self.open_col.currentText().strip()
        high_col = self.high_col.currentText().strip()
        low_col = self.low_col.currentText().strip()
        close_col = self.close_col.currentText().strip()
        ticker = self.ticker_name.text().strip()

        # Check if resample was applied (columns are already standard OHLC)
        resample_applied = all(
            col in self._df.columns
            for col in ["datetime", "open", "high", "low", "close"]
        )

        if resample_applied:
            # Data has already been processed with resample
            df_processed = self._df.copy()
        else:
            # Get column names from combo boxes
            datetime_col = self.datetime_col.currentText().strip()
            open_col = self.open_col.currentText().strip()
            high_col = self.high_col.currentText().strip()
            low_col = self.low_col.currentText().strip()
            close_col = self.close_col.currentText().strip()
            ticker = self.ticker_name.text().strip()

            # Validate all required columns are selected
            if not datetime_col:
                self.status_label.setText("Please select Datetime column")
                self.status_label.setStyleSheet("color: #ef5350;")
                return

            if not open_col:
                self.status_label.setText("Please select Open column")
                self.status_label.setStyleSheet("color: #ef5350;")
                return

            if not high_col:
                self.status_label.setText("Please select High column")
                self.status_label.setStyleSheet("color: #ef5350;")
                return

            if not low_col:
                self.status_label.setText("Please select Low column")
                self.status_label.setStyleSheet("color: #ef5350;")
                return

            if not close_col:
                self.status_label.setText("Please select Close column")
                self.status_label.setStyleSheet("color: #ef5350;")
                return

            if not ticker:
                self.status_label.setText("Please enter a ticker name")
                self.status_label.setStyleSheet("color: #ef5350;")
                return

            # Validate columns exist
            for col in [datetime_col, open_col, high_col, low_col, close_col]:
                if col not in self._df.columns:
                    self.status_label.setText(f"Column not found: {col}")
                    self.status_label.setStyleSheet("color: #ef5350;")
                    return

            # Process data - rename columns to standard OHLC format
            df_processed = self._df.copy()

            # Rename columns to standard OHLC format
            df_processed = df_processed.rename(
                columns={
                    datetime_col: "datetime",
                    open_col: "open",
                    high_col: "high",
                    low_col: "low",
                    close_col: "close",
                }
            )

        if not ticker:
            self.status_label.setText("Please enter a ticker name")
            self.status_label.setStyleSheet("color: #ef5350;")
            return

        # Ensure datetime is converted
        if "datetime" in df_processed.columns:
            df_processed["datetime"] = pd.to_datetime(
                df_processed["datetime"], errors="coerce"
            )
            df_processed = df_processed.dropna(subset=["datetime"])

        # Ensure OHLC are numeric
        for col in ["open", "high", "low", "close"]:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

        # Drop rows with NaN in OHLC
        df_processed = df_processed.dropna(subset=["open", "high", "low", "close"])

        if df_processed.empty:
            self.status_label.setText("No valid data after processing")
            self.status_label.setStyleSheet("color: #ef5350;")
            return

        # Sort by datetime
        df_processed = df_processed.sort_values("datetime")

        logging.debug(
            f"NextSTEPDialog DEBUG: Emitting data_loaded with {len(df_processed)} rows, ticker={ticker}"
        )
        logging.debug(f"NextSTEPDialog DEBUG: Columns={list(df_processed.columns)}")

        # Emit data
        self.data_loaded.emit(df_processed, ticker)
        super().accept()


# ============================================================
# TRADING VIEW CHART WIDGET
# ============================================================


class TradingViewChart(QWebEngineView):
    """Enhanced TradingView-style chart with working drawing tools and theme support"""

    chart_ready = pyqtSignal()
    chart_error = pyqtSignal(str)
    js_log = pyqtSignal(str)
    data_loaded = pyqtSignal(int)
    drawing_mode_changed = pyqtSignal(str)

    def __init__(self, theme_manager=None):
        super().__init__()

        self._theme_manager = theme_manager or ThemeManager()

        # Set focus policy to accept focus on mouse click (needed for zoom/keyboard)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._page = WebEnginePage(self)
        self._page.js_message.connect(self.js_log.emit)
        self.setPage(self._page)

        self._is_ready = False
        self._pending_data = None
        self._ready_timer = None

        self.loadFinished.connect(self._on_load_finished)
        self.init_chart()

    def _on_load_finished(self, ok):
        if ok:
            logging.info("HTML loaded, initializing chart...")
            self._start_ready_check()
        else:
            self.chart_error.emit("Failed to load HTML")

    def _start_ready_check(self):
        self._ready_check_count = 0
        self._ready_timer = QTimer(self)
        self._ready_timer.timeout.connect(self._check_ready)
        self._ready_timer.start(200)

    def _check_ready(self):
        self._ready_check_count += 1

        if self._ready_check_count > 50:
            self._ready_timer.stop()
            self.chart_error.emit("Chart initialization timeout")
            return

        self.page().runJavaScript(
            "typeof window.chartReady !== 'undefined' && window.chartReady === true",
            self._on_ready_result,
        )

    def _on_ready_result(self, is_ready):
        if is_ready:
            if self._ready_timer:
                self._ready_timer.stop()

            logging.info("Chart is ready!")
            self._is_ready = True
            self.chart_ready.emit()

            if self._pending_data:
                data, ticker = self._pending_data
                self._pending_data = None
                self._set_data_now(data, ticker)

    def is_ready(self):
        return self._is_ready

    def set_data(self, df, ticker):
        try:
            logging.debug(
                f"set_data called: ticker={ticker}, _is_ready={self._is_ready}"
            )
            data = self._prepare_data(df)
            logging.debug(f"Prepared {len(data)} candles for '{ticker}'")

            if not data:
                self.chart_error.emit("No data after preparation")
                return

            if self._is_ready:
                logging.info(f"Chart is ready, setting data now...")
                self._set_data_now(data, ticker)
            else:
                logging.info("Chart not ready, queuing data...")
                self._pending_data = (data, ticker)

        except Exception as e:
            import traceback

            self.chart_error.emit(f"Data preparation error: {e}")
            traceback.print_exc()

    def _set_data_now(self, data, ticker):
        js_data = json.dumps(data)
        ticker_escaped = ticker.replace("'", "\\'").replace('"', '\\"')
        js_code = f"window.setChartData({js_data}, '{ticker_escaped}')"

        def on_result(success):
            if success:
                logging.info(f"Chart data set: {len(data)} candles")
                self.data_loaded.emit(len(data))
            else:
                logging.error("Failed to set chart data")
                self.chart_error.emit("Failed to set chart data")

        self.page().runJavaScript(js_code, on_result)

    def _prepare_data(self, df):
        logging.debug(
            f"_prepare_data: input df shape={df.shape}, columns={list(df.columns)}"
        )
        d = df.copy()

        time_col = None
        for col in [
            "Datetime",
            "datetime",
            "Date",
            "date",
            "Time",
            "time",
            "timestamp",
        ]:
            if col in d.columns:
                time_col = col
                break

        logging.debug(f"_prepare_data: time_col={time_col}")

        if time_col:
            d["time"] = pd.to_datetime(d[time_col])
        elif isinstance(d.index, pd.DatetimeIndex):
            d["time"] = d.index
        else:
            d["time"] = pd.to_datetime(d.index)

        col_mapping = {}
        for col in d.columns:
            lower = col.lower()
            if lower in ["open", "high", "low", "close", "volume"]:
                col_mapping[col] = lower
        d.rename(columns=col_mapping, inplace=True)

        logging.debug(f"_prepare_data: columns after mapping={list(d.columns)}")

        missing = [c for c in ["open", "high", "low", "close"] if c not in d.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if "volume" not in d.columns:
            d["volume"] = 0

        for c in ["open", "high", "low", "close", "volume"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        d["time"] = (pd.to_datetime(d["time"]).astype("int64") // 10**9).astype(int)

        d = d[["time", "open", "high", "low", "close", "volume"]].dropna()
        d = d.sort_values("time").drop_duplicates(subset=["time"])

        # Add capitalized aliases for user script compatibility
        d["Open"] = d["open"]
        d["High"] = d["high"]
        d["Low"] = d["low"]
        d["Close"] = d["close"]
        d["Volume"] = d["volume"]

        logging.debug(f"_prepare_data: final d shape={d.shape}")

        records = []
        for _, row in d.iterrows():
            records.append(
                {
                    "time": int(row["time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )

        logging.debug(f"_prepare_data: returning {len(records)} records")
        self.prepared_df = d.copy()  # Store transformed DF for Python-side calculations
        return records

    # ========== Chart Controls ==========

    def toggle_magnet(self, enabled):
        if self._is_ready:
            self.page().runJavaScript(f"window.toggleMagnet({str(enabled).lower()})")

    def set_scale_mode(self, mode):
        if self._is_ready:
            self.page().runJavaScript(f"window.setScaleMode('{mode}')")

    def toggle_auto_scale(self, enabled):
        if self._is_ready:
            self.page().runJavaScript(f"window.toggleAutoScale({str(enabled).lower()})")

    def toggle_grid(self, enabled):
        if self._is_ready:
            self.page().runJavaScript(f"window.toggleGrid({str(enabled).lower()})")

    def set_crosshair_mode(self, mode):
        if self._is_ready:
            self.page().runJavaScript(f"window.setCrosshairMode({mode})")

    def fit_content(self):
        if self._is_ready:
            self.page().runJavaScript("window.fitContent()")

    def goto_time(self, timestamp):
        if self._is_ready:
            self.page().runJavaScript(f"window.gotoTime({timestamp})")

    # ========== Drawing Tools ==========

    def enable_horizontal_line(self):
        if self._is_ready:
            self.page().runJavaScript("window.enableHorizontalLine()")
            self.drawing_mode_changed.emit("horizontal")

    def enable_vertical_line(self):
        if self._is_ready:
            self.page().runJavaScript("window.enableVerticalLine()")
            self.drawing_mode_changed.emit("vertical")

    def enable_trend_line(self):
        if self._is_ready:
            self.page().runJavaScript("window.enableTrendLine()")
            self.drawing_mode_changed.emit("trendline")

    def enable_measure_tool(self):
        if self._is_ready:
            self.page().runJavaScript("window.enableMeasureTool()")
            self.drawing_mode_changed.emit("measure")

    def clear_drawings(self):
        if self._is_ready:
            self.page().runJavaScript("window.clearDrawings()")
            self.drawing_mode_changed.emit("")

    def cancel_drawing(self):
        if self._is_ready:
            self.page().runJavaScript("window.cancelDrawing()")
            self.drawing_mode_changed.emit("")

    def delete_selected_drawing(self):
        if self._is_ready:
            self.page().runJavaScript("window.deleteSelectedDrawing()")

    # ========== Indicators ==========

    def add_indicator(self, ind_id, ind_type, params, color):
        logging.debug(
            f"[DEBUG] add_indicator called: id={ind_id}, type={ind_type}, params={params}, color={color}, ready={self._is_ready}"
        )
        if self._is_ready:
            # TA-Lib Calculation for MACD (Python-side)
            if ind_type == "MACD" and talib and hasattr(self, "prepared_df"):
                try:
                    df = self.prepared_df
                    close_prices = df["close"].values

                    fastPeriod = int(params.get("fastPeriod", 12))
                    slowPeriod = int(params.get("slowPeriod", 26))
                    signalPeriod = int(params.get("signalPeriod", 9))

                    m, s, h = talib.MACD(
                        close_prices,
                        fastperiod=fastPeriod,
                        slowperiod=slowPeriod,
                        signalperiod=signalPeriod,
                    )

                    macd_line = []
                    signal_line = []
                    hist_line = []

                    for i in range(len(df)):
                        if not (np.isnan(m[i]) or np.isnan(s[i])):
                            t = int(df.iloc[i]["time"])
                            macd_line.append({"time": t, "value": float(m[i])})
                            signal_line.append({"time": t, "value": float(s[i])})
                            if not np.isnan(h[i]):
                                hist_line.append({"time": t, "value": float(h[i])})

                    params["talib_data"] = {
                        "macd": macd_line,
                        "signal": signal_line,
                        "hist": hist_line,
                    }
                    logging.info(
                        f"[DEBUG] Calculated MACD using TA-Lib: {len(macd_line)} points"
                    )
                except Exception as e:
                    logging.error(f"TA-Lib MACD calculation error: {e}")

            # CUSTOM SCRIPT EXECUTION
            elif ind_type == "CUSTOM" and hasattr(self, "prepared_df"):
                code = params.get("code", "")
                try:
                    # Isolated namespace for user script
                    # We pass a copy to avoid scripts breaking the original
                    exec_df = self.prepared_df.copy()
                    namespace = {
                        "df": exec_df,
                        "talib": talib,
                        "np": np,
                        "pd": pd,
                        "result": None,
                    }
                    # We use namespace as BOTH globals and locals to ensure
                    # functions defined in the script can see the variables.
                    exec(code, namespace)

                    # If user assigned 'result', use it. Otherwise, if they modified 'df' and it's returned, use it.
                    result = namespace.get("result")
                    if result is None:
                        # Fallback: Check if df was modified at all (rough check)
                        if len(namespace.get("df").columns) > len(
                            self.prepared_df.columns
                        ):
                            result = namespace.get("df")
                        else:
                            result = namespace.get("df")

                    if result is not None:
                        params["talib_data"] = {}
                        params["markers"] = []

                        if isinstance(result, pd.DataFrame):
                            # Handle DataFrame: extract numeric columns as lines, boolean/signal as markers
                            for col in result.columns:
                                if col in [
                                    "time",
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                    "Open",
                                    "High",
                                    "Low",
                                    "Close",
                                    "Volume",
                                ]:
                                    continue

                                series = result[col]
                                # Check if it's a signal column (Boolean or sparse numeric)
                                if col.upper() in [
                                    "BUY",
                                    "SELL",
                                    "SIGNAL",
                                    "BUY_SIGNAL",
                                    "SELL_SIGNAL",
                                ]:
                                    marker_type = (
                                        "buy"
                                        if "BUY" in col.upper()
                                        else (
                                            "sell"
                                            if "SELL" in col.upper()
                                            else "signal"
                                        )
                                    )
                                    self._extract_markers(
                                        result, col, marker_type, params["markers"]
                                    )
                                elif pd.api.types.is_numeric_dtype(series):
                                    params["talib_data"][str(col)] = (
                                        self._serialize_series(series)
                                    )

                        elif isinstance(result, dict):
                            # Multi-series dict
                            for key, series in result.items():
                                if str(key).upper() in ["BUY", "SELL", "SIGNAL"]:
                                    marker_type = str(key).lower()
                                    self._extract_markers(
                                        self.prepared_df,
                                        series,
                                        marker_type,
                                        params["markers"],
                                    )
                                else:
                                    params["talib_data"][str(key)] = (
                                        self._serialize_series(series)
                                    )
                        else:
                            # Single series
                            params["talib_data"]["main"] = self._serialize_series(
                                result
                            )

                        logging.info(
                            f"[DEBUG] Executed custom script successfully. Lines: {list(params['talib_data'].keys())}, Markers: {len(params['markers'])}"
                        )
                except Exception as e:
                    logging.error(f"Custom indicator execution error: {e}")
                    params["error"] = str(e)
                    import traceback

                    traceback.print_exc()

            params_json = json.dumps(params)
            js_code = f"window.addIndicator('{ind_id}', '{ind_type}', {params_json}, '{color}')"
            logging.debug(f"[DEBUG] Running JS: {js_code[:200]}")

            def on_result(result):
                logging.debug(f"[DEBUG] addIndicator JS result: {result}")

            self.page().runJavaScript(js_code, on_result)

    def _serialize_series(self, series):
        """Standardize various outputs (numpy, pandas, list) into JS-ready format"""
        try:
            if hasattr(series, "values"):  # Pandas Series
                data = series.values
            elif isinstance(series, list):
                data = np.array(series)
            else:
                data = series  # Assume numpy array

            clean_data = []
            df = self.prepared_df
            for i in range(len(df)):
                val = data[i]
                if not np.isnan(val):
                    clean_data.append(
                        {"time": int(df.iloc[i]["time"]), "value": float(val)}
                    )
            return clean_data
        except Exception as e:
            logging.error(f"Serialization error: {e}")
            return []

    def _extract_markers(self, df, series_or_col, marker_type, markers_list):
        """Extract BUY/SELL markers from a boolean or signal column"""
        try:
            if isinstance(series_or_col, str):
                series = df[series_or_col]
            else:
                series = series_or_col

            # Ensure we have a boolean-like series
            if hasattr(series, "values"):
                vals = series.values
            else:
                vals = np.array(series_or_col)

            for i in range(len(df)):
                if vals[i] == True or (
                    isinstance(vals[i], (int, float)) and vals[i] > 0
                ):
                    # For BUY, place at LOW. For SELL, place at HIGH.
                    time_val = int(df.iloc[i]["time"])
                    pos = "belowBar" if marker_type == "buy" else "aboveBar"
                    shape = "arrowUp" if marker_type == "buy" else "arrowDown"
                    color = "#26a69a" if marker_type == "buy" else "#ef5350"
                    text = "BUY" if marker_type == "buy" else "SELL"

                    markers_list.append(
                        {
                            "time": time_val,
                            "position": pos,
                            "color": color,
                            "shape": shape,
                            "text": text,
                        }
                    )
        except Exception as e:
            logging.error(f"Marker extraction error: {e}")

    def remove_indicator(self, ind_id):
        if self._is_ready:
            logging.info(f"[DEBUG] removing indicator via JS: {ind_id}")
            self.page().runJavaScript(f"window.removeIndicator('{ind_id}')")
        else:
            logging.info(
                f"[DEBUG] remove_indicator called but chart not ready: {ind_id}"
            )

    def update_indicator_color(self, ind_id, color):
        if self._is_ready:
            self.page().runJavaScript(
                f"window.updateIndicatorColor('{ind_id}', '{color}')"
            )

    def clear_indicators(self):
        if self._is_ready:
            self.page().runJavaScript("window.clearIndicators()")

    # ========== Theme ==========

    def apply_theme(self, theme):
        """Apply theme changes to the chart"""
        if self._is_ready:
            colors = self._theme_manager.get_chart_colors(theme)
            js_code = f"""
                window.applyTheme({json.dumps(colors)})
            """
            self.page().runJavaScript(js_code)

    def init_chart(self):
        colors = self._theme_manager.get_chart_colors()

        html = (
            """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { 
            width: 100%; 
            height: 100%; 
            background: """
            + colors["bg"]
            + """; 
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        #chart-container { 
            width: 100%; 
            height: 100%; 
            position: absolute;
            top: 0;
            left: 0;
            cursor: crosshair;
        }
        .legend {
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1000;
            background: rgba(19, 23, 34, 0.95);
            padding: 12px 15px;
            border-radius: 8px;
            color: #d1d4dc;
            font-size: 12px;
            border: 1px solid #2a2e39;
            min-width: 250px;
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .legend .ticker { 
            font-size: 18px; 
            font-weight: bold; 
            color: #2196F3;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        .legend .ohlc { 
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 8px;
        }
        .legend .ohlc-item { text-align: center; }
        .legend .ohlc-label { 
            color: #737780; 
            font-size: 10px; 
            font-weight: 600;
            margin-bottom: 2px;
        }
        .legend .ohlc-value { 
            font-weight: bold; 
            font-size: 13px; 
        }
        .indicator-values {
            margin-top: 8px;
            border-top: 1px solid #2a2e39;
            padding-top: 8px;
        }
        .indicator-item {
            display: flex;
            justify-content: space-between;
            margin: 3px 0;
            font-size: 11px;
        }
        .indicator-name {
            font-weight: 600;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #737780;
            font-size: 16px;
            z-index: 500;
        }
        #status-msg {
            position: absolute;
            bottom: 10px;
            right: 10px;
            color: #26a69a;
            font-size: 11px;
            z-index: 1000;
            background: rgba(19, 23, 34, 0.9);
            padding: 6px 12px;
            border-radius: 4px;
            pointer-events: none;
        }
        #measure-tooltip {
            position: absolute;
            z-index: 2000;
            background: rgba(33, 150, 243, 0.95);
            color: white;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            pointer-events: none;
            display: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            white-space: nowrap;
        }
        #measure-tooltip .measure-row {
            margin: 3px 0;
        }
        #measure-tooltip .measure-label {
            color: rgba(255,255,255,0.7);
            margin-right: 6px;
        }
        #measure-tooltip .measure-value {
            font-weight: bold;
        }
        #measure-tooltip .positive { color: #81C784; }
        #measure-tooltip .negative { color: #E57373; }
        #measure-tooltip .from-to {
            font-size: 10px;
            color: rgba(255,255,255,0.5);
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }
        
        #measure-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1500;
            display: none;
        }
        #measure-overlay svg {
            width: 100%;
            height: 100%;
        }
        
        /* Drawing preview overlay */
        #drawing-preview {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1400;
            display: none;
        }
        #drawing-preview svg {
            width: 100%;
            height: 100%;
        }
    </style>
</head>
<body>
    <div id="loading">Initializing chart...</div>
    <div id="status-msg"></div>
    <div id="measure-tooltip"></div>
    <div id="measure-overlay">
        <svg>
            <line id="measure-line" stroke="#2196F3" stroke-width="2" stroke-dasharray="6,4"/>
            <circle id="measure-start-dot" r="5" fill="#2196F3"/>
            <circle id="measure-end-dot" r="4" fill="#2196F3"/>
        </svg>
    </div>
    <div id="drawing-preview">
        <svg>
            <!-- Trend line preview -->
            <line id="trendline-preview" stroke="#9C27B0" stroke-width="2" stroke-dasharray="6,4"/>
        </svg>
    </div>
    <div id="chart-container"></div>
    <div id="legend" class="legend" style="display: none;">
        <div class="ticker" id="ticker">-</div>
        <div class="ohlc">
            <div class="ohlc-item"><div class="ohlc-label">OPEN</div><div class="ohlc-value" id="open-val">-</div></div>
            <div class="ohlc-item"><div class="ohlc-label">HIGH</div><div class="ohlc-value" id="high-val">-</div></div>
            <div class="ohlc-item"><div class="ohlc-label">LOW</div><div class="ohlc-value" id="low-val">-</div></div>
            <div class="ohlc-item"><div class="ohlc-label">CLOSE</div><div class="ohlc-value" id="close-val">-</div></div>
            <div class="ohlc-item"><div class="ohlc-label">% CHG</div><div class="ohlc-value" id="pct-change">-</div></div>
        </div>
        <div class="indicator-values" id="indicator-values"></div>
    </div>

    <script>
        window.chartReady = false;
        let chart = null;
        let candleSeries = null;
        let volumeSeries = null;
        let candleData = [];
        let indicators = {};
        
        // Drawing state
        let drawings = [];
        let drawingMode = null;
        let drawingPoints = [];
        let selectedDrawingId = null;
        let drawingIdCounter = 0;
        let magnetEnabled = false;
        
        // Measure tool state
        let measureStartPoint = null;
        let measureStartPixel = null;
        
        // Throttle control
        let lastUpdateTime = 0;
        const THROTTLE_MS = 16;
        
        // Last known crosshair data for magnet
        let lastCrosshairData = null;
        
        // Next pane index for panel indicators (pane 0 = main chart)
        let nextPaneIndex = 1;
        
        // Chart colors (will be updated by applyTheme)
        let chartColors = {
            bg: '#131722',
            text: '#d1d4dc',
            grid: '#1e222d',
            border: '#2a2e39',
            up: '#26a69a',
            down: '#ef5350'
        };

        function setStatus(msg, color) {
            const el = document.getElementById('status-msg');
            if (el) {
                el.textContent = msg;
                el.style.color = color || '#26a69a';
            }
        }

        function hideLoading() {
            const el = document.getElementById('loading');
            if (el) el.style.display = 'none';
        }

        function showLegend() {
            const el = document.getElementById('legend');
            if (el) el.style.display = 'block';
        }

        function loadScript(url) {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = url;
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }

        // Get price at crosshair Y position
        function getPriceFromY(y) {
            if (!chart || !candleSeries) return null;
            try {
                return candleSeries.coordinateToPrice(y);
            } catch (e) {
                return null;
            }
        }

        // Get the appropriate price based on magnet mode
        function getDrawingPrice(param) {
            if (magnetEnabled) {
                // Magnet mode: snap to close price of the candle
                if (param.seriesData) {
                    const data = param.seriesData.get(candleSeries);
                    if (data && data.close !== undefined) {
                        return data.close;
                    }
                }
                // Fallback: use lastCrosshairData
                if (lastCrosshairData && lastCrosshairData.close !== undefined) {
                    return lastCrosshairData.close;
                }
            }
            
            // Non-magnet mode: use actual crosshair Y position
            if (param.point && param.point.y !== undefined) {
                const price = getPriceFromY(param.point.y);
                if (price !== null) return price;
            }
            
            // Fallback
            if (param.seriesData) {
                const data = param.seriesData.get(candleSeries);
                if (data) return (data.high + data.low) / 2;
            }
            return null;
        }
        
        // Get pixel Y coordinate for a price (for magnet snapping visual)
        function getYFromPrice(price) {
            if (!chart || !candleSeries) return null;
            try {
                return candleSeries.priceToCoordinate(price);
            } catch (e) {
                return null;
            }
        }

        // Find bar index for a given time
        function findBarIndex(time) {
            if (!candleData.length) return 0;
            let closest = 0;
            let minDiff = Math.abs(candleData[0].time - time);
            for (let i = 1; i < candleData.length; i++) {
                const diff = Math.abs(candleData[i].time - time);
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = i;
                }
            }
            return closest;
        }
        
        // Get pixel position for a time (for trend line preview)
        function getPixelFromTime(time) {
            if (!chart || !candleData.length) return null;
            try {
                const logicalIndex = findBarIndex(time);
                const coords = chart.timeScale().coordinateToLogical(logicalIndex);
                if (coords !== null) {
                    return coords;
                }
            } catch (e) {
                // Fallback: estimate position based on data
            }
            return null;
        }

        async function initChart() {
            try {
                console.log('Loading lightweight-charts...');
                setStatus('Loading library...', '#FF9800');
                
                const urls = [
                    'qrc:///qtres/assets/lightweight-charts.standalone.production.js',
                    'https://unpkg.com/lightweight-charts@5.0.2/dist/lightweight-charts.standalone.production.js'
                ];
                
                let loaded = false;
                for (const url of urls) {
                    try {
                        await loadScript(url);
                        console.log('Loaded from:', url);
                        loaded = true;
                        break;
                    } catch (e) {
                        console.warn('Failed:', url);
                    }
                }
                
                if (!loaded) throw new Error('Failed to load library');
                
                if (!loaded) throw new Error('Failed to load library');
                if (typeof LightweightCharts === 'undefined') throw new Error('LightweightCharts undefined');
                
                const container = document.getElementById('chart-container');
                
                chart = LightweightCharts.createChart(container, {
                    width: container.clientWidth,
                    height: container.clientHeight,
                    layout: {
                        background: { type: 'solid', color: chartColors.bg },
                        textColor: chartColors.text,
                    },
                    grid: {
                        vertLines: { color: chartColors.grid, style: 1 },
                        horzLines: { color: chartColors.grid, style: 1 },
                    },
                    crosshair: {
                        mode: LightweightCharts.CrosshairMode.Normal,
                        vertLine: { 
                            color: '#758696', 
                            width: 1,
                            style: 3,
                            labelBackgroundColor: '#2196F3' 
                        },
                        horzLine: { 
                            color: '#758696', 
                            width: 1,
                            style: 3,
                            labelBackgroundColor: '#2196F3' 
                        },
                    },
                    rightPriceScale: { 
                        borderColor: chartColors.border,
                        scaleMargins: { top: 0.1, bottom: 0.2 },
                        mode: 0,
                    },
                    timeScale: { 
                        borderColor: chartColors.border,
                        timeVisible: true,
                        secondsVisible: false,
                        rightOffset: 12,
                        barSpacing: 6,
                        fixLeftEdge: true,
                        lockVisibleTimeRangeOnResize: true,
                    },
                    handleScroll: {
                        mouseWheel: true,
                        pressedMouseMove: true,
                        horzTouchDrag: true,
                        vertTouchDrag: true,
                    },
                    handleScale: {
                        axisPressedMouseMove: true,
                        mouseWheel: true,
                        pinch: true,
                    },
                });

                // Keep track of indicator panels
                let panelCount = 0;
                const panels = {};
                
                // Create additional price scale for indicators
                window.createIndicatorPanel = function(id, heightPercent) {
                    const panelId = 'panel_' + id;
                    const priceScale = chart.priceScale(panelId);
                    priceScale.applyOptions({
                        scaleMargins: { top: heightPercent || 0.7, bottom: 0 },
                        borderColor: '#2a2e39',
                    });
                    panels[id] = panelId;
                    return panelId;
                };
                
                // Add indicator to a separate panel
                window.addIndicatorToPanel = function(id, series, panelId) {
                    if (panelId) {
                        series.applyOptions({ priceScaleId: panelId });
                    }
                };
                
                candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
                    upColor: chartColors.up,
                    downColor: chartColors.down,
                    borderUpColor: chartColors.up,
                    borderDownColor: chartColors.down,
                    wickUpColor: chartColors.up,
                    wickDownColor: chartColors.down,
                });

                volumeSeries = chart.addSeries(LightweightCharts.HistogramSeries, {
                    color: chartColors.up,
                    priceFormat: { type: 'volume' },
                    priceScaleId: 'volume',
                });
                chart.priceScale('volume').applyOptions({
                    scaleMargins: { top: 0.82, bottom: 0 }  // Reduced from 0.85 to give more space
                });

                chart.subscribeCrosshairMove(handleCrosshairMove);
                chart.subscribeClick(handleChartClick);
                chart.subscribeCrosshairMove(handleTrendLinePreview);

                document.addEventListener('keydown', handleKeyDown);

                new ResizeObserver(entries => {
                    if (!chart || entries.length === 0) return;
                    const { width, height } = entries[0].contentRect;
                    chart.applyOptions({ width, height });
                }).observe(container);

                window.chartReady = true;
                hideLoading();
                setStatus('Ready', '#26a69a');
                console.log('Chart initialized');

            } catch (error) {
                console.error('Init error:', error);
                setStatus('Error: ' + error.message, '#ef5350');
            }
        }

        function handleKeyDown(e) {
            if (e.key === 'Delete' || e.key === 'Backspace') {
                if (selectedDrawingId !== null) {
                    deleteSelectedDrawing();
                    e.preventDefault();
                }
            }
        }
        
        // Handle trend line preview during drawing
        function handleTrendLinePreview(param) {
            if (drawingMode === 'trendline' && drawingPoints.length === 1 && param.point) {
                const previewLine = document.getElementById('trendline-preview');
                const previewOverlay = document.getElementById('drawing-preview');
                
                if (previewLine && previewOverlay) {
                    const startPoint = drawingPoints[0];
                    const startY = getYFromPrice(startPoint.price);
                    
                    if (startY !== null) {
                        // Get the X position for the first point (from time)
                        const logicalIndex = findBarIndex(startPoint.time);
                        const logicalRange = chart.timeScale().getVisibleLogicalRange();
                        
                        if (logicalRange) {
                            const barWidth = chart.timeScale().width() / (logicalRange.to - logicalRange.from);
                            const startX = (logicalIndex - logicalRange.from) * barWidth;
                            
                            // Current crosshair position
                            const endX = param.point.x;
                            const endY = param.point.y;
                            
                            // Snap Y to price if magnet enabled
                            let snappedEndY = endY;
                            if (magnetEnabled) {
                                const snappedY = getDrawingPrice(param);
                                if (snappedY !== null) {
                                    snappedEndY = getYFromPrice(snappedY);
                                }
                            }
                            
                            previewLine.setAttribute('x1', startX);
                            previewLine.setAttribute('y1', startY);
                            previewLine.setAttribute('x2', endX);
                            previewLine.setAttribute('y2', snappedEndY);
                            
                            previewOverlay.style.display = 'block';
                        }
                    }
                }
            } else {
                document.getElementById('drawing-preview').style.display = 'none';
            }
        }

        function handleCrosshairMove(param) {
            const now = Date.now();
            if (now - lastUpdateTime < THROTTLE_MS) return;
            lastUpdateTime = now;
            
            if (!param || !param.time || !param.seriesData) {
                return;
            }
            
            const data = param.seriesData.get(candleSeries);
            if (data) {
                // Store for magnet snapping
                lastCrosshairData = data;
                
                const color = data.close >= data.open ? '#26a69a' : '#ef5350';
                document.getElementById('open-val').textContent = data.open.toFixed(2);
                document.getElementById('open-val').style.color = color;
                document.getElementById('high-val').textContent = data.high.toFixed(2);
                document.getElementById('high-val').style.color = color;
                document.getElementById('low-val').textContent = data.low.toFixed(2);
                document.getElementById('low-val').style.color = color;
                document.getElementById('close-val').textContent = data.close.toFixed(2);
                document.getElementById('close-val').style.color = color;
                
                // Calculate % change from previous close
                const pctChangeEl = document.getElementById('pct-change');
                // Find previous candle
                const timeValue = data.time;
                let prevClose = null;
                for (let i = candleData.length - 1; i >= 0; i--) {
                    if (candleData[i].time < timeValue) {
                        prevClose = candleData[i].close;
                        break;
                    }
                }
                if (prevClose && prevClose > 0) {
                    const pctChange = ((data.close - prevClose) / prevClose) * 100;
                    pctChangeEl.textContent = (pctChange >= 0 ? '+' : '') + pctChange.toFixed(2) + '%';
                    pctChangeEl.style.color = pctChange >= 0 ? '#26a69a' : '#ef5350';
                } else {
                    pctChangeEl.textContent = '-';
                    pctChangeEl.style.color = color;
                }
            }
            
            // Update indicator values
            let html = '';
            for (const [id, ind] of Object.entries(indicators)) {
                const val = param.seriesData.get(ind.series);
                if (val && val.value !== undefined) {
                    html += `<div class="indicator-item">
                        <span class="indicator-name" style="color:${ind.color}">${ind.name}</span>
                        <span>${val.value.toFixed(2)}</span>
                    </div>`;
                }
            }
            document.getElementById('indicator-values').innerHTML = html;
            
            // Update measure tool if active
            if (drawingMode === 'measure' && measureStartPoint && param.point) {
                updateMeasureDisplay(param);
            }
        }

        function updateMeasureDisplay(param) {
            const tooltip = document.getElementById('measure-tooltip');
            const overlay = document.getElementById('measure-overlay');
            const line = document.getElementById('measure-line');
            const startDot = document.getElementById('measure-start-dot');
            const endDot = document.getElementById('measure-end-dot');
            
            if (!measureStartPoint || !measureStartPixel || !param.point) {
                tooltip.style.display = 'none';
                overlay.style.display = 'none';
                return;
            }
            
            const currentPrice = getDrawingPrice(param);
            const currentTime = param.time;
            
            if (currentPrice === null || !currentTime) {
                return;
            }
            
            // Calculate differences
            const priceDiff = currentPrice - measureStartPoint.price;
            const pricePercent = (priceDiff / measureStartPoint.price) * 100;
            const startIdx = findBarIndex(measureStartPoint.time);
            const endIdx = findBarIndex(currentTime);
            const barsDiff = endIdx - startIdx;
            
            const colorClass = priceDiff >= 0 ? 'positive' : 'negative';
            const sign = priceDiff >= 0 ? '+' : '';
            
            // Update tooltip
            tooltip.innerHTML = `
                <div class="measure-row">
                    <span class="measure-label">Points:</span>
                    <span class="measure-value ${colorClass}">${sign}${priceDiff.toFixed(2)}</span>
                </div>
                <div class="measure-row">
                    <span class="measure-label">Percent:</span>
                    <span class="measure-value ${colorClass}">${sign}${pricePercent.toFixed(2)}%</span>
                </div>
                <div class="measure-row">
                    <span class="measure-label">Bars:</span>
                    <span class="measure-value">${barsDiff}</span>
                </div>
                <div class="from-to">
                    ${measureStartPoint.price.toFixed(2)} → ${currentPrice.toFixed(2)}
                </div>
            `;
            
            // Position tooltip
            const tooltipX = Math.min(param.point.x + 25, window.innerWidth - 180);
            const tooltipY = Math.max(param.point.y - 100, 10);
            tooltip.style.display = 'block';
            tooltip.style.left = tooltipX + 'px';
            tooltip.style.top = tooltipY + 'px';
            
            // Get end Y position - snap to price if magnet enabled
            let endY = param.point.y;
            if (magnetEnabled) {
                const snappedY = getYFromPrice(currentPrice);
                if (snappedY !== null) {
                    endY = snappedY;
                }
            }
            
            // Draw SVG line
            line.setAttribute('x1', measureStartPixel.x);
            line.setAttribute('y1', measureStartPixel.y);
            line.setAttribute('x2', param.point.x);
            line.setAttribute('y2', endY);
            
            startDot.setAttribute('cx', measureStartPixel.x);
            startDot.setAttribute('cy', measureStartPixel.y);
            
            endDot.setAttribute('cx', param.point.x);
            endDot.setAttribute('cy', endY);
            
            overlay.style.display = 'block';
        }

        function hideMeasureTool() {
            measureStartPoint = null;
            measureStartPixel = null;
            document.getElementById('measure-tooltip').style.display = 'none';
            document.getElementById('measure-overlay').style.display = 'none';
        }

        function handleChartClick(param) {
            if (!param.time) return;
            
            const clickPrice = getDrawingPrice(param);
            if (clickPrice === null) return;
            
            // Hide preview when clicking
            document.getElementById('drawing-preview').style.display = 'none';
            
            // If not in drawing mode, try to select a drawing
            if (!drawingMode) {
                trySelectDrawing(param.time, clickPrice);
                return;
            }
            
            if (drawingMode === 'horizontal') {
                addHorizontalLine(clickPrice);
                drawingMode = null;
                setStatus('H-Line added (click to select, Del to delete)', '#26a69a');
                
            } else if (drawingMode === 'vertical') {
                addVerticalLine(param.time);
                drawingMode = null;
                setStatus('V-Line added (click to select, Del to delete)', '#26a69a');
                
            } else if (drawingMode === 'trendline') {
                drawingPoints.push({ time: param.time, price: clickPrice });
                
                if (drawingPoints.length === 1) {
                    setStatus('Click second point for trend line', '#FF9800');
                } else if (drawingPoints.length === 2) {
                    addTrendLine(drawingPoints[0], drawingPoints[1]);
                    drawingPoints = [];
                    drawingMode = null;
                    setStatus('Trend line added (click to select, Del to delete)', '#26a69a');
                }
                
            } else if (drawingMode === 'measure') {
                if (!measureStartPoint) {
                    // First click - set start point
                    measureStartPoint = { time: param.time, price: clickPrice };
                    
                    // Get pixel position - snap if magnet
                    let startY = param.point.y;
                    if (magnetEnabled) {
                        const snappedY = getYFromPrice(clickPrice);
                        if (snappedY !== null) {
                            startY = snappedY;
                        }
                    }
                    measureStartPixel = { x: param.point.x, y: startY };
                    
                    setStatus('Move cursor to measure, click to finish', '#2196F3');
                } else {
                    // Second click - finish measurement
                    hideMeasureTool();
                    drawingMode = null;
                    setStatus('Measurement complete', '#26a69a');
                }
            }
        }

        function trySelectDrawing(time, price) {
            if (selectedDrawingId !== null) {
                deselectDrawing();
            }
            
            const tolerance = 0.005;
            
            for (const drawing of drawings) {
                if (drawing.type === 'horizontal') {
                    if (Math.abs(price - drawing.price) / drawing.price < tolerance) {
                        selectDrawing(drawing.id);
                        return;
                    }
                } else if (drawing.type === 'vertical') {
                    const clickIdx = findBarIndex(time);
                    const lineIdx = findBarIndex(drawing.time);
                    if (Math.abs(clickIdx - lineIdx) <= 1) {
                        selectDrawing(drawing.id);
                        return;
                    }
                } else if (drawing.type === 'trendline') {
                    if (isNearTrendLine(drawing, time, price, tolerance)) {
                        selectDrawing(drawing.id);
                        return;
                    }
                }
            }
        }

        function isNearTrendLine(drawing, clickTime, clickPrice, tolerance) {
            const p1 = drawing.point1;
            const p2 = drawing.point2;
            
            const minTime = Math.min(p1.time, p2.time);
            const maxTime = Math.max(p1.time, p2.time);
            const timeMargin = (maxTime - minTime) * 0.1;
            
            if (clickTime < minTime - timeMargin || clickTime > maxTime + timeMargin) return false;
            
            const timeDiff = p2.time - p1.time;
            if (timeDiff === 0) return Math.abs(clickPrice - p1.price) / p1.price < tolerance;
            
            const ratio = (clickTime - p1.time) / timeDiff;
            const expectedPrice = p1.price + (p2.price - p1.price) * ratio;
            
            return Math.abs(clickPrice - expectedPrice) / expectedPrice < tolerance * 2;
        }

        function selectDrawing(id) {
            selectedDrawingId = id;
            
            const drawing = drawings.find(d => d.id === id);
            if (!drawing) return;
            
            if (drawing.type === 'horizontal' && drawing.line) {
                drawing.originalColor = drawing.line.options().color;
                candleSeries.removePriceLine(drawing.line);
                drawing.line = candleSeries.createPriceLine({
                    price: drawing.price,
                    color: '#FFD700',
                    lineWidth: 3,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    axisLabelVisible: true,
                    title: '● H-Line',
                });
            } else if (drawing.series) {
                drawing.originalColor = drawing.series.options().color;
                drawing.series.applyOptions({ color: '#FFD700', lineWidth: 3 });
            }
            
            setStatus('Drawing selected - Press Delete to remove', '#FFD700');
        }

        function deselectDrawing() {
            if (selectedDrawingId === null) return;
            
            const drawing = drawings.find(d => d.id === selectedDrawingId);
            if (drawing) {
                if (drawing.type === 'horizontal' && drawing.line) {
                    const originalColor = drawing.originalColor || '#2196F3';
                    candleSeries.removePriceLine(drawing.line);
                    drawing.line = candleSeries.createPriceLine({
                        price: drawing.price,
                        color: originalColor,
                        lineWidth: 2,
                        lineStyle: LightweightCharts.LineStyle.Solid,
                        axisLabelVisible: true,
                        title: 'H-Line',
                    });
                } else if (drawing.series) {
                    const originalColor = drawing.originalColor || '#2196F3';
                    drawing.series.applyOptions({ color: originalColor, lineWidth: 2 });
                }
            }
            
            selectedDrawingId = null;
            setStatus('Ready', '#26a69a');
        }

        window.deleteSelectedDrawing = function() {
            if (selectedDrawingId === null) {
                setStatus('No drawing selected', '#737780');
                return;
            }
            
            const idx = drawings.findIndex(d => d.id === selectedDrawingId);
            if (idx === -1) return;
            
            const drawing = drawings[idx];
            
            if (drawing.type === 'horizontal' && drawing.line) {
                candleSeries.removePriceLine(drawing.line);
            } else if (drawing.series) {
                chart.removeSeries(drawing.series);
            }
            
            drawings.splice(idx, 1);
            selectedDrawingId = null;
            setStatus('Drawing deleted', '#26a69a');
        };

        function addHorizontalLine(price) {
            const id = ++drawingIdCounter;
            const line = candleSeries.createPriceLine({
                price: price,
                color: '#2196F3',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                axisLabelVisible: true,
                title: 'H-Line',
            });
            drawings.push({ id, type: 'horizontal', line, price });
        }

        function addVerticalLine(time) {
            const id = ++drawingIdCounter;
            
            const visibleRange = chart.timeScale().getVisibleLogicalRange();
            let minPrice = Infinity, maxPrice = -Infinity;
            
            if (visibleRange && candleData.length) {
                const from = Math.max(0, Math.floor(visibleRange.from));
                const to = Math.min(candleData.length - 1, Math.ceil(visibleRange.to));
                
                for (let i = from; i <= to; i++) {
                    if (candleData[i]) {
                        minPrice = Math.min(minPrice, candleData[i].low);
                        maxPrice = Math.max(maxPrice, candleData[i].high);
                    }
                }
            }
            
            if (minPrice === Infinity || maxPrice === -Infinity) {
                minPrice = Math.min(...candleData.map(d => d.low));
                maxPrice = Math.max(...candleData.map(d => d.high));
            }
            
            const padding = (maxPrice - minPrice) * 0.1;
            minPrice -= padding;
            maxPrice += padding;
            
            const lineSeries = chart.addSeries(LightweightCharts.LineSeries, {
                color: '#FF9800',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
            });
            
            lineSeries.setData([
                { time: time, value: minPrice },
                { time: time, value: maxPrice }
            ]);
            
            drawings.push({ id, type: 'vertical', series: lineSeries, time });
        }

        function addTrendLine(point1, point2) {
            const id = ++drawingIdCounter;
            const lineSeries = chart.addSeries(LightweightCharts.LineSeries, {
                color: '#9C27B0',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
            });
            
            lineSeries.setData([
                { time: point1.time, value: point1.price },
                { time: point2.time, value: point2.price }
            ]);
            
            drawings.push({ id, type: 'trendline', series: lineSeries, point1, point2 });
        }

        window.setChartData = function(data, ticker) {
            try {
                if (!chart || !candleSeries) {
                    console.error('Chart not ready');
                    return false;
                }
                
                const parsed = typeof data === 'string' ? JSON.parse(data) : data;
                console.log('Setting', parsed.length, 'candles');
                
                if (!parsed || parsed.length === 0) {
                    console.error('Empty data');
                    return false;
                }
                
                document.getElementById('ticker').textContent = ticker;
                candleData = parsed;
                
                candleSeries.setData(parsed);
                
                const volData = parsed.map(d => ({
                    time: d.time,
                    value: d.volume || 0,
                    color: d.close >= d.open ? 'rgba(38,166,154,0.5)' : 'rgba(239,83,80,0.5)'
                }));
                volumeSeries.setData(volData);
                
                chart.timeScale().fitContent();
                showLegend();
                setStatus(parsed.length + ' candles loaded', '#26a69a');
                
                return true;
                
            } catch (e) {
                console.error('setChartData error:', e);
                return false;
            }
        };

        // ========== CHART CONTROLS ==========

        window.toggleMagnet = function(enabled) {
            magnetEnabled = enabled;
            if (!chart) return;
            chart.applyOptions({
                crosshair: {
                    mode: enabled ? LightweightCharts.CrosshairMode.Magnet : LightweightCharts.CrosshairMode.Normal
                }
            });
            console.log('Magnet mode:', enabled);
        };

        window.setScaleMode = function(mode) {
            if (!chart) return;
            chart.priceScale('right').applyOptions({ 
                mode: mode === 'logarithmic' ? 1 : 0 
            });
        };

        window.toggleAutoScale = function(enabled) {
            if (!chart) return;
            chart.priceScale('right').applyOptions({ autoScale: enabled });
        };

        window.toggleGrid = function(enabled) {
            if (!chart) return;
            const color = enabled ? '#1e222d' : 'transparent';
            chart.applyOptions({
                grid: {
                    vertLines: { color },
                    horzLines: { color },
                }
            });
        };

        window.setCrosshairMode = function(mode) {
            if (!chart) return;
            const modes = [
                LightweightCharts.CrosshairMode.Normal,
                LightweightCharts.CrosshairMode.Magnet,
                LightweightCharts.CrosshairMode.Hidden
            ];
            chart.applyOptions({ crosshair: { mode: modes[mode] || modes[0] } });
        };

        window.fitContent = function() {
            if (!chart) return;
            chart.timeScale().fitContent();
            setStatus('Fit to content', '#26a69a');
        };

        window.gotoTime = function(timestamp) {
            if (!chart || !candleData.length) return;
            try {
                const targetTime = parseInt(timestamp);
                let nearestIndex = findBarIndex(targetTime);
                
                const barsVisible = 100;
                const from = Math.max(0, nearestIndex - barsVisible / 2);
                const to = Math.min(candleData.length, nearestIndex + barsVisible / 2);
                
                chart.timeScale().setVisibleLogicalRange({ from, to });
                setStatus('Jumped to time', '#26a69a');
            } catch (e) {
                console.error('Goto error:', e);
            }
        };

        // ========== THEME SUPPORT ==========

        window.applyTheme = function(colors) {
            if (!chart) return;
            
            chartColors = colors;
            
            chart.applyOptions({
                layout: {
                    background: { type: 'solid', color: colors.bg },
                    textColor: colors.text,
                },
                grid: {
                    vertLines: { color: colors.grid },
                    horzLines: { color: colors.grid },
                },
                rightPriceScale: {
                    borderColor: colors.border,
                },
                timeScale: {
                    borderColor: colors.border,
                },
            });
            
            // Update candle series colors
            candleSeries.applyOptions({
                upColor: colors.up,
                downColor: colors.down,
                borderUpColor: colors.up,
                borderDownColor: colors.down,
                wickUpColor: colors.up,
                wickDownColor: colors.down,
            });
            
            console.log('Theme applied:', colors);
        };

        // ========== DRAWING TOOLS ==========

        window.enableHorizontalLine = function() {
            deselectDrawing();
            hideMeasureTool();
            drawingMode = 'horizontal';
            drawingPoints = [];
            setStatus('Click on chart to place horizontal line', '#FF9800');
        };

        window.enableVerticalLine = function() {
            deselectDrawing();
            hideMeasureTool();
            drawingMode = 'vertical';
            drawingPoints = [];
            setStatus('Click on chart to place vertical line', '#FF9800');
        };

        window.enableTrendLine = function() {
            deselectDrawing();
            hideMeasureTool();
            drawingMode = 'trendline';
            drawingPoints = [];
            setStatus('Click first point for trend line', '#FF9800');
        };

        window.enableMeasureTool = function() {
            deselectDrawing();
            hideMeasureTool();
            drawingMode = 'measure';
            setStatus('Click start point to measure', '#2196F3');
        };

        window.cancelDrawing = function() {
            drawingMode = null;
            drawingPoints = [];
            deselectDrawing();
            hideMeasureTool();
            document.getElementById('drawing-preview').style.display = 'none';
            setStatus('Cancelled', '#737780');
        };

        window.clearDrawings = function() {
            drawings.forEach(d => {
                if (d.line) candleSeries.removePriceLine(d.line);
                else if (d.series) chart.removeSeries(d.series);
            });
            drawings = [];
            drawingMode = null;
            drawingPoints = [];
            selectedDrawingId = null;
            hideMeasureTool();
            document.getElementById('drawing-preview').style.display = 'none';
            setStatus('Drawings cleared', '#26a69a');
        };

        // ========== INDICATORS ==========

        // Global map for pane reuse: { type: paneIndex }
        let paneMap = {};

        window.addIndicator = function(id, type, params, color) {
            console.log('[DEBUG] addIndicator called:', id, type, params, color);
            if (!candleData.length) return;
            
            // Multi-line indicator suffixes
            const suffixes = ['_upper', '_middle', '_lower', '_macd', '_signal', '_hist', 
                            '_k', '_d', '_tenkan', '_kijun', '_senkouA', '_senkouB'];
            
            // 1. Determine which pane to use
            let targetPane = null;
            
            // Check if this specific ID already exists in a pane
            if (indicators[id] && indicators[id].pane !== undefined) {
                targetPane = indicators[id].pane;
            } else {
                for (const s of suffixes) {
                    if (indicators[id + s] && indicators[id + s].pane !== undefined) {
                        targetPane = indicators[id + s].pane;
                        break;
                    }
                }
            }
            
            // If not found by ID, check if this TYPE of indicator already has a reserved pane
            if (targetPane === null && paneMap[type] !== undefined) {
                // For CUSTOM indicators, only reuse the pane if we are in 'New Pane' mode
                // If it's a 'Main Chart' overlay (pane 0), don't force it into the reserved pane.
                if (type !== 'CUSTOM' || params.pane === 'New Pane') {
                    targetPane = paneMap[type];
                    console.log('[DEBUG] Reusing pane', targetPane, 'for type', type);
                }
            }
            
            // If still no pane, assign a new one for this type
            if (targetPane === null) {
                // Determine if this indicator needs its own pane (oscillators like MACD, RSI, etc.)
                const needsPane = ['MACD', 'RSI', 'ATR', 'ADX', 'STOCH', 'STOCHRSI', 'MOMENTUM', 'OBV'].includes(type) || 
                                (type === 'CUSTOM' && params.pane === 'New Pane');
                if (needsPane) {
                    targetPane = nextPaneIndex++;
                    paneMap[type] = targetPane; // Reserve this pane for this type
                    console.log('[DEBUG] Assigned new pane', targetPane, 'for type', type);
                } else {
                    targetPane = 0; // Overlay on main chart (SMA, EMA, BB, etc.)
                }
            }

            // 2. Remove existing instance with this ID (to replace it)
            // Special cleanup for generic/custom multi-line keys
            Object.keys(indicators).forEach(function(key) {
                if (key === id || key.startsWith(id + '_')) {
                    chart.removeSeries(indicators[key].series);
                    delete indicators[key];
                }
            });
            
            let data = [];
            let name = '';
            
            // Selection logic for type specific blocks
            const existingPane = targetPane; 
            const myPane = targetPane;
            
            if (type === 'SMA') {
                const period = params.period || 20;
                name = 'SMA(' + period + ')';
                
                for (let i = period - 1; i < candleData.length; i++) {
                    let sum = 0;
                    for (let j = 0; j < period; j++) sum += candleData[i-j].close;
                    data.push({ time: candleData[i].time, value: sum / period });
                }
                
            } else if (type === 'EMA') {
                const period = params.period || 20;
                name = 'EMA(' + period + ')';
                
                const mult = 2 / (period + 1);
                let sum = 0;
                for (let i = 0; i < period && i < candleData.length; i++) sum += candleData[i].close;
                let ema = sum / Math.min(period, candleData.length);
                
                if (candleData.length >= period) {
                    data.push({ time: candleData[period - 1].time, value: ema });
                    for (let i = period; i < candleData.length; i++) {
                        ema = (candleData[i].close - ema) * mult + ema;
                        data.push({ time: candleData[i].time, value: ema });
                    }
                }
                
            } else if (type === 'VWMA') {
                const period = params.period || 20;
                name = 'VWMA(' + period + ')';
                
                for (let i = period - 1; i < candleData.length; i++) {
                    let sumPV = 0;
                    let sumV = 0;
                    for (let j = 0; j < period; j++) {
                        const vol = candleData[i-j].volume || 1;
                        sumPV += candleData[i-j].close * vol;
                        sumV += vol;
                    }
                    data.push({ time: candleData[i].time, value: sumPV / sumV });
                }
                
            } else if (type === 'WMA') {
                const period = params.period || 20;
                name = 'WMA(' + period + ')';
                
                for (let i = period - 1; i < candleData.length; i++) {
                    let sum = 0;
                    let weightSum = 0;
                    for (let j = 0; j < period; j++) {
                        const weight = period - j;
                        sum += candleData[i-j].close * weight;
                        weightSum += weight;
                    }
                    data.push({ time: candleData[i].time, value: sum / weightSum });
                }
                
            } else if (type === 'DEMA') {
                const period = params.period || 20;
                name = 'DEMA(' + period + ')';
                
                // Calculate EMA
                const mult = 2 / (period + 1);
                let emaSum = 0;
                for (let i = 0; i < period; i++) emaSum += candleData[i].close;
                let ema1 = emaSum / period;
                const emaData = [];
                emaData.push({ time: candleData[period - 1].time, value: ema1 });
                for (let i = period; i < candleData.length; i++) {
                    ema1 = (candleData[i].close - ema1) * mult + ema1;
                    emaData.push({ time: candleData[i].time, value: ema1 });
                }
                
                // Calculate EMA of EMA
                let ema2 = emaData[0].value;
                const ema2Data = [];
                ema2Data.push({ time: emaData[period - 1].time, value: ema2 });
                for (let i = period; i < emaData.length; i++) {
                    ema2 = (emaData[i].value - ema2) * mult + ema2;
                    ema2Data.push({ time: emaData[i].time, value: ema2 });
                }
                
                // DEMA = 2 * EMA1 - EMA2(EMA1)
                for (let i = 0; i < ema2Data.length; i++) {
                    const ema1Idx = i + period - 1;
                    if (ema1Idx < emaData.length) {
                        const dema = 2 * emaData[ema1Idx].value - ema2Data[i].value;
                        data.push({ time: ema2Data[i].time, value: dema });
                    }
                }
                
            } else if (type === 'TEMA') {
                const period = params.period || 20;
                name = 'TEMA(' + period + ')';
                
                const mult = 2 / (period + 1);
                let sum = 0;
                for (let i = 0; i < period; i++) sum += candleData[i].close;
                let ema1 = sum / period;
                
                // EMA1
                const ema1Data = [];
                ema1Data.push({ time: candleData[period - 1].time, value: ema1 });
                for (let i = period; i < candleData.length; i++) {
                    ema1 = (candleData[i].close - ema1) * mult + ema1;
                    ema1Data.push({ time: candleData[i].time, value: ema1 });
                }
                
                // EMA2 (EMA of EMA1)
                let ema2 = ema1Data[0].value;
                const ema2Data = [];
                ema2Data.push({ time: ema1Data[period - 1].time, value: ema2 });
                for (let i = period; i < ema1Data.length; i++) {
                    ema2 = (ema1Data[i].value - ema2) * mult + ema2;
                    ema2Data.push({ time: ema1Data[i].time, value: ema2 });
                }
                
                // EMA3 (EMA of EMA2)
                let ema3 = ema2Data[0].value;
                const ema3Data = [];
                ema3Data.push({ time: ema2Data[period - 1].time, value: ema3 });
                for (let i = period; i < ema2Data.length; i++) {
                    ema3 = (ema2Data[i].value - ema3) * mult + ema3;
                    ema3Data.push({ time: ema2Data[i].time, value: ema3 });
                }
                
                // TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
                for (let i = 0; i < ema3Data.length; i++) {
                    const idx1 = i + period - 1;
                    const idx2 = i + period - 1;
                    if (idx1 < ema1Data.length && idx2 < ema2Data.length) {
                        const tema = 3 * ema1Data[idx1].value - 3 * ema2Data[idx2].value + ema3Data[i].value;
                        data.push({ time: ema3Data[i].time, value: tema });
                    }
                }
                
            } else if (type === 'PSAR') {
                const step = params.step || 0.02;
                const max = params.max || 0.2;
                name = 'PSAR';
                
                let sar = candleData[0].close;
                let ep = candleData[0].high;
                let up = true;
                let af = step;
                
                const sarData = [];
                sarData.push({ time: candleData[0].time, value: sar });
                
                for (let i = 1; i < candleData.length; i++) {
                    const high = candleData[i].high;
                    const low = candleData[i].low;
                    
                    if (up) {
                        sar = sar + af * (ep - sar);
                        if (high > ep) {
                            ep = high;
                            af = Math.min(af + step, max);
                        }
                        if (low < sar) {
                            up = false;
                            sar = ep;
                            ep = low;
                            af = step;
                        }
                    } else {
                        sar = sar + af * (ep - sar);
                        if (low < ep) {
                            ep = low;
                            af = Math.min(af + step, max);
                        }
                        if (high > sar) {
                            up = true;
                            sar = ep;
                            ep = high;
                            af = step;
                        }
                    }
                    
                    sarData.push({ time: candleData[i].time, value: sar });
                }
                
                data = sarData;
            } else if (type === 'KC') {
                const period = params.period || 20;
                const atrPeriod = params.atrPeriod || 10;
                const multiplier = params.multiplier || 2;
                name = 'Keltner(' + period + ')';
                
                // Calculate EMA for middle band
                const mult = 2 / (period + 1);
                let sum = 0;
                for (let i = 0; i < period; i++) sum += candleData[i].close;
                let ema = sum / period;
                const middleData = [];
                middleData.push({ time: candleData[period - 1].time, value: ema });
                for (let i = period; i < candleData.length; i++) {
                    ema = (candleData[i].close - ema) * mult + ema;
                    middleData.push({ time: candleData[i].time, value: ema });
                }
                
                // Calculate ATR
                const trData = [];
                for (let i = 0; i < candleData.length; i++) {
                    const high = candleData[i].high;
                    const low = candleData[i].low;
                    const prevClose = i > 0 ? candleData[i-1].close : candleData[i].close;
                    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
                    trData.push(tr);
                }
                
                let atrSum = 0;
                for (let i = 0; i < atrPeriod; i++) atrSum += trData[i];
                let atr = atrSum / atrPeriod;
                
                const upperData = [];
                const lowerData = [];
                
                for (let i = atrPeriod - 1; i < middleData.length; i++) {
                    const idx = i + period - atrPeriod;
                    if (idx < candleData.length) {
                        const atrVal = trData[i];
                        upperData.push({ time: middleData[i].time, value: middleData[i].value + multiplier * atrVal });
                        lowerData.push({ time: middleData[i].time, value: middleData[i].value - multiplier * atrVal });
                    }
                }
                
                // Create series for Keltner Channel
                const upperSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#E91E63',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                upperSeries.setData(upperData);
                
                const middleSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#E91E63',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                middleSeries.setData(middleData);
                
                const lowerSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#E91E63',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                lowerSeries.setData(lowerData);
                
                indicators[id + '_upper'] = { series: upperSeries, color: color || '#E91E63', name: 'KC Upper' };
                indicators[id + '_middle'] = { series: middleSeries, color: color || '#E91E63', name: 'KC Middle' };
                indicators[id + '_lower'] = { series: lowerSeries, color: color || '#E91E63', name: 'KC Lower' };
                return;
            } else if (type === 'DC') {
                const period = params.period || 20;
                name = 'Donchian(' + period + ')';
                
                const upperData = [];
                const lowerData = [];
                
                for (let i = period - 1; i < candleData.length; i++) {
                    let highest = candleData[i].high;
                    let lowest = candleData[i].low;
                    
                    for (let j = 0; j < period; j++) {
                        if (candleData[i - j].high > highest) highest = candleData[i - j].high;
                        if (candleData[i - j].low < lowest) lowest = candleData[i - j].low;
                    }
                    
                    upperData.push({ time: candleData[i].time, value: highest });
                    lowerData.push({ time: candleData[i].time, value: lowest });
                }
                
                // Create series for Donchian Channel
                const upperSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#3F51B5',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                upperSeries.setData(upperData);
                
                const lowerSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#3F51B5',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                lowerSeries.setData(lowerData);
                
                indicators[id + '_upper'] = { series: upperSeries, color: color || '#3F51B5', name: 'DC Upper' };
                indicators[id + '_lower'] = { series: lowerSeries, color: color || '#3F51B5', name: 'DC Lower' };
                return;
            } else if (type === 'MOMENTUM') {
                const period = params.period || 10;
                name = 'Momentum(' + period + ')';
                
                for (let i = period; i < candleData.length; i++) {
                    const momentum = candleData[i].close - candleData[i - period].close;
                    data.push({ time: candleData[i].time, value: momentum });
                }
            } else if (type === 'STOCHRSI') {
                const rsiPeriod = params.rsiPeriod || 14;
                const kPeriod = params.kPeriod || 3;
                const dPeriod = params.dPeriod || 3;
                name = 'StochRSI';
                
                // First calculate RSI
                let gains = 0, losses = 0;
                const rsiValues = [];
                
                const changes = [];
                for (let i = 1; i < candleData.length; i++) {
                    const change = candleData[i].close - candleData[i-1].close;
                    changes.push(change);
                }
                
                for (let i = 0; i < rsiPeriod; i++) {
                    if (changes[i] >= 0) gains += changes[i];
                    else losses -= changes[i];
                }
                
                let avgGain = gains / rsiPeriod;
                let avgLoss = losses / rsiPeriod;
                
                let rs = avgGain / (avgLoss || 0.0001);
                let rsi = 100 - (100 / (1 + rs));
                rsiValues.push(rsi);
                
                for (let i = rsiPeriod; i < changes.length; i++) {
                    const change = changes[i];
                    const gain = change >= 0 ? change : 0;
                    const loss = change < 0 ? -change : 0;
                    
                    avgGain = (avgGain * (rsiPeriod - 1) + gain) / rsiPeriod;
                    avgLoss = (avgLoss * (rsiPeriod - 1) + loss) / rsiPeriod;
                    
                    rs = avgGain / (avgLoss || 0.0001);
                    rsi = 100 - (100 / (1 + rs));
                    rsiValues.push(rsi);
                }
                
                // Calculate Stochastic of RSI
                const kData = [];
                const dData = [];
                
                for (let i = kPeriod - 1; i < rsiValues.length; i++) {
                    let highest = rsiValues[i];
                    let lowest = rsiValues[i];
                    
                    for (let j = 0; j < kPeriod; j++) {
                        if (rsiValues[i - j] > highest) highest = rsiValues[i - j];
                        if (rsiValues[i - j] < lowest) lowest = rsiValues[i - j];
                    }
                    
                    const range = highest - lowest || 1;
                    const k = ((rsiValues[i] - lowest) / range) * 100;
                    kData.push({ time: candleData[i + rsiPeriod].time, value: k });
                }
                
                // Smooth %K and calculate %D
                let kSum = 0;
                for (let i = 0; i < dPeriod; i++) kSum += kData[i].value;
                let smoothK = kSum / dPeriod;
                dData.push({ time: kData[dPeriod - 1].time, value: smoothK });
                
                for (let i = dPeriod; i < kData.length; i++) {
                    smoothK = (smoothK * (dPeriod - 1) + kData[i].value) / dPeriod;
                    dData.push({ time: kData[i].time, value: smoothK });
                }
                
                // Create panel for Stochastic RSI
// Create series for Stochastic RSI
                const kSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#2196F3',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                kSeries.setData(kData);
                
                const dSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#FF9800',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                dSeries.setData(dData);
                
                indicators[id + '_k'] = { series: kSeries, color: '#2196F3', name: '%K', pane: myPane };
                indicators[id + '_d'] = { series: dSeries, color: '#FF9800', name: '%D', pane: myPane };
                return;
            } else if (type === 'AD') {
                const myPane = nextPaneIndex++;
                name = 'A/D Line';
                
                let ad = 0;
                for (let i = 0; i < candleData.length; i++) {
                    const high = candleData[i].high;
                    const low = candleData[i].low;
                    const close = candleData[i].close;
                    
                    const mfm = ((close - low) - (high - close)) / (high - low || 1);
                    ad += mfm * (candleData[i].volume || 0);
                    
                    data.push({ time: candleData[i].time, value: ad });
                }
                
                // Create panel for A/D
const series = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#8BC34A',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                series.setData(data);
                
                indicators[id] = { series: series, color: color || '#8BC34A', name: name, pane: myPane };
                return;
            } else if (type === 'ICHIMOKU') {
                const tenkan = params.tenkan || 9;
                const kijun = params.kijun || 26;
                const senkou = params.senkou || 52;
                name = 'Ichimoku';
                
                // Calculate Tenkan-sen (Conversion Line)
                const tenkanData = [];
                for (let i = tenkan - 1; i < candleData.length; i++) {
                    let highest = candleData[i].high;
                    let lowest = candleData[i].low;
                    
                    for (let j = 0; j < tenkan; j++) {
                        if (candleData[i - j].high > highest) highest = candleData[i - j].high;
                        if (candleData[i - j].low < lowest) lowest = candleData[i - j].low;
                    }
                    
                    tenkanData.push({ time: candleData[i].time, value: (highest + lowest) / 2 });
                }
                
                // Calculate Kijun-sen (Base Line)
                const kijunData = [];
                for (let i = kijun - 1; i < candleData.length; i++) {
                    let highest = candleData[i].high;
                    let lowest = candleData[i].low;
                    
                    for (let j = 0; j < kijun; j++) {
                        if (candleData[i - j].high > highest) highest = candleData[i - j].high;
                        if (candleData[i - j].low < lowest) lowest = candleData[i - j].low;
                    }
                    
                    kijunData.push({ time: candleData[i].time, value: (highest + lowest) / 2 });
                }
                
                // Calculate Senkou Span A (Leading Span A)
                const senkouAData = [];
                for (let i = kijun - 1; i < tenkanData.length; i++) {
                    const idx = i - kijun + tenkan - 1;
                    if (idx >= 0 && idx < kijunData.length) {
                        senkouAData.push({
                            time: tenkanData[i].time,
                            value: (tenkanData[i].value + kijunData[idx].value) / 2
                        });
                    }
                }
                
                // Calculate Senkou Span B (Leading Span B)
                const senkouBData = [];
                for (let i = senkou - 1; i < candleData.length; i++) {
                    let highest = candleData[i].high;
                    let lowest = candleData[i].low;
                    
                    for (let j = 0; j < senkou; j++) {
                        if (candleData[i - j].high > highest) highest = candleData[i - j].high;
                        if (candleData[i - j].low < lowest) lowest = candleData[i - j].low;
                    }
                    
                    senkouBData.push({ time: candleData[i].time, value: (highest + lowest) / 2 });
                }
                
                // Create series for Ichimoku Cloud
                const tenkanSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#2196F3',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                tenkanSeries.setData(tenkanData);
                
                const kijunSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#FF5722',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                kijunSeries.setData(kijunData);
                
                const senkouASeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#4CAF50',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                senkouASeries.setData(senkouAData);
                
                const senkouBSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#F44336',
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                senkouBSeries.setData(senkouBData);
                
                indicators[id + '_tenkan'] = { series: tenkanSeries, color: '#2196F3', name: 'Tenkan' };
                indicators[id + '_kijun'] = { series: kijunSeries, color: '#FF5722', name: 'Kijun' };
                indicators[id + '_senkouA'] = { series: senkouASeries, color: '#4CAF50', name: 'Senkou A' };
                indicators[id + '_senkouB'] = { series: senkouBSeries, color: '#F44336', name: 'Senkou B' };
                return;
            } else if (type === 'HEIKINASHI') {
                name = 'Heikin-Ashi';
                
                // Calculate Heikin-Ashi candles
                const haData = [];
                let haOpen = candleData[0].close;
                let haClose = (candleData[0].open + candleData[0].high + candleData[0].close) / 4;
                haOpen = (haClose + haOpen) / 2;
                
                haData.push({
                    time: candleData[0].time,
                    open: haOpen,
                    high: Math.max(candleData[0].high, Math.max(haOpen, haClose)),
                    low: Math.min(candleData[0].low, Math.min(haOpen, haClose)),
                    close: haClose
                });
                
                for (let i = 1; i < candleData.length; i++) {
                    const open = haOpen;
                    const close = (candleData[i].open + candleData[i].high + candleData[i].low + candleData[i].close) / 4;
                    haOpen = (open + close) / 2;
                    
                    const high = Math.max(candleData[i].high, Math.max(haOpen, close));
                    const low = Math.min(candleData[i].low, Math.min(haOpen, close));
                    
                    haData.push({
                        time: candleData[i].time,
                        open: haOpen,
                        high: high,
                        low: low,
                        close: close
                    });
                }
                
                // Update the candle series with Heikin-Ashi data
                if (typeof candleSeries !== 'undefined') {
                    candleSeries.setData(haData);
                    indicators[id] = { series: candleSeries, color: '#FF9800', name: 'Heikin-Ashi' };
                }
                return;
            } else if (type === 'BB') {
                const period = params.period || 20;
                const stdDev = params.stdDev || 2;
                name = 'BB(' + period + ')';
                
                const upper = [], middle = [], lower = [];
                
                for (let i = period - 1; i < candleData.length; i++) {
                    let sum = 0;
                    for (let j = 0; j < period; j++) sum += candleData[i-j].close;
                    const sma = sum / period;
                    
                    let variance = 0;
                    for (let j = 0; j < period; j++) {
                        variance += Math.pow(candleData[i-j].close - sma, 2);
                    }
                    const std = Math.sqrt(variance / period);
                    
                    middle.push({ time: candleData[i].time, value: sma });
                    upper.push({ time: candleData[i].time, value: sma + stdDev * std });
                    lower.push({ time: candleData[i].time, value: sma - stdDev * std });
                }
                
                const opts = { color: color, lineWidth: 1, crosshairMarkerVisible: false, priceLineVisible: false, lastValueVisible: false };
                const uLine = chart.addSeries(LightweightCharts.LineSeries, opts);
                const mLine = chart.addSeries(LightweightCharts.LineSeries, Object.assign({}, opts, { lineStyle: 2 }));
                const lLine = chart.addSeries(LightweightCharts.LineSeries, opts);
                
                uLine.setData(upper);
                mLine.setData(middle);
                lLine.setData(lower);
                
                indicators[id + '_upper'] = { series: uLine, color: color, name: 'BB Upper(' + period + ')' };
                indicators[id + '_middle'] = { series: mLine, color: color, name: 'BB Middle(' + period + ')' };
                indicators[id + '_lower'] = { series: lLine, color: color, name: 'BB Lower(' + period + ')' };
                return;
                
            } else if (type === 'VWAP') {
                name = 'VWAP';
                let cumVol = 0, cumVP = 0;
                for (let i = 0; i < candleData.length; i++) {
                    const tp = (candleData[i].high + candleData[i].low + candleData[i].close) / 3;
                    const vol = candleData[i].volume || 1;
                    cumVol += vol;
                    cumVP += tp * vol;
                    data.push({ time: candleData[i].time, value: cumVP / cumVol });
                }
            } else if (type === 'RSI') {
                const period = params.period || 14;
                name = 'RSI(' + period + ')';
                
                // RSI calculation
                // Calculate RSI
                let gains = 0;
                let losses = 0;
                
                for (let i = 0; i < period; i++) {
                    const diff = candleData[i+1].close - candleData[i].close;
                    if (diff > 0) gains += diff;
                    else losses -= diff;
                }
                
                let avgGain = gains / period;
                let avgLoss = losses / period;
                
                const rsiData = [];
                rsiData.push({ time: candleData[period].time, value: 100 - (100 / (1 + avgGain / avgLoss)) });
                
                for (let i = period + 1; i < candleData.length; i++) {
                    const diff = candleData[i].close - candleData[i-1].close;
                    
                    if (diff > 0) {
                        avgGain = (avgGain * (period - 1) + diff) / period;
                        avgLoss = (avgLoss * (period - 1)) / period;
                    } else {
                        avgGain = (avgGain * (period - 1)) / period;
                        avgLoss = (avgLoss * (period - 1) - diff) / period;
                    }
                    
                    const rs = avgGain / avgLoss;
                    rsiData.push({ time: candleData[i].time, value: 100 - (100 / (1 + rs)) });
                }
                
                data = rsiData;
                
                // Create panel for RSI
const series = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#673AB7',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                series.setData(data);
                
                indicators[id] = { series: series, color: color || '#673AB7', name: name, pane: myPane };
                return;
            } else if (type === 'MACD') {
                const fastPeriod = params.fastPeriod || 12;
                const slowPeriod = params.slowPeriod || 26;
                const signalPeriod = params.signalPeriod || 9;
                name = 'MACD';
                
                // Calculate Fast EMA
                const fastMult = 2 / (fastPeriod + 1);
                let fastSum = 0;
                for (let i = 0; i < fastPeriod; i++) fastSum += candleData[i].close;
                let fastEMA = fastSum / fastPeriod;
                const fastData = [];
                fastData.push({ time: candleData[fastPeriod - 1].time, value: fastEMA });
                
                for (let i = fastPeriod; i < candleData.length; i++) {
                    fastEMA = (candleData[i].close - fastEMA) * fastMult + fastEMA;
                    fastData.push({ time: candleData[i].time, value: fastEMA });
                }
                
                // Calculate Slow EMA
                const slowMult = 2 / (slowPeriod + 1);
                let slowSum = 0;
                for (let i = 0; i < slowPeriod; i++) slowSum += candleData[i].close;
                let slowEMA = slowSum / slowPeriod;
                const slowData = [];
                slowData.push({ time: candleData[slowPeriod - 1].time, value: slowEMA });
                
                for (let i = slowPeriod; i < candleData.length; i++) {
                    slowEMA = (candleData[i].close - slowEMA) * slowMult + slowEMA;
                    slowData.push({ time: candleData[i].time, value: slowEMA });
                }
                
                // Calculate Histogram - align properly by time
                let macdLine = [];
                let signalData = [];
                let histogramData = [];

                if (params.talib_data) {
                    console.log('[DEBUG] Using TA-Lib data for MACD');
                    macdLine = params.talib_data.macd;
                    signalData = params.talib_data.signal;
                    histogramData = params.talib_data.hist;
                    
                    // Add colors to histogram
                    histogramData.forEach(p => {
                        p.color = p.value >= 0 ? '#26a69a' : '#ef5350';
                    });
                } else {
                    // FALLBACK: Manual calculation if Python/TA-Lib fails
                    const manualMacdLine = [];
                    const offset = slowPeriod - fastPeriod;
                    for (let i = offset; i < fastData.length; i++) {
                        const slowIdx = i - offset;
                        if (slowIdx < slowData.length) {
                            manualMacdLine.push({
                                time: fastData[i].time,
                                value: fastData[i].value - slowData[slowIdx].value
                            });
                        }
                    }
                    macdLine = manualMacdLine;

                    // Manual Signal line logic
                    const manualSignalData = [];
                    if (macdLine.length >= signalPeriod) {
                        const signalMult = 2 / (signalPeriod + 1);
                        let signalSum = 0;
                        for (let i = 0; i < signalPeriod && i < macdLine.length; i++) signalSum += macdLine[i].value;
                        let signalEMA = signalSum / Math.min(signalPeriod, macdLine.length);
                        manualSignalData.push({ time: macdLine[signalPeriod - 1].time, value: signalEMA });
                        for (let i = signalPeriod; i < macdLine.length; i++) {
                            signalEMA = (macdLine[i].value - signalEMA) * signalMult + signalEMA;
                            manualSignalData.push({ time: macdLine[i].time, value: signalEMA });
                        }
                    }
                    signalData = manualSignalData;

                    // Manual Histogram logic
                    const manualHistData = [];
                    for (let i = 0; i < macdLine.length; i++) {
                        const macdTime = macdLine[i].time;
                        let signalVal = 0;
                        for (let j = 0; j < signalData.length; j++) {
                            if (signalData[j].time === macdTime) {
                                signalVal = signalData[j].value;
                                break;
                            }
                        }
                        const histVal = macdLine[i].value - signalVal;
                        manualHistData.push({
                            time: macdTime,
                            value: histVal,
                            color: histVal >= 0 ? '#26a69a' : '#ef5350'
                        });
                    }
                    histogramData = manualHistData;
                }
                
                // Create panel for MACD (reduce volume panel space first)
                const panelHeight = 0.25; // 25% height
                const panelTop = 0.55; // Start at 55% from top
// Create series for MACD
                const macdSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#2196F3',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                macdSeries.setData(macdLine);
                
                const signalSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#FF9800',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                signalSeries.setData(signalData);
                
                const histSeries = chart.addSeries(LightweightCharts.HistogramSeries, {
                    color: '#8884d8',
                    priceFormat: { type: 'volume' }

                }, myPane);
                histSeries.setData(histogramData);
                
                indicators[id + '_macd'] = { series: macdSeries, color: '#2196F3', name: 'MACD', pane: myPane };
                indicators[id + '_signal'] = { series: signalSeries, color: '#FF9800', name: 'Signal', pane: myPane };
                indicators[id + '_hist'] = { series: histSeries, color: '#8884d8', name: 'Histogram', pane: myPane };
                return;
            } else if (type === 'CUSTOM') {
                const pane = (params.pane === 'Main Chart') ? 0 : targetPane;
                if (params.talib_data) {
                    const keys = Object.keys(params.talib_data);
                    keys.forEach((key, index) => {
                        const lineData = params.talib_data[key];
                        // Distribute colors for multi-line if default color is used
                        const colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0'];
                        const lineColor = (keys.length > 1) ? (colors[index % colors.length]) : (color || '#2196F3');
                        
                        const series = chart.addSeries(LightweightCharts.LineSeries, {
                            color: lineColor,
                            lineWidth: 2,
                            title: key === 'main' ? 'Custom' : key
                        }, pane);
                        series.setData(lineData);
                        indicators[id + '_' + key] = { series: series, color: lineColor, name: key, pane: pane };
                    });
                }
                
                // Add Markers if any (Signaling BUY/SELL)
                if (params.markers && params.markers.length > 0) {
                    // Sorting markers by time is often required by LW Charts
                    params.markers.sort((a, b) => a.time - b.time);
                    candleSeries.setMarkers(params.markers);
                    console.log('[DEBUG] Applied', params.markers.length, 'markers');
                }

                if (params.error) {
                    console.log('Custom script technical note:', params.error);
                }
                return;
            } else if (type === 'STOCH') {
                const kPeriod = params.kPeriod || 14;
                const dPeriod = params.dPeriod || 3;
                const smooth = params.smooth || 3;
                name = 'Stochastic';
                
                // Calculate Stochastic
                const kData = [];
                const dData = [];
                
                for (let i = kPeriod - 1; i < candleData.length; i++) {
                    let highest = candleData[i].high;
                    let lowest = candleData[i].low;
                    
                    // Find highest high and lowest low over kPeriod
                    for (let j = 0; j < kPeriod; j++) {
                        if (candleData[i - j].high > highest) highest = candleData[i - j].high;
                        if (candleData[i - j].low < lowest) lowest = candleData[i - j].low;
                    }
                    
                    const range = highest - lowest || 1;
                    const close = candleData[i].close;
                    const k = ((close - lowest) / range) * 100;
                    kData.push({ time: candleData[i].time, value: k });
                }
                
                // Calculate %D as SMA of %K
                for (let i = smooth - 1; i < kData.length; i++) {
                    let dSum = 0;
                    for (let j = 0; j < smooth; j++) dSum += kData[i - j].value;
                    dData.push({ time: kData[i].time, value: dSum / smooth });
                }
                
                // Create panel for Stochastic
// Create series for Stochastic
                const kSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#2196F3',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                kSeries.setData(kData);
                
                const dSeries = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#FF9800',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                dSeries.setData(dData);
                
                indicators[id + '_k'] = { series: kSeries, color: '#2196F3', name: '%K', pane: myPane };
                indicators[id + '_d'] = { series: dSeries, color: '#FF9800', name: '%D', pane: myPane };
                return;
            } else if (type === 'SUPERTREND') {
                const period = params.period || 10;
                const multiplier = params.multiplier || 2.0;
                name = 'Supertrend';
                
                // Calculate ATR first (inline for Supertrend)
                const atrData = [];
                for (let i = 0; i < candleData.length; i++) {
                    const high = candleData[i].high;
                    const low = candleData[i].low;
                    const prevClose = i > 0 ? candleData[i-1].close : candleData[i].close;
                    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
                    atrData.push({ time: candleData[i].time, value: tr });
                }
                
                // Smooth ATR with the same length as candleData
                const smoothAtr = [];
                let atrSum = 0;
                for (let i = 0; i < period; i++) {
                    atrSum += atrData[i].value;
                }
                let atr = atrSum / period;
                smoothAtr.push({ time: candleData[period - 1].time, value: atr });
                
                for (let i = period; i < candleData.length; i++) {
                    atr = (atr * (period - 1) + atrData[i].value) / period;
                    smoothAtr.push({ time: candleData[i].time, value: atr });
                }
                
                // Calculate Supertrend using the correct algorithm
                const supertrendData = [];
                const finalUpper = [];
                const finalLower = [];
                const trend = [];
                
                // Start from period - 1 to align with smoothAtr
                for (let i = period - 1; i < candleData.length; i++) {
                    const hl2 = (candleData[i].high + candleData[i].low) / 2;
                    const atrVal = smoothAtr[i - (period - 1)].value;
                    
                    const upperBasic = hl2 + (multiplier * atrVal);
                    const lowerBasic = hl2 - (multiplier * atrVal);
                    
                    const arrIdx = i - (period - 1); // Index in smoothAtr/finalUpper/finalLower/trend arrays
                    
                    if (arrIdx === 0) {
                        // First value
                        finalUpper.push({ time: candleData[i].time, value: upperBasic });
                        finalLower.push({ time: candleData[i].time, value: lowerBasic });
                        trend.push({ time: candleData[i].time, value: 1 });
                        supertrendData.push({ time: candleData[i].time, value: lowerBasic, color: '#26a69a' });
                    } else {
                        const prevFinalUpper = finalUpper[arrIdx - 1].value;
                        const prevFinalLower = finalLower[arrIdx - 1].value;
                        const prevClose = candleData[i - 1].close;
                        
                        // Update final upper band
                        if (prevClose <= prevFinalUpper) {
                            finalUpper.push({ time: candleData[i].time, value: Math.min(upperBasic, prevFinalUpper) });
                        } else {
                            finalUpper.push({ time: candleData[i].time, value: upperBasic });
                        }
                        
                        // Update final lower band
                        if (prevClose >= prevFinalLower) {
                            finalLower.push({ time: candleData[i].time, value: Math.max(lowerBasic, prevFinalLower) });
                        } else {
                            finalLower.push({ time: candleData[i].time, value: lowerBasic });
                        }
                        
                        // Determine trend and supertrend value
                        let currentTrend = trend[arrIdx - 1].value;
                        const close = candleData[i].close;
                        
                        if (close > finalUpper[arrIdx - 1].value) {
                            currentTrend = 1;
                            supertrendData.push({ time: candleData[i].time, value: finalLower[arrIdx].value, color: '#26a69a' });
                        } else if (close < finalLower[arrIdx - 1].value) {
                            currentTrend = -1;
                            supertrendData.push({ time: candleData[i].time, value: finalUpper[arrIdx].value, color: '#ef5350' });
                        } else {
                            supertrendData.push({ 
                                time: candleData[i].time, 
                                value: currentTrend === 1 ? finalLower[arrIdx].value : finalUpper[arrIdx].value,
                                color: currentTrend === 1 ? '#26a69a' : '#ef5350'
                            });
                        }
                        
                        trend.push({ time: candleData[i].time, value: currentTrend });
                    }
                }
                
                const series = chart.addSeries(LightweightCharts.LineSeries, {
                    color: '#FF5722',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                series.setData(supertrendData);
                indicators[id] = { series: series, color: '#FF5722', name: 'Supertrend' };
                return;
            } else if (type === 'ATR') {
                const period = params.period || 14;
                name = 'ATR(' + period + ')';
                
                // Calculate True Range
                const trData = [];
                for (let i = 0; i < candleData.length; i++) {
                    const high = candleData[i].high;
                    const low = candleData[i].low;
                    const prevClose = i > 0 ? candleData[i-1].close : candleData[i].close;
                    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
                    trData.push({ time: candleData[i].time, value: tr });
                }
                
                // Calculate ATR using Wilder's smoothing
                let trSum = 0;
                for (let i = 0; i < period; i++) trSum += trData[i].value;
                let atr = trSum / period;
                
                data.push({ time: candleData[period - 1].time, value: atr });
                
                for (let i = period; i < trData.length; i++) {
                    atr = (atr * (period - 1) + trData[i].value) / period;
                    data.push({ time: candleData[i].time, value: atr });
                }
                
                // Create panel for ATR
const series = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#607D8B',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                series.setData(data);
                
                indicators[id] = { series: series, color: color || '#607D8B', name: name, pane: myPane };
                return;
            } else if (type === 'OBV') {
                const myPane = (existingPane !== null) ? existingPane : nextPaneIndex++;
                name = 'OBV';
                
                let obv = 0;
                for (let i = 0; i < candleData.length; i++) {
                    const volume = candleData[i].volume || 0;
                    const close = candleData[i].close;
                    const prevClose = i > 0 ? candleData[i-1].close : close;
                    
                    if (close > prevClose) {
                        obv += volume;
                    } else if (close < prevClose) {
                        obv -= volume;
                    }
                    
                    data.push({ time: candleData[i].time, value: obv });
                }
                
                // Create panel for OBV
const series = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#4CAF50',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false

                }, myPane);
                series.setData(data);
                
                indicators[id] = { series: series, color: color || '#4CAF50', name: name, pane: myPane };
                return;
            } else if (type === 'ADX') {
                const period = params.period || 14;
                name = 'ADX(' + period + ')';
                
                // Calculate +DI, -DI, and ADX
                let plusDM = [], minusDM = [];
                
                for (let i = 0; i < candleData.length; i++) {
                    const high = candleData[i].high;
                    const low = candleData[i].low;
                    const prevHigh = i > 0 ? candleData[i-1].high : high;
                    const prevLow = i > 0 ? candleData[i-1].low : low;
                    
                    const up = high - prevHigh;
                    const down = prevLow - low;
                    
                    if (up > down && up > 0) plusDM.push({ time: candleData[i].time, value: up });
                    else plusDM.push({ time: candleData[i].time, value: 0 });
                    
                    if (down > up && down > 0) minusDM.push({ time: candleData[i].time, value: down });
                    else minusDM.push({ time: candleData[i].time, value: 0 });
                }
                
                // Calculate ATR
                const trData = [];
                for (let i = 0; i < candleData.length; i++) {
                    const high = candleData[i].high;
                    const low = candleData[i].low;
                    const prevClose = i > 0 ? candleData[i-1].close : candleData[i].close;
                    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
                    trData.push(tr);
                }
                
                // Smooth +DI, -DI, and ATR
                let plusDMSum = 0, minusDMSum = 0, trSum = 0;
                
                for (let i = 0; i < period; i++) {
                    plusDMSum += plusDM[i].value;
                    minusDMSum += minusDM[i].value;
                    trSum += trData[i];
                }
                
                let smoothPlusDM = plusDMSum;
                let smoothMinusDM = minusDMSum;
                let smoothTR = trSum;
                
                const adxData = [];
                
                for (let i = period; i < candleData.length; i++) {
                    smoothPlusDM = (smoothPlusDM * (period - 1) + plusDM[i].value) / period;
                    smoothMinusDM = (smoothMinusDM * (period - 1) + minusDM[i].value) / period;
                    smoothTR = (smoothTR * (period - 1) + trData[i]) / period;
                    
                    const plusDI = (smoothPlusDM / smoothTR) * 100;
                    const minusDI = (smoothMinusDM / smoothTR) * 100;
                    
                    const dx = (Math.abs(plusDI - minusDI) / (plusDI + minusDI)) * 100;
                    
                    adxData.push({ time: candleData[i].time, value: dx });
                }
                
                // Smooth ADX
                let adxSum = 0;
                for (let i = 0; i < period; i++) adxSum += adxData[i].value;
                let adx = adxSum / period;
                
                data.push({ time: adxData[period - 1].time, value: adx });
                
                for (let i = period; i < adxData.length; i++) {
                    adx = (adx * (period - 1) + adxData[i].value) / period;
                    data.push({ time: adxData[i].time, value: adx });
                }
                
                // Create panel for ADX
if (data.length > 0) {
                    const series = chart.addSeries(LightweightCharts.LineSeries, {
                        color: color || '#795548',
                        lineWidth: 2,
                        priceLineVisible: false,
                        lastValueVisible: false,
                        crosshairMarkerVisible: false

                    }, myPane);
                    series.setData(data);
                    indicators[id] = { series: series, color: color || '#795548', name: name, pane: myPane };
                }
                return;
            }
            
            // Common series creation for simple overlay indicators
            // (SMA, WMA, DEMA, TEMA, PSAR, EMA, Momentum, StochRSI, AD, Heikin-Ashi, VWAP, etc.)
            if (data.length > 0) {
                console.log('[DEBUG] Creating overlay series for', name, 'with', data.length, 'points');
                const series = chart.addSeries(LightweightCharts.LineSeries, {
                    color: color || '#2196F3',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false
                });
                series.setData(data);
                indicators[id] = { series: series, color: color || '#2196F3', name: name };
            }
        };

        window.removeIndicator = function(id) {
            console.log('[DEBUG] removeIndicator called for:', id);
            
            // Remove any series associated with this ID (including multi-line suffixes)
            Object.keys(indicators).forEach(function(key) {
                if (key === id || key.startsWith(id + '_')) {
                    console.log('[DEBUG] Removing series:', key);
                    chart.removeSeries(indicators[key].series);
                    delete indicators[key];
                }
            });
        };

        window.updateIndicatorColor = function(id, color) {
            // Multi-line indicator suffixes
            const suffixes = ['_upper', '_middle', '_lower', '_macd', '_signal', '_hist', 
                            '_k', '_d', '_tenkan', '_kijun', '_senkouA', '_senkouB'];
            
            // Update single-line indicator
            if (indicators[id]) {
                indicators[id].series.applyOptions({ color: color });
                indicators[id].color = color;
            }
            
            // Update multi-line indicators
            suffixes.forEach(function(suffix) {
                if (indicators[id + suffix]) {
                    indicators[id + suffix].series.applyOptions({ color: color });
                    indicators[id + suffix].color = color;
                }
            });
        };

        window.clearIndicators = function() {
            Object.keys(indicators).forEach(function(id) {
                chart.removeSeries(indicators[id].series);
            });
            indicators = {};
            document.getElementById('indicator-values').innerHTML = '';
        };

        // Initialize
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initChart);
        } else {
            initChart();
        }
    </script>
</body>
</html>
        """
        )
        self.setHtml(html)


# ============================================================
# UI COMPONENTS
# ============================================================


class IndicatorListItem(QWidget):
    """Custom widget for indicator list item with color and delete"""

    color_changed = pyqtSignal(str, str)
    delete_clicked = pyqtSignal(str)

    def __init__(self, ind_id, ind_name, color):
        super().__init__()
        self.ind_id = ind_id
        self.current_color = color

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(18, 18)
        self.color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color}; 
                border-radius: 9px; 
                border: 2px solid #363c4e;
            }}
            QPushButton:hover {{
                border: 2px solid #d1d4dc;
            }}
        """)
        self.color_btn.clicked.connect(self.change_color)
        self.color_btn.setToolTip("Change color")
        layout.addWidget(self.color_btn)

        name_label = QLabel(ind_name)
        name_label.setStyleSheet("color: #d1d4dc; font-size: 12px;")
        layout.addWidget(name_label, stretch=1)

        del_btn = QPushButton("×")
        del_btn.setFixedSize(20, 20)
        del_btn.setStyleSheet("""
            QPushButton { 
                background-color: transparent; 
                color: #737780; 
                border: none; 
                font-size: 16px; 
                font-weight: bold; 
            }
            QPushButton:hover { 
                background-color: #ef5350; 
                color: white; 
                border-radius: 10px; 
            }
        """)
        del_btn.clicked.connect(lambda: self.delete_clicked.emit(self.ind_id))
        del_btn.setToolTip("Remove indicator")
        layout.addWidget(del_btn)

    def change_color(self):
        color = QColorDialog.getColor(
            QColor(self.current_color), self, "Choose Indicator Color"
        )
        if color.isValid():
            self.current_color = color.name()
            self.color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.current_color}; 
                    border-radius: 9px; 
                    border: 2px solid #363c4e;
                }}
                QPushButton:hover {{
                    border: 2px solid #d1d4dc;
                }}
            """)
            self.color_changed.emit(self.ind_id, self.current_color)


# ============================================================
# INDICATOR SELECTION DIALOG (TradingView-style)
# ============================================================


class IndicatorItem(QWidget):
    """Individual indicator item in the selection list"""

    clicked = pyqtSignal(str, str, str)  # type, name, default_color

    def __init__(self, ind_type, name, description, icon, default_color, category):
        super().__init__()
        self.ind_type = ind_type
        self.name = name
        self.default_color = default_color
        self.category = category

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border-radius: 6px;
                padding: 8px;
            }
            QWidget:hover {
                background-color: #2a2e39;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px;")
        icon_label.setFixedWidth(30)
        layout.addWidget(icon_label)

        # Text container
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        name_label = QLabel(name)
        name_label.setStyleSheet("color: #d1d4dc; font-size: 13px; font-weight: bold;")
        text_layout.addWidget(name_label)

        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #737780; font-size: 11px;")
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, stretch=1)

        # Color indicator
        color_dot = QLabel()
        color_dot.setFixedSize(12, 12)
        color_dot.setStyleSheet(
            f"background-color: {default_color}; border-radius: 6px;"
        )
        layout.addWidget(color_dot)

    def mousePressEvent(self, event):
        self.clicked.emit(self.ind_type, self.name, self.default_color)


class IndicatorSelectionDialog(QDialog):
    """TradingView-style indicator selection dialog"""

    indicator_selected = pyqtSignal(str, str, str)  # type, name, color

    # Define available indicators
    INDICATORS = [
        # Moving Averages
        {
            "type": "SMA",
            "name": "Simple Moving Average",
            "desc": "Average price over N periods",
            "icon": "📈",
            "color": "#2196F3",
            "category": "Moving Averages",
        },
        {
            "type": "EMA",
            "name": "Exponential Moving Average",
            "desc": "Weighted average favoring recent prices",
            "icon": "📈",
            "color": "#FF9800",
            "category": "Moving Averages",
        },
        {
            "type": "WMA",
            "name": "Weighted Moving Average",
            "desc": "Linearly weighted average",
            "icon": "📈",
            "color": "#00BCD4",
            "category": "Moving Averages",
        },
        {
            "type": "DEMA",
            "name": "Double EMA",
            "desc": "Smoother EMA with less lag",
            "icon": "📈",
            "color": "#8BC34A",
            "category": "Moving Averages",
        },
        {
            "type": "TEMA",
            "name": "Triple EMA",
            "desc": "Even less lag than DEMA",
            "icon": "📈",
            "color": "#CDDC39",
            "category": "Moving Averages",
        },
        # Bands & Channels
        {
            "type": "BB",
            "name": "Bollinger Bands",
            "desc": "Volatility bands around SMA",
            "icon": "📊",
            "color": "#9C27B0",
            "category": "Bands & Channels",
        },
        {
            "type": "KC",
            "name": "Keltner Channel",
            "desc": "ATR-based volatility channel",
            "icon": "📊",
            "color": "#E91E63",
            "category": "Bands & Channels",
        },
        {
            "type": "DC",
            "name": "Donchian Channel",
            "desc": "Highest high & lowest low channel",
            "icon": "📊",
            "color": "#3F51B5",
            "category": "Bands & Channels",
        },
        # Volume
        {
            "type": "VWAP",
            "name": "VWAP",
            "desc": "Volume Weighted Average Price",
            "icon": "📉",
            "color": "#4CAF50",
            "category": "Volume",
        },
        {
            "type": "VWMA",
            "name": "Volume Weighted MA",
            "desc": "Moving average weighted by volume",
            "icon": "📉",
            "color": "#009688",
            "category": "Volume",
        },
        # Trend
        {
            "type": "SUPERTREND",
            "name": "Supertrend",
            "desc": "Trend following indicator",
            "icon": "🔄",
            "color": "#FF5722",
            "category": "Trend",
        },
        {
            "type": "PSAR",
            "name": "Parabolic SAR",
            "desc": "Stop and reverse indicator",
            "icon": "⚫",
            "color": "#795548",
            "category": "Trend",
        },
        # Momentum (for future - shown but may need implementation)
        {
            "type": "RSI",
            "name": "RSI",
            "desc": "Relative Strength Index (0-100)",
            "icon": "📶",
            "color": "#673AB7",
            "category": "Momentum",
        },
        {
            "type": "MACD",
            "name": "MACD",
            "desc": "Moving Average Convergence Divergence",
            "icon": "📶",
            "color": "#F44336",
            "category": "Momentum",
        },
        {
            "type": "STOCH",
            "name": "Stochastic",
            "desc": "Stochastic Oscillator",
            "icon": "📶",
            "color": "#03A9F4",
            "category": "Momentum",
        },
        # Additional Momentum Indicators
        {
            "type": "STOCHRSI",
            "name": "Stochastic RSI",
            "desc": "Stochastic oscillator of RSI values",
            "icon": "📶",
            "color": "#00BCD4",
            "category": "Momentum",
        },
        {
            "type": "MOMENTUM",
            "name": "Momentum",
            "desc": "Rate of change of price",
            "icon": "📶",
            "color": "#FF4081",
            "category": "Momentum",
        },
        # Volatility Indicators
        {
            "type": "ATR",
            "name": "ATR",
            "desc": "Average True Range for volatility",
            "icon": "📈",
            "color": "#607D8B",
            "category": "Volatility",
        },
        {
            "type": "ADX",
            "name": "ADX",
            "desc": "Average Directional Index",
            "icon": "📊",
            "color": "#795548",
            "category": "Trend",
        },
        # Volume Indicators
        {
            "type": "OBV",
            "name": "OBV",
            "desc": "On Balance Volume",
            "icon": "📉",
            "color": "#4CAF50",
            "category": "Volume",
        },
        {
            "type": "AD",
            "name": "A/D Line",
            "desc": "Accumulation/Distribution Line",
            "icon": "📉",
            "color": "#8BC34A",
            "category": "Volume",
        },
        # Advanced Indicators
        {
            "type": "ICHIMOKU",
            "name": "Ichimoku Cloud",
            "desc": "Multiple moving averages for trend analysis",
            "icon": "☁️",
            "color": "#2196F3",
            "category": "Trend",
        },
        {
            "type": "HEIKINASHI",
            "name": "Heikin-Ashi",
            "desc": "Modified candlestick chart",
            "icon": "🕯️",
            "color": "#FF9800",
            "category": "Chart Type",
        },
        {
            "type": "CUSTOM",
            "name": "Custom Python Script",
            "desc": "Write your own Python/TA-Lib indicator",
            "icon": "⚡",
            "color": "#FFD700",
            "category": "Advanced",
        },
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Add Indicator")
        self.setModal(True)
        self.setMinimumSize(500, 600)
        self.resize(550, 650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with search
        header = QFrame()
        header.setStyleSheet("background-color: #1e222d; padding: 15px;")
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(10)

        title = QLabel("📊 Indicators")
        title.setStyleSheet("color: #d1d4dc; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search indicators...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #2a2e39;
                border: 1px solid #363c4e;
                border-radius: 6px;
                padding: 10px 15px;
                color: #d1d4dc;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
        """)
        self.search_input.textChanged.connect(self.filter_indicators)
        header_layout.addWidget(self.search_input)

        layout.addWidget(header)

        # Scrollable indicator list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #131722;
            }
            QScrollBar:vertical {
                background: #1e222d;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #363c4e;
                border-radius: 4px;
            }
        """)

        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(10, 10, 10, 10)
        self.list_layout.setSpacing(0)

        self.indicator_widgets = []
        self.populate_indicators()

        scroll.setWidget(self.list_container)
        layout.addWidget(scroll)

        # Footer
        footer = QFrame()
        footer.setStyleSheet("background-color: #1e222d; padding: 10px 15px;")
        footer_layout = QHBoxLayout(footer)

        hint = QLabel("💡 Click an indicator to add it to the chart")
        hint.setStyleSheet("color: #737780; font-size: 11px;")
        footer_layout.addWidget(hint)

        footer_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #d1d4dc;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #363c4e;
            }
        """)
        close_btn.clicked.connect(self.close)
        footer_layout.addWidget(close_btn)

        layout.addWidget(footer)

        self.setStyleSheet("""
            QDialog {
                background-color: #131722;
            }
        """)

    def populate_indicators(self):
        """Populate the indicator list grouped by category"""
        categories = {}
        for ind in self.INDICATORS:
            cat = ind["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ind)

        for category, indicators in categories.items():
            # Category header
            cat_label = QLabel(category)
            cat_label.setStyleSheet("""
                color: #2196F3; 
                font-size: 12px; 
                font-weight: bold; 
                padding: 15px 12px 8px 12px;
                background-color: transparent;
            """)
            self.list_layout.addWidget(cat_label)

            # Indicators in category
            for ind in indicators:
                item = IndicatorItem(
                    ind["type"],
                    ind["name"],
                    ind["desc"],
                    ind["icon"],
                    ind["color"],
                    ind["category"],
                )
                item.clicked.connect(self.on_indicator_clicked)
                self.list_layout.addWidget(item)
                self.indicator_widgets.append(
                    (item, ind["name"].lower(), ind["type"].lower(), category.lower())
                )

        self.list_layout.addStretch()

    def filter_indicators(self, text):
        """Filter indicators based on search text"""
        search = text.lower().strip()

        for item, name, ind_type, category in self.indicator_widgets:
            if (
                search == ""
                or search in name
                or search in ind_type
                or search in category
            ):
                item.setVisible(True)
            else:
                item.setVisible(False)

    def on_indicator_clicked(self, ind_type, name, color):
        self.indicator_selected.emit(ind_type, name, color)
        self.close()


class IndicatorPanel(QGroupBox):
    """Enhanced indicator panel with TradingView-style list and adjustable parameters"""

    indicator_added = pyqtSignal(
        str, str, dict, str
    )  # id, type, params, color  # type, params, color
    indicator_removed = pyqtSignal(str)
    indicator_color_changed = pyqtSignal(str, str)

    def __init__(self):
        super().__init__("📊 Indicators")
        logging.info(f"[DEBUG] IndicatorPanel __init__ called. ID: {id(self)}")
        self.indicator_counter = 0
        self.active_indicators = {}  # Store active indicator info

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Add indicator button
        add_btn = QPushButton("+ Add Indicator")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        add_btn.clicked.connect(self.show_indicator_dialog)
        layout.addWidget(add_btn)

        # Active indicators label
        active_label = QLabel("Active Indicators:")
        active_label.setStyleSheet("color: #737780; font-size: 11px; margin-top: 10px;")
        layout.addWidget(active_label)

        # Active indicators list
        self.indicator_list = QListWidget()
        self.indicator_list.setMinimumHeight(100)
        self.indicator_list.setMaximumHeight(200)
        self.indicator_list.setStyleSheet("""
            QListWidget {
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                border: none;
                padding: 2px;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #2a2e39;
            }
        """)
        layout.addWidget(self.indicator_list)

        # Clear all button
        clear_btn = QPushButton("🗑️ Clear All Indicators")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #ef5350;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #ef5350;
                color: white;
                border-color: #ef5350;
            }
        """)
        clear_btn.clicked.connect(lambda: self.indicator_removed.emit("all"))
        layout.addWidget(clear_btn)

    def show_indicator_dialog(self):
        dialog = IndicatorSelectionDialog(self)
        dialog.indicator_selected.connect(self.on_indicator_selected)
        dialog.exec()

    def on_indicator_selected(self, ind_type, name, color):
        """When user selects an indicator from the dialog, show parameter configuration"""
        print(
            f">>> IndicatorPanel.on_indicator_selected called: type={ind_type} (Instance ID: {id(self)}) <<<"
        )
        try:
            # Check if indicator of this type already exists
            existing_id = None
            logging.info(
                f"[DEBUG] Checking existing for {ind_type}. Active count: {len(self.active_indicators)}"
            )

            for i_id, i_data in self.active_indicators.items():
                curr_type = str(i_data.get("type"))
                if curr_type == ind_type:
                    existing_id = i_id
                    logging.info(f"[DEBUG] Found existing: {existing_id}")
                    break

            # CUSTOM HANDLER
            if ind_type == "CUSTOM":
                initial_code = None
                initial_pane = "New Pane"
                if existing_id:
                    initial_code = (
                        self.active_indicators[existing_id]
                        .get("params", {})
                        .get("code")
                    )
                    initial_pane = (
                        self.active_indicators[existing_id]
                        .get("params", {})
                        .get("pane")
                    )

                param_dialog = CustomPythonIndicatorDialog(
                    self, initial_code, initial_pane
                )
            else:
                # Standard parameter dialog
                param_dialog = IndicatorParameterDialog(ind_type, self)

            if param_dialog.exec() == QDialog.DialogCode.Accepted:
                params = (
                    param_dialog.get_params()
                    if ind_type != "CUSTOM"
                    else param_dialog.get_data()
                )

                if existing_id:
                    ind_id = existing_id
                    print(f">>> Updating existing {ind_id} <<<")

                    display_name = name
                    if ind_type == "CUSTOM":
                        display_name = "Python Script"
                    elif params:
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        display_name = f"{name} ({param_str})"

                    self.active_indicators[ind_id]["params"] = params
                    self.active_indicators[ind_id]["name"] = display_name
                    self.active_indicators[ind_id]["color"] = color

                    # Update list item text
                    for i in range(self.indicator_list.count()):
                        item = self.indicator_list.item(i)
                        widget = self.indicator_list.itemWidget(item)
                        if widget and widget.ind_id == ind_id:
                            new_widget = IndicatorListItem(ind_id, display_name, color)
                            new_widget.color_changed.connect(
                                self.indicator_color_changed.emit
                            )
                            new_widget.delete_clicked.connect(
                                self.remove_indicator_from_list
                            )
                            self.indicator_list.setItemWidget(item, new_widget)
                            break
                else:
                    # Create new
                    self.indicator_counter += 1
                    ind_id = f"{ind_type.lower()}_{self.indicator_counter}"
                    print(f">>> Creating new {ind_id} <<<")

                    display_name = name
                    if ind_type == "CUSTOM":
                        display_name = "Python Script"
                    elif params:
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        display_name = f"{name} ({param_str})"

                    self.active_indicators[ind_id] = {
                        "type": ind_type,
                        "name": display_name,
                        "color": color,
                        "params": params,
                    }
                    logging.info(
                        f"[DEBUG] Added new indicator: {ind_id} to active_indicators"
                    )
                    self.add_indicator_to_list(ind_id, display_name, color)

                # Emit signal with ID
                self.indicator_added.emit(ind_id, ind_type, params, color)
        except Exception as e:
            import traceback

            logging.error(f"Error in on_indicator_selected: {e}")
            traceback.print_exc()

    def add_indicator_to_list(self, ind_id, name, color):
        item = QListWidgetItem()
        widget = IndicatorListItem(ind_id, name, color)
        widget.color_changed.connect(self.indicator_color_changed.emit)
        widget.delete_clicked.connect(self.remove_indicator_from_list)
        item.setSizeHint(QSize(0, 36))
        self.indicator_list.addItem(item)
        self.indicator_list.setItemWidget(item, widget)

    def remove_indicator_from_list(self, ind_id):
        logging.info(f"[DEBUG] Removing indicator {ind_id} from list")
        for i in range(self.indicator_list.count()):
            item = self.indicator_list.item(i)
            widget = self.indicator_list.itemWidget(item)
            if widget and widget.ind_id == ind_id:
                self.indicator_list.takeItem(i)
                if ind_id in self.active_indicators:
                    del self.active_indicators[ind_id]
                self.indicator_removed.emit(ind_id)
                break

    def clear_all_indicators(self):
        logging.info(f"[DEBUG] Clearing ALL indicators (Instance ID: {id(self)})")
        self.indicator_list.clear()
        self.active_indicators.clear()


class ChartToolbar(QFrame):
    """Enhanced toolbar with drawing tools, chart controls, and theme toggle"""

    magnet_toggled = pyqtSignal(bool)
    theme_toggled = pyqtSignal(str)
    trend_line_clicked = pyqtSignal()
    horizontal_line_clicked = pyqtSignal()
    vertical_line_clicked = pyqtSignal()
    measure_tool_clicked = pyqtSignal()
    clear_drawings_clicked = pyqtSignal()
    scale_mode_changed = pyqtSignal(str)
    grid_toggled = pyqtSignal(bool)
    fit_content_clicked = pyqtSignal()
    about_clicked = pyqtSignal()
    layout_changed = pyqtSignal(int)
    nextstep_clicked = pyqtSignal()

    # Style constants
    BUTTON_NORMAL = """
        QPushButton { 
            background-color: #2a2e39; 
            border: 1px solid #363c4e; 
            border-radius: 4px; 
            padding: 6px 10px; 
            color: #d1d4dc; 
            font-weight: bold; 
        }
        QPushButton:hover { 
            background-color: #363c4e; 
            border-color: #2196F3;
        }
    """

    BUTTON_ACTIVE = """
        QPushButton { 
            background-color: #2196F3; 
            border: 1px solid #1976D2; 
            border-radius: 4px; 
            padding: 6px 10px; 
            color: white; 
            font-weight: bold; 
        }
        QPushButton:hover { 
            background-color: #1976D2; 
        }
    """

    TOGGLE_NORMAL = """
        QPushButton { 
            background-color: #2a2e39; 
            border: 1px solid #363c4e; 
            border-radius: 4px; 
            padding: 6px 10px; 
            color: #d1d4dc; 
            font-weight: bold; 
        }
        QPushButton:hover { 
            background-color: #363c4e; 
            border-color: #2196F3;
        }
    """

    TOGGLE_CHECKED = """
        QPushButton { 
            background-color: #4CAF50; 
            border: 1px solid #388E3C; 
            border-radius: 4px; 
            padding: 6px 10px; 
            color: white; 
            font-weight: bold; 
        }
        QPushButton:hover { 
            background-color: #388E3C; 
        }
    """

    def __init__(self):
        super().__init__()
        self.setMaximumHeight(50)
        self.setStyleSheet(
            "background-color: #1e222d; border-bottom: 1px solid #363c4e;"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)

        # Theme toggle button
        self.theme_btn = QPushButton("🌙")
        self.theme_btn.setToolTip("Toggle Theme (Dark/Light)")
        self.theme_btn.setFixedWidth(35)
        self.theme_btn.setStyleSheet(self.TOGGLE_NORMAL)
        self.theme_btn.clicked.connect(self._on_theme_toggled)
        layout.addWidget(self.theme_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #363c4e;")
        layout.addWidget(sep)

        draw_label = QLabel("🎨 Draw:")
        draw_label.setStyleSheet(
            "color: #737780; font-weight: bold; margin-right: 5px;"
        )
        layout.addWidget(draw_label)

        # Drawing buttons - store references
        self.h_line_btn = QPushButton("━ H-Line (H)")
        self.h_line_btn.setStyleSheet(self.BUTTON_NORMAL)
        self.h_line_btn.clicked.connect(self._on_h_line_clicked)
        layout.addWidget(self.h_line_btn)

        self.v_line_btn = QPushButton("┃ V-Line (V)")
        self.v_line_btn.setStyleSheet(self.BUTTON_NORMAL)
        self.v_line_btn.clicked.connect(self._on_v_line_clicked)
        layout.addWidget(self.v_line_btn)

        self.trend_btn = QPushButton("📈 Trend (T)")
        self.trend_btn.setStyleSheet(self.BUTTON_NORMAL)
        self.trend_btn.clicked.connect(self._on_trend_clicked)
        layout.addWidget(self.trend_btn)

        self.measure_btn = QPushButton("📏 Measure (M)")
        self.measure_btn.setStyleSheet(self.BUTTON_NORMAL)
        self.measure_btn.clicked.connect(self._on_measure_clicked)
        layout.addWidget(self.measure_btn)

        self.clear_draw_btn = QPushButton("🗑️")
        self.clear_draw_btn.setToolTip("Clear Drawings (Ctrl+D)")
        self.clear_draw_btn.setFixedWidth(35)
        self.clear_draw_btn.setStyleSheet(self.BUTTON_NORMAL)
        self.clear_draw_btn.clicked.connect(self.clear_drawings_clicked.emit)
        layout.addWidget(self.clear_draw_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #363c4e;")
        layout.addWidget(sep)

        chart_label = QLabel("⚙️ Chart:")
        chart_label.setStyleSheet(
            "color: #737780; font-weight: bold; margin-right: 5px;"
        )
        layout.addWidget(chart_label)

        # Magnet button with proper checked styling
        self.magnet_btn = QPushButton("🧲 Magnet")
        self.magnet_btn.setCheckable(True)
        self.magnet_btn.setStyleSheet(self.TOGGLE_NORMAL)
        self.magnet_btn.toggled.connect(self._on_magnet_toggled)
        layout.addWidget(self.magnet_btn)

        # Grid button with proper checked styling
        self.grid_btn = QPushButton("# Grid")
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(True)
        self.grid_btn.setStyleSheet(self.TOGGLE_CHECKED)
        self.grid_btn.toggled.connect(self._on_grid_toggled)
        layout.addWidget(self.grid_btn)

        self.fit_btn = QPushButton("⊡ Fit (F)")
        self.fit_btn.setStyleSheet(self.BUTTON_NORMAL)
        self.fit_btn.clicked.connect(self.fit_content_clicked.emit)
        layout.addWidget(self.fit_btn)

        layout.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Normal", "Logarithmic"])
        self.scale_combo.currentTextChanged.connect(
            lambda t: self.scale_mode_changed.emit(t.lower())
        )
        layout.addWidget(self.scale_combo)

        # Layout selector for multiple charts
        layout.addWidget(QLabel("Layout:"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(
            [
                "1 Chart",
                "2 Charts (Vertical)",
                "2 Charts (Horizontal)",
                "3 Charts",
                "4 Charts",
            ]
        )
        self.layout_combo.setCurrentIndex(0)
        self.layout_combo.setToolTip("Number of charts to display")
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        layout.addWidget(self.layout_combo)

        layout.addStretch()

        # NextSTEP button for custom data upload
        nextstep_btn = QPushButton("📁 NextSTEP")
        nextstep_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                border: 1px solid #7B1FA2;
                border-radius: 4px;
                padding: 6px 12px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #AB47BC;
                border-color: #2196F3;
            }
        """)
        nextstep_btn.setToolTip("Upload custom data (Excel/CSV)")
        nextstep_btn.clicked.connect(self.nextstep_clicked.emit)
        layout.addWidget(nextstep_btn)

        # About button
        about_btn = QPushButton("ℹ️ About")
        about_btn.setStyleSheet("""
            QPushButton {
                background-color: #363c4e;
                border: 1px solid #434a5c;
                border-radius: 4px;
                padding: 6px 12px;
                color: #d1d4dc;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #434a5c;
                border-color: #2196F3;
            }
        """)
        about_btn.clicked.connect(self.about_clicked.emit)
        layout.addWidget(about_btn)

        # Store all drawing buttons for easy management
        self._drawing_buttons = {
            "horizontal": self.h_line_btn,
            "vertical": self.v_line_btn,
            "trendline": self.trend_btn,
            "measure": self.measure_btn,
        }

    def _on_theme_toggled(self):
        """Handle theme toggle button click"""
        self.theme_toggled.emit("toggle")

    def set_theme_icon(self, is_dark):
        """Update the theme button icon based on current theme"""
        if is_dark:
            self.theme_btn.setText("🌙")
            self.theme_btn.setToolTip("Switch to Light Theme")
        else:
            self.theme_btn.setText("☀️")
            self.theme_btn.setToolTip("Switch to Dark Theme")

    def _on_magnet_toggled(self, checked):
        if checked:
            self.magnet_btn.setStyleSheet(self.TOGGLE_CHECKED)
        else:
            self.magnet_btn.setStyleSheet(self.TOGGLE_NORMAL)
        self.magnet_toggled.emit(checked)

    def _on_grid_toggled(self, checked):
        if checked:
            self.grid_btn.setStyleSheet(self.TOGGLE_CHECKED)
        else:
            self.grid_btn.setStyleSheet(self.TOGGLE_NORMAL)
        self.grid_toggled.emit(checked)

    def _on_layout_changed(self, index):
        """Handle layout selection change"""
        self.layout_changed.emit(index)

    def _on_h_line_clicked(self):
        self.horizontal_line_clicked.emit()

    def _on_v_line_clicked(self):
        self.vertical_line_clicked.emit()

    def _on_trend_clicked(self):
        self.trend_line_clicked.emit()

    def _on_measure_clicked(self):
        self.measure_tool_clicked.emit()

    def set_active_tool(self, mode):
        """Highlight the active drawing tool button"""
        for tool_name, btn in self._drawing_buttons.items():
            if tool_name == mode:
                btn.setStyleSheet(self.BUTTON_ACTIVE)
            else:
                btn.setStyleSheet(self.BUTTON_NORMAL)

    def clear_active_tool(self):
        """Reset all drawing buttons to normal state"""
        for btn in self._drawing_buttons.values():
            btn.setStyleSheet(self.BUTTON_NORMAL)


# ============================================================
# CLICKABLE CHART WRAPPER
# ============================================================


class ClickableChartWrapper(QWidget):
    """Wrapper widget that catches mouse clicks on charts"""

    chart_clicked = pyqtSignal(int)

    def __init__(self, chart_index, parent=None):
        super().__init__(parent)
        self._chart_index = chart_index
        self._parent = parent

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QWidget { margin: 0; padding: 0; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._chart = None

    def add_chart(self, chart):
        """Add a chart to this wrapper"""
        self._chart = chart
        self.layout().addWidget(chart)

    def mousePressEvent(self, event):
        """Catch mouse clicks and emit signal, then pass to chart"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.chart_clicked.emit(self._chart_index)
            # Give focus to the chart for zoom/keyboard to work
            if self._chart:
                self._chart.setFocus()
        # Accept and pass the event to the chart for native handling (zoom, scroll, etc.)
        event.accept()
        super().mousePressEvent(event)

    def toggle_magnet(self, enabled):
        """Toggle magnet mode on the underlying chart"""
        if self._chart:
            self._chart.toggle_magnet(enabled)

    def apply_theme(self, theme):
        """Apply theme to the underlying chart"""
        if self._chart:
            self._chart.apply_theme(theme)


# ============================================================
# MAIN WINDOW
# ============================================================


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📈 Advanced TradingView-Style Chart - Pro Edition - Krish")
        self.setGeometry(50, 50, 1600, 1000)

        self.expiry_dates_list = []
        self.thread_manager = ThreadManager()
        self._shortcuts = []

        # Initialize theme manager
        self.theme_manager = ThemeManager("dark")

        self.setup_dark_theme()
        self.setup_ui()
        self.setup_shortcuts()

    def setup_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #131722; color: #d1d4dc; }
            QGroupBox { 
                border: 1px solid #2a2e39; border-radius: 6px; margin-top: 14px; 
                padding-top: 12px; font-weight: bold; background-color: #1e222d;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #2196F3; font-size: 13px; }
            QComboBox, QLineEdit, QSpinBox, QDateEdit { 
                background-color: #2a2e39; border: 1px solid #363c4e; border-radius: 4px; 
                padding: 6px; color: #d1d4dc; min-height: 26px; 
            }
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDateEdit:hover { border-color: #2196F3; }
            QComboBox QAbstractItemView { background-color: #2a2e39; color: #d1d4dc; selection-background-color: #2196F3; border: 1px solid #363c4e; }
            QPushButton { 
                background-color: #2196F3; border: none; border-radius: 4px; 
                padding: 8px 14px; color: white; font-weight: bold; min-height: 28px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #1565C0; }
            QPushButton:disabled { background-color: #2a2e39; color: #737780; }
            QLabel { color: #737780; }
            QTextEdit { 
                background-color: #0d1117; border: 1px solid #2a2e39; color: #8b949e; 
                font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; border-radius: 4px;
            }
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:vertical { background: #1e222d; width: 10px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #363c4e; border-radius: 5px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #434a5c; }
            QSplitter::handle { background-color: #2a2e39; }
        """)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # === LEFT PANEL ===
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Data Settings Group
        data_group = QGroupBox("📊 Data Settings")
        data_layout = QVBoxLayout(data_group)

        # Chart Selector (for multi-chart mode)
        row = QHBoxLayout()
        row.addWidget(QLabel("Chart:"))
        self.chart_selector = QComboBox()
        self.chart_selector.addItems(["Chart 1", "Chart 2", "Chart 3", "Chart 4"])
        self.chart_selector.setCurrentIndex(0)
        self.chart_selector.currentIndexChanged.connect(self.on_chart_selector_changed)
        row.addWidget(self.chart_selector, 1)
        data_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Type:"))
        self.data_type = QComboBox()
        self.data_type.addItems(["Options", "Spot"])
        self.data_type.currentTextChanged.connect(self.on_data_type_change)
        row.addWidget(self.data_type, 1)
        data_layout.addLayout(row)

        # Exchange selection
        row = QHBoxLayout()
        row.addWidget(QLabel("Exchange:"))
        self.exchange = QComboBox()
        self.exchange.addItems(["NSE", "BSE", "NYSE", "MCX"])
        self.exchange.currentTextChanged.connect(self.on_exchange_change)
        row.addWidget(self.exchange, 1)
        data_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Symbol:"))
        self.symbol = QComboBox()
        self.symbol.addItems(["NIFTY", "BANKNIFTY", "SENSEX", "SPY"])
        self.symbol.currentTextChanged.connect(self.on_symbol_change)
        row.addWidget(self.symbol, 1)
        data_layout.addLayout(row)

        self.expiry_type_row = QWidget()
        row = QHBoxLayout(self.expiry_type_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Expiry Type:"))
        self.expiry_type = QComboBox()
        self.expiry_type.addItems(["Weekly", "Monthly"])
        self.expiry_type.currentTextChanged.connect(self.on_expiry_type_change)
        row.addWidget(self.expiry_type, 1)
        data_layout.addWidget(self.expiry_type_row)

        # Interval - Changed to allow custom input with preset options
        row = QHBoxLayout()
        row.addWidget(QLabel("Interval:"))
        self.interval = QComboBox()
        # Preset timeframes
        self.timeframe_presets = [
            "1min",
            "2min",
            "3min",
            "5min",
            "10min",
            "15min",
            "30min",
            "45min",
            "60min",
            "1h",
            "2h",
            "3h",
            "4h",
            "5h",
            "6h",
            "12h",
            "18h",
            "1D",
            "2D",
            "3D",
            "1W",
            "1M",
        ]
        self.interval.addItems(self.timeframe_presets)
        self.interval.setCurrentText("15min")
        self.interval.setEditable(True)
        self.interval.lineEdit().setPlaceholderText("e.g., 2min, 45min, 2H, 5H, 1D")
        self.interval.lineEdit().setMaxLength(10)
        row.addWidget(self.interval, 1)
        data_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Start:"))
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.start_date.dateChanged.connect(self.on_start_date_changed)
        row.addWidget(self.start_date, 1)
        data_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("End:"))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        row.addWidget(self.end_date, 1)
        data_layout.addLayout(row)

        # Expiry - Changed to QComboBox
        self.expiry_row = QWidget()
        row = QHBoxLayout(self.expiry_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Expiry:"))
        self.expiry_combo = QComboBox()
        self.expiry_combo.setStyleSheet("QComboBox { min-width: 120px; }")
        self.expiry_combo.currentTextChanged.connect(self.on_expiry_selected)
        row.addWidget(self.expiry_combo, 1)
        data_layout.addWidget(self.expiry_row)

        self.strike_row = QWidget()
        row = QHBoxLayout(self.strike_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Strike:"))
        self.strike_price = QLineEdit()
        self.strike_price.setPlaceholderText("e.g., 19500")
        row.addWidget(self.strike_price, 1)
        data_layout.addWidget(self.strike_row)

        self.side_row = QWidget()
        row = QHBoxLayout(self.side_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Side:"))
        self.option_type = QComboBox()
        self.option_type.addItems(["CE", "PE"])
        row.addWidget(self.option_type, 1)
        data_layout.addWidget(self.side_row)

        self.fetch_btn = QPushButton("📊 Fetch & Plot Data")
        self.fetch_btn.setStyleSheet("QPushButton { font-size: 13px; padding: 10px; }")
        self.fetch_btn.clicked.connect(self.fetch_and_plot)
        data_layout.addWidget(self.fetch_btn)

        # Download button for BANKNIFTY and MCX
        self.download_btn = QPushButton("💾 Download CSV")
        self.download_btn.setStyleSheet(
            "QPushButton { font-size: 13px; padding: 10px; }"
        )
        self.download_btn.clicked.connect(self.download_data)
        self.download_btn.setEnabled(False)  # Enable after data is fetched
        data_layout.addWidget(self.download_btn)

        self.status = QLabel("Ready")
        self.status.setStyleSheet(
            "color: #26a69a; font-weight: bold; padding: 8px; background-color: #0d1117; border-radius: 4px;"
        )
        self.status.setWordWrap(True)
        data_layout.addWidget(self.status)

        left_layout.addWidget(data_group)

        # Indicator Panel (TradingView style)
        self.indicator_panel = IndicatorPanel()
        self.indicator_panel.indicator_added.connect(self.on_indicator_added)
        self.indicator_panel.indicator_removed.connect(self.on_indicator_removed)
        self.indicator_panel.indicator_color_changed.connect(
            self.on_indicator_color_changed
        )
        left_layout.addWidget(self.indicator_panel)

        # Debug console
        debug_group = QGroupBox("🔧 Console Log")
        debug_layout = QVBoxLayout(debug_group)
        self.debug_text = QTextEdit()
        self.debug_text.setMaximumHeight(120)
        self.debug_text.setReadOnly(True)
        debug_layout.addWidget(self.debug_text)
        left_layout.addWidget(debug_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # === RIGHT PANEL - CHART ===
        chart_panel = QWidget()
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)

        self.toolbar = ChartToolbar()
        chart_layout.addWidget(self.toolbar)

        # Chart container for multiple charts
        self.chart_container = QWidget()
        self.chart_container_layout = QVBoxLayout(self.chart_container)
        self.chart_container_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_container_layout.setSpacing(1)
        chart_layout.addWidget(self.chart_container)

        # Charts list
        self.charts = []
        self.current_layout_index = 0
        self._chart_data = {}  # Store data for each chart by index
        self._selected_chart_index = 0  # Currently selected chart for updates

        # Create initial chart
        self._create_charts(1)

        # Connect toolbar signals
        self.toolbar.magnet_toggled.connect(self.on_magnet_toggled)
        self.toolbar.theme_toggled.connect(self.on_theme_toggled)
        self.toolbar.layout_changed.connect(self.on_layout_changed)
        self.toolbar.trend_line_clicked.connect(self._on_trend_line_clicked)
        self.toolbar.horizontal_line_clicked.connect(self._on_horizontal_line_clicked)
        self.toolbar.vertical_line_clicked.connect(self._on_vertical_line_clicked)
        self.toolbar.measure_tool_clicked.connect(self._on_measure_tool_clicked)
        self.toolbar.clear_drawings_clicked.connect(self.on_clear_drawings)
        self.toolbar.scale_mode_changed.connect(self._on_scale_mode_changed)
        self.toolbar.grid_toggled.connect(self._on_grid_toggled)
        self.toolbar.fit_content_clicked.connect(self._on_fit_content_clicked)
        self.toolbar.about_clicked.connect(self.show_about_dialog)
        self.toolbar.nextstep_clicked.connect(self.on_nextstep_clicked)

        splitter.addWidget(left_scroll)
        splitter.addWidget(chart_panel)
        splitter.setSizes([350, 1250])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        self.on_data_type_change()

    def setup_shortcuts(self):
        self._shortcuts.clear()

        def add_shortcut(key, callback):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            self._shortcuts.append(shortcut)

        add_shortcut("H", self.on_horizontal_line_shortcut)
        add_shortcut("V", self.on_vertical_line_shortcut)
        add_shortcut("T", self.on_trend_line_shortcut)
        add_shortcut("M", self.on_measure_tool_shortcut)
        add_shortcut("Escape", self.on_cancel_drawing_shortcut)
        add_shortcut("F", self.on_fit_content_shortcut)
        add_shortcut("Ctrl+G", self.show_goto_dialog)
        add_shortcut("Ctrl+D", self.on_clear_drawings_shortcut)
        add_shortcut("Ctrl+I", self.on_clear_indicators_shortcut)
        add_shortcut("Delete", self.on_delete_selected_shortcut)
        add_shortcut("F1", self.show_about_dialog)
        add_shortcut("Ctrl+S", self.on_switch_chart_shortcut)

        self.log(
            "Shortcuts: H/V/T (Lines), M (Measure), F (Fit), Ctrl+G (Go To), Ctrl+S (Switch Chart), F1 (About), Esc (Cancel), Del (Delete)"
        )

    # ============================================================
    # CHART MANAGEMENT (Multi-Chart Support)
    # ============================================================

    def _get_chart_count_from_layout(self, layout_index):
        """Convert layout index to number of charts"""
        layout_map = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4}
        return layout_map.get(layout_index, 1)

    def _create_charts(self, count):
        """Create specified number of charts"""
        # Clear existing charts
        self._clear_charts()

        # Store the current chart data for re-applying
        current_data = getattr(self, "_current_chart_data", None)
        current_ticker = getattr(self, "_current_ticker", None)

        # Create charts based on layout
        self.charts = []

        layout_idx = self.toolbar.layout_combo.currentIndex()

        if layout_idx == 2:  # 2 Charts Horizontal
            # Use horizontal layout
            self.chart_container_layout.setDirection(QVBoxLayout.Direction.LeftToRight)
            for i in range(count):
                chart = self._create_single_chart(i)
                self.charts.append(chart)
                self.chart_container_layout.addWidget(chart)
        else:
            # Use vertical layout for all other layouts
            self.chart_container_layout.setDirection(QVBoxLayout.Direction.TopToBottom)
            for i in range(count):
                chart = self._create_single_chart(i)
                self.charts.append(chart)
                self.chart_container_layout.addWidget(chart)

        # Connect signals for all charts
        for i, wrapper in enumerate(self.charts):
            chart = wrapper._chart
            chart.chart_ready.connect(lambda idx=i: self._on_chart_ready_idx(idx))
            chart.chart_error.connect(self.on_chart_error)
            chart.js_log.connect(self.log)
            chart.data_loaded.connect(self.on_data_loaded)
            chart.drawing_mode_changed.connect(self.on_drawing_mode_changed)

        # Re-apply data if available (per-chart)
        for idx in range(len(self.charts)):
            chart_data = self._get_chart_data(idx)
            if chart_data:
                self.charts[idx]._chart.set_data(
                    chart_data["data"], chart_data["ticker"]
                )
            elif idx == 0 and current_data and current_ticker:
                # For first chart, apply current data if no stored data
                self.charts[idx]._chart.set_data(current_data, current_ticker)
                self._chart_data[idx] = {"data": current_data, "ticker": current_ticker}

    def _create_single_chart(self, index):
        """Create a single chart widget with click wrapper"""
        wrapper = ClickableChartWrapper(index, self)
        wrapper.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        wrapper.setStyleSheet("QWidget { margin: 0; padding: 0; }")

        # Create the chart
        chart = TradingViewChart(theme_manager=self.theme_manager)
        chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Store the chart index
        chart._chart_index = index

        wrapper.add_chart(chart)

        # Connect click signal
        wrapper.chart_clicked.connect(self._select_chart)

        return wrapper

    def _select_chart(self, index):
        """Select a chart and update UI"""
        if index >= len(self.charts):
            return

        # Update selected chart index
        old_index = self._selected_chart_index
        self._selected_chart_index = index

        # Update chart selector dropdown
        self.chart_selector.blockSignals(True)
        self.chart_selector.setCurrentIndex(index)
        self.chart_selector.blockSignals(False)

        # Visual feedback - flash the chart border
        self._highlight_chart(index)

        self.log(f"Selected Chart {index + 1}")

    def _highlight_chart(self, index):
        """Temporarily highlight the selected chart"""
        # Store original styles
        if not hasattr(self, "_chart_styles"):
            self._chart_styles = {}

        # Restore previous chart style
        if (
            hasattr(self, "_last_selected")
            and self._last_selected in self._chart_styles
        ):
            old_chart = (
                self.charts[self._last_selected]
                if self._last_selected < len(self.charts)
                else None
            )
            if old_chart:
                old_chart.setStyleSheet(self._chart_styles.get(self._last_selected, ""))

        # Store current style and add highlight
        if index < len(self.charts):
            chart = self.charts[index]
            self._chart_styles[index] = chart.styleSheet()
            chart.setStyleSheet("border: 2px solid #2196F3;")
            self._last_selected = index

            # Remove highlight after 1 second
            QTimer.singleShot(1000, lambda: self._remove_highlight(index))

    def _remove_highlight(self, index):
        """Remove the highlight from a chart"""
        if index < len(self.charts):
            chart = self.charts[index]
            if index in self._chart_styles:
                chart.setStyleSheet(self._chart_styles[index])
            else:
                chart.setStyleSheet("")

    def _clear_charts(self):
        """Remove all charts from container"""
        # Clear highlight styles cache
        if hasattr(self, "_chart_styles"):
            self._chart_styles.clear()

        while self.chart_container_layout.count() > 0:
            item = self.chart_container_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                widget.deleteLater()
        self.charts = []

    def _apply_data_to_chart(self, data, ticker, chart_index=None):
        """Apply data to a specific chart (or selected chart if None)"""
        if chart_index is None:
            chart_index = self._selected_chart_index

        if chart_index < len(self.charts):
            self.charts[chart_index]._chart.set_data(data, ticker)
            # Store data for this chart
            self._chart_data[chart_index] = {"data": data, "ticker": ticker}

    def _apply_data_to_all_charts(self, data, ticker):
        """Apply data to all charts"""
        for wrapper in self.charts:
            wrapper._chart.set_data(data, ticker)

    def _get_chart_data(self, chart_index):
        """Get stored data for a chart"""
        return self._chart_data.get(chart_index, None)

    def _apply_indicators_to_all_charts(self):
        """Apply current indicators to all charts"""
        indicators = self.indicator_panel.active_indicators
        for wrapper in self.charts:
            for ind_id, ind_data in indicators.items():
                wrapper._chart.add_indicator(
                    ind_id, ind_data["type"], ind_data["params"], ind_data["color"]
                )

    def _on_chart_ready_idx(self, index):
        """Called when a chart is ready"""
        if index == 0:
            self.log("Chart engine initialized")
            self.set_status("Chart ready - waiting for data")

    def on_layout_changed(self, layout_index):
        """Handle layout change from toolbar"""
        self.current_layout_index = layout_index
        count = self._get_chart_count_from_layout(layout_index)
        self.log(f"Layout changed to {count} chart(s)")

        # Update chart selector
        self.chart_selector.blockSignals(True)
        self.chart_selector.clear()
        for i in range(count):
            self.chart_selector.addItem(f"Chart {i + 1}")
        self.chart_selector.setCurrentIndex(min(self._selected_chart_index, count - 1))
        self.chart_selector.blockSignals(False)

        # Create charts
        self._create_charts(count)

    # Convenience property for backward compatibility
    @property
    def chart(self):
        """Return the first chart for backward compatibility"""
        return self.charts[0] if self.charts else None

    # ============================================================
    # CHART ACTIONS (Updated for multi-chart support)
    # ============================================================

    def _on_trend_line_clicked(self):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.enable_trend_line()

    def _on_horizontal_line_clicked(self):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.enable_horizontal_line()

    def _on_vertical_line_clicked(self):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.enable_vertical_line()

    def _on_measure_tool_clicked(self):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.enable_measure_tool()

    def _on_scale_mode_changed(self, mode):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.set_scale_mode(mode)

    def _on_grid_toggled(self, enabled):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.toggle_grid(enabled)

    def _on_fit_content_clicked(self):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.fit_content()

    def show_about_dialog(self):
        dialog = AboutDialog(self)
        dialog.exec()

    def on_nextstep_clicked(self):
        """Handle NextSTEP custom data upload"""
        dialog = NextSTEPDialog(self)
        dialog.data_loaded.connect(self.on_nextstep_data_loaded)
        dialog.exec()

    def on_nextstep_data_loaded(self, df, ticker):
        """Handle loaded custom data and plot it"""
        logging.debug(
            f"NextSTEP DEBUG: Received signal with {len(df)} rows, ticker={ticker}"
        )
        self.log(f"NextSTEP: Loaded {len(df)} rows for {ticker}")
        self.set_status(f"Plotting custom data: {ticker}", "#9C27B0")

        # Ensure datetime column is properly formatted
        if "datetime" in df.columns:
            # Convert to datetime if not already
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            # Drop rows with invalid datetime
            df = df.dropna(subset=["datetime"])
            # Sort by datetime
            df = df.sort_values("datetime")

        logging.debug(
            f"NextSTEP DEBUG: After processing, {len(df)} rows, datetime min={df['datetime'].min() if 'datetime' in df.columns else 'N/A'}"
        )

        # Store data for re-applying on layout changes (per-chart)
        self._chart_data[self._selected_chart_index] = {"data": df, "ticker": ticker}

        # Store for download
        self._last_fetched_df = df
        self._last_ticker = ticker
        if "datetime" in df.columns and not df.empty:
            try:
                self._last_start_date = df["datetime"].min().strftime("%Y%m%d")
                self._last_end_date = df["datetime"].max().strftime("%Y%m%d")
            except Exception as e:
                logging.error(f"NextSTEP DEBUG: Date formatting error: {e}")
                self._last_start_date = "20000101"
                self._last_end_date = datetime.now().strftime("%Y%m%d")

        # Enable download button
        self.download_btn.setEnabled(True)

        # Apply data to selected chart
        logging.debug(
            f"NextSTEP DEBUG: Applying data to chart {self._selected_chart_index}, charts count={len(self.charts)}"
        )
        self._apply_data_to_chart(df, ticker)

        # Apply indicators to selected chart (same as on_data_fetched)
        indicators = self.indicator_panel.active_indicators
        wrapper = (
            self.charts[self._selected_chart_index]
            if self._selected_chart_index < len(self.charts)
            else None
        )
        logging.debug(
            f"NextSTEP DEBUG: wrapper={wrapper is not None}, indicators count={len(indicators)}"
        )
        if wrapper:
            for ind_id, ind_data in indicators.items():
                wrapper._chart.add_indicator(
                    ind_id, ind_data["type"], ind_data["params"], ind_data["color"]
                )

    def on_horizontal_line_shortcut(self):
        self.log("H pressed - Horizontal Line")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.enable_horizontal_line()

    def on_vertical_line_shortcut(self):
        self.log("V pressed - Vertical Line")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.enable_vertical_line()

    def on_trend_line_shortcut(self):
        self.log("T pressed - Trend Line")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.enable_trend_line()

    def on_measure_tool_shortcut(self):
        self.log("M pressed - Measure Tool")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.enable_measure_tool()

    def on_cancel_drawing_shortcut(self):
        self.log("Esc pressed - Cancel")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.cancel_drawing()

    def on_fit_content_shortcut(self):
        self.log("F pressed - Fit Content")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.fit_content()

    def on_clear_drawings_shortcut(self):
        self.log("Ctrl+D pressed - Clear Drawings")
        self.on_clear_drawings()

    def on_clear_indicators_shortcut(self):
        self.log("Ctrl+I pressed - Clear Indicators")
        self.on_clear_indicators()

    def on_delete_selected_shortcut(self):
        self.log("Delete pressed - Delete Selected")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.delete_selected_drawing()

    def on_switch_chart_shortcut(self):
        """Switch to the next chart (Ctrl+S)"""
        count = self._get_chart_count_from_layout(
            self.toolbar.layout_combo.currentIndex()
        )
        if count <= 1:
            return

        # Cycle to next chart
        next_index = (self._selected_chart_index + 1) % count
        self._selected_chart_index = next_index

        # Update chart selector dropdown
        self.chart_selector.blockSignals(True)
        self.chart_selector.setCurrentIndex(next_index)
        self.chart_selector.blockSignals(False)

        # Re-apply data to the selected chart
        if next_index in self._chart_data:
            chart_info = self._chart_data[next_index]
            self._apply_data_to_chart(chart_info["data"], chart_info["ticker"])

        self.log(f"Ctrl+S pressed - Switched to Chart {next_index + 1}")

    def on_clear_drawings(self):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.clear_drawings()
        self.toolbar.clear_active_tool()

    def on_clear_indicators(self):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.clear_indicators()
        self.indicator_panel.clear_all_indicators()

    def on_drawing_mode_changed(self, mode):
        """Called when drawing mode changes in the chart"""
        if mode:
            self.toolbar.set_active_tool(mode)
            self.log(f"Tool active: {mode}")
        else:
            self.toolbar.clear_active_tool()

    def show_goto_dialog(self):
        self.log("Ctrl+G pressed - Go To Dialog")
        dialog = GoToDialog(self)
        idx = self._selected_chart_index
        if idx < len(self.charts):
            dialog.goto_requested.connect(self.charts[idx]._chart.goto_time)
        dialog.exec()

    def on_magnet_toggled(self, enabled):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.toggle_magnet(enabled)
        self.log(f"Magnet mode: {'ON' if enabled else 'OFF'}")

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.debug_text.append(f"[{timestamp}] {msg}")
        scrollbar = self.debug_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_status(self, msg, color="#26a69a"):
        self.status.setText(msg)
        self.status.setStyleSheet(
            f"color: {color}; font-weight: bold; padding: 8px; background-color: #0d1117; border-radius: 4px;"
        )

    def on_chart_ready(self):
        self.log("Chart engine initialized")
        self.set_status("Chart ready - waiting for data")

    def on_chart_error(self, error):
        self.log(f"Chart Error: {error}")
        self.set_status(f"Error: {error}", "#ef5350")

    def on_data_loaded(self, count):
        self.set_status(f"Loaded {count} candles successfully")

    def on_chart_selector_changed(self, index):
        """Handle chart selector change"""
        count = self._get_chart_count_from_layout(
            self.toolbar.layout_combo.currentIndex()
        )
        self._selected_chart_index = min(index, count - 1)
        self.log(f"Selected Chart {self._selected_chart_index + 1}")

    def on_data_type_change(self):
        is_options = self.data_type.currentText() == "Options"

        self.update_symbol_list()

        self.expiry_type_row.setVisible(is_options)
        self.expiry_row.setVisible(is_options)
        self.strike_row.setVisible(is_options)
        self.side_row.setVisible(is_options)

        self.end_date.setEnabled(True)

        if is_options:
            self.fetch_expiry_dates_async()

    def on_exchange_change(self):
        self.update_symbol_list()
        if self.data_type.currentText() == "Options":
            self.fetch_expiry_dates_async()

        # Update download button state - enable for all symbols if data exists
        if hasattr(self, "_last_fetched_df") and self._last_fetched_df is not None:
            self.download_btn.setEnabled(True)
        else:
            self.download_btn.setEnabled(False)

        # For MCX, only allow Monthly options (segment='O')
        exchange = self.exchange.currentText()
        symbol = self.symbol.currentText()
        if self.data_type.currentText() == "Options" and (
            symbol == "BANKNIFTY" or exchange == "MCX"
        ):
            self.expiry_type.blockSignals(True)
            self.expiry_type.setCurrentText("Monthly")
            self.expiry_type.setEnabled(False)  # Disable to force Monthly
            self.expiry_type.blockSignals(False)
        elif self.data_type.currentText() == "Options":
            self.expiry_type.setEnabled(True)

    def update_symbol_list(self):
        """Update symbol list based on exchange and data type selection"""
        exchange = self.exchange.currentText()
        is_options = self.data_type.currentText() == "Options"

        # Define symbols per exchange
        nse_symbols = ["NIFTY", "BANKNIFTY"]
        bse_symbols = ["SENSEX"]
        nyse_symbols = ["SPY"]
        mcx_symbols = [
            "COPPER",
            "CRUDEOIL",
            "CRUDEOILM",
            "GOLD",
            "GOLDM",
            "NATGASMINI",
            "NATURALGAS",
            "NICKEL",
            "SILVER",
            "SILVERM",
            "ZINC",
            "ZINCMINI",
        ]

        # Spot symbols (different naming)
        nse_spot = ["NIFTY_50", "NIFTY_BANK"]

        if exchange == "NSE":
            symbols = nse_symbols if is_options else nse_spot
        elif exchange == "BSE":
            symbols = bse_symbols
        elif exchange == "NYSE":
            symbols = (
                nyse_symbols if is_options else nyse_symbols
            )  # SPY for both options and spot
        elif exchange == "MCX":
            symbols = mcx_symbols
        else:
            symbols = nse_symbols if is_options else nse_spot

        self.symbol.blockSignals(True)
        self.symbol.clear()
        self.symbol.addItems(symbols)
        self.symbol.blockSignals(False)

        # For BANKNIFTY and MCX, only allow Monthly options
        symbol = self.symbol.currentText()
        if is_options and (symbol == "BANKNIFTY" or exchange == "MCX"):
            self.expiry_type.blockSignals(True)
            self.expiry_type.setCurrentText("Monthly")
            self.expiry_type.setEnabled(False)  # Disable to force Monthly
            self.expiry_type.blockSignals(False)
        else:
            self.expiry_type.setEnabled(True)

    def on_symbol_change(self):
        if self.data_type.currentText() == "Options":
            self.fetch_expiry_dates_async()

        # Update download button state - enable for all symbols if data exists
        if hasattr(self, "_last_fetched_df") and self._last_fetched_df is not None:
            self.download_btn.setEnabled(True)
        else:
            self.download_btn.setEnabled(False)

        # For BANKNIFTY and MCX, only allow Monthly options
        exchange = self.exchange.currentText()
        symbol = self.symbol.currentText()
        if self.data_type.currentText() == "Options" and (
            symbol == "BANKNIFTY" or exchange == "MCX"
        ):
            self.expiry_type.blockSignals(True)
            self.expiry_type.setCurrentText("Monthly")
            self.expiry_type.setEnabled(False)  # Disable to force Monthly
            self.expiry_type.blockSignals(False)
        elif self.data_type.currentText() == "Options":
            self.expiry_type.setEnabled(True)

    def on_expiry_type_change(self):
        self.fetch_expiry_dates_async()

    def on_start_date_changed(self):
        if self.data_type.currentText() == "Options":
            self.update_expiry_combo()

    def on_expiry_selected(self, expiry_str):
        if expiry_str and self.data_type.currentText() == "Options":
            try:
                expiry_date = datetime.strptime(expiry_str, "%d-%m-%Y").date()
                self.end_date.setDate(
                    QDate(expiry_date.year, expiry_date.month, expiry_date.day)
                )
            except (ValueError, TypeError):
                pass

    def fetch_expiry_dates_async(self):
        if self.data_type.currentText() != "Options":
            return

        self.set_status("Loading expiry dates...", "#FF9800")

        worker = ExpiryWorker()
        worker.set_params(
            self.symbol.currentText(),
            self.expiry_type.currentText(),
            self.exchange.currentText(),
        )

        self.thread_manager.create_thread(
            worker=worker,
            started_callback=worker.fetch_expiry_dates,
            finished_callbacks=[(worker.finished, self.on_expiry_dates_loaded)],
            error_callback=lambda e: self.log(f"Expiry fetch error: {e}"),
        )

    def on_expiry_dates_loaded(self, expiry_list):
        self.expiry_dates_list = expiry_list
        self.log(f"Loaded {len(expiry_list)} expiry dates")
        self.set_status("Ready")
        self.update_expiry_combo()

    def update_expiry_combo(self):
        if not self.expiry_dates_list:
            return

        start = self.start_date.date().toPyDate()

        filtered_expiries = [d for d in self.expiry_dates_list if d.date() >= start]

        filtered_expiries.sort()

        self.expiry_combo.blockSignals(True)
        self.expiry_combo.clear()

        for exp in filtered_expiries:
            exp_str = exp.strftime("%d-%m-%Y")
            self.expiry_combo.addItem(exp_str)

        self.expiry_combo.blockSignals(False)

        if self.expiry_combo.count() > 0:
            self.expiry_combo.setCurrentIndex(0)
            self.on_expiry_selected(self.expiry_combo.currentText())

        self.log(f"Showing {len(filtered_expiries)} expiry dates from {start}")

    def on_theme_toggled(self, _):
        """Handle theme toggle from toolbar"""
        new_theme = self.theme_manager.toggle_theme()
        is_dark = new_theme == "dark"

        # Update toolbar icon
        self.toolbar.set_theme_icon(is_dark)

        # Apply theme to all charts
        for chart in self.charts:
            chart.apply_theme(new_theme)

        # Apply theme to main window
        self.apply_theme_colors(new_theme)

        self.log(f"Theme switched to: {new_theme.upper()}")

    def apply_theme_colors(self, theme):
        """Apply theme colors to UI elements"""
        colors = self.theme_manager.get_theme_colors(theme)

        if theme == "dark":
            self.setup_dark_theme()
        else:
            # Light theme stylesheet
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #f5f5f5; color: #1a1a1a; }
                QGroupBox { 
                    border: 1px solid #cccccc; border-radius: 6px; margin-top: 14px; 
                    padding-top: 12px; font-weight: bold; background-color: #ffffff;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #1976D2; font-size: 13px; }
                QComboBox, QLineEdit, QSpinBox, QDateEdit { 
                    background-color: #ffffff; border: 1px solid #cccccc; border-radius: 4px; 
                    padding: 6px; color: #1a1a1a; min-height: 26px; 
                }
                QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDateEdit:hover { border-color: #1976D2; }
                QComboBox QAbstractItemView { background-color: #ffffff; color: #1a1a1a; selection-background-color: #1976D2; border: 1px solid #cccccc; }
                QPushButton { 
                    background-color: #1976D2; border: none; border-radius: 4px; 
                    padding: 8px 14px; color: white; font-weight: bold; min-height: 28px;
                }
                QPushButton:hover { background-color: #1565C0; }
                QPushButton:pressed { background-color: #0d47a1; }
                QPushButton:disabled { background-color: #e0e0e0; color: #999999; }
                QLabel { color: #666666; }
                QTextEdit { 
                    background-color: #ffffff; border: 1px solid #cccccc; color: #333333; 
                    font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; border-radius: 4px;
                }
                QScrollArea { border: none; background-color: transparent; }
                QScrollBar:vertical { background: #e0e0e0; width: 10px; border-radius: 5px; }
                QScrollBar::handle:vertical { background: #bdbdbd; border-radius: 5px; min-height: 20px; }
                QScrollBar::handle:vertical:hover { background: #9e9e9e; }
                QSplitter::handle { background-color: #cccccc; }
            """)

    def on_indicator_added(self, ind_type, params, color):
        ind_id = f"{ind_type.lower()}_{self.indicator_panel.indicator_counter}"
        self.log(f"Adding {ind_type} with params {params}")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.add_indicator(ind_id, ind_type, params, color)

    def on_indicator_removed(self, ind_id):
        self.log(f"[DEBUG] MainWindow.on_indicator_removed called for {ind_id}")
        if ind_id == "all":
            # Clear indicators from all charts
            for chart_wrapper in self.charts:
                chart_wrapper._chart.clear_indicators()
            self.indicator_panel.clear_all_indicators()
            self.log("All indicators cleared from all charts")
        else:
            # Remove from ALL charts, not just selected chart
            self.log(f"Removing {ind_id} from {len(self.charts)} charts")
            for i, chart_wrapper in enumerate(self.charts):
                if hasattr(chart_wrapper, "_chart"):
                    self.log(f"Removing {ind_id} from chart {i}")
                    chart_wrapper._chart.remove_indicator(ind_id)
                else:
                    self.log(f"Chart wrapper {i} has no _chart attribute")
            self.log(f"Removed {ind_id} from all charts")

    def on_indicator_color_changed(self, ind_id, color):
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.update_indicator_color(ind_id, color)
        self.log(f"Changed {ind_id} color to {color}")

    def fetch_and_plot(self):
        if self.data_type.currentText() == "Options":
            strike_text = self.strike_price.text().strip()
            if not strike_text:
                self.set_status("Please enter strike price", "#ef5350")
                return
            try:
                int(float(strike_text))
            except ValueError:
                self.set_status("Invalid strike price", "#ef5350")
                return

            if not self.expiry_combo.currentText():
                self.set_status("Please select expiry date", "#ef5350")
                return

        self.fetch_btn.setEnabled(False)
        self.set_status("Fetching data from API...", "#FF9800")

        worker = ApiWorker()

        # Get interval value (user can type custom value)
        interval_text = self.interval.currentText().strip()
        if not interval_text:
            interval_text = "15min"

        params = {
            "data_type": self.data_type.currentText(),
            "symbol": self.symbol.currentText(),
            "start": self.start_date.date().toPyDate().strftime("%d-%m-%Y"),
            "end": self.end_date.date().toPyDate().strftime("%d-%m-%Y"),
            "freq": interval_text,
            "exchange": self.exchange.currentText(),
        }

        if self.data_type.currentText() == "Options":
            params["strike"] = int(float(self.strike_price.text()))
            params["side"] = self.option_type.currentText()
            params["expiry"] = self.expiry_combo.currentText()

        worker.set_params(**params)

        self.thread_manager.create_thread(
            worker=worker,
            started_callback=worker.fetch_data,
            finished_callbacks=[
                (worker.finished, self.on_data_fetched),
                (worker.progress, self.log),
            ],
            error_callback=self.on_fetch_error,
        )

    def on_indicator_added(self, ind_id, ind_type, params, color):
        # If ID is not provided (shouldn't happen with updated signal), generate it
        if not ind_id:
            ind_id = f"{ind_type.lower()}_{self.indicator_panel.indicator_counter}"

        self.log(f"Adding/Updating {ind_type} (ID: {ind_id}) with params {params}")
        idx = self._selected_chart_index
        if idx < len(self.charts):
            self.charts[idx]._chart.add_indicator(ind_id, ind_type, params, color)

    def on_data_fetched(self, df, ticker):
        self.fetch_btn.setEnabled(True)

        if df is None or df.empty:
            self.set_status("No data returned", "#ef5350")
            return

        self.log(f"Received {len(df)} rows for {ticker}")
        self.set_status(
            f"Plotting {len(df)} candles on Chart {self._selected_chart_index + 1}...",
            "#FF9800",
        )

        # Store data for re-applying on layout changes (per-chart)
        self._chart_data[self._selected_chart_index] = {"data": df, "ticker": ticker}

        # Store last fetched data for download
        self._last_fetched_df = df
        self._last_ticker = ticker
        self._last_start_date = self.start_date.date().toPyDate().strftime("%Y%m%d")
        self._last_end_date = self.end_date.date().toPyDate().strftime("%Y%m%d")

        # Enable download button after data is fetched
        if hasattr(self, "_last_fetched_df") and self._last_fetched_df is not None:
            self.download_btn.setEnabled(True)

        # Apply data to selected chart only
        self._apply_data_to_chart(df, ticker)

        # Apply indicators to selected chart only
        indicators = self.indicator_panel.active_indicators
        self.log(
            f"[DEBUG] on_data_fetched applying indicators: {list(indicators.keys())}"
        )

        wrapper = (
            self.charts[self._selected_chart_index]
            if self._selected_chart_index < len(self.charts)
            else None
        )
        if wrapper:
            for ind_id, ind_data in indicators.items():
                self.log(f"[DEBUG] Re-applying indicator {ind_id} to chart")
                wrapper._chart.add_indicator(
                    ind_id, ind_data["type"], ind_data["params"], ind_data["color"]
                )

    def on_fetch_error(self, error):
        self.fetch_btn.setEnabled(True)
        self.log(f"Fetch failed: {error}")
        self.set_status("Data fetch failed", "#ef5350")

    def download_data(self):
        """Download fetched data to CSV file in data folder"""
        if not hasattr(self, "_last_fetched_df") or self._last_fetched_df is None:
            self.log("No data available for download")
            return

        try:
            # Create data folder if it doesn't exist
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_folder = os.path.join(script_dir, "data")
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
                self.log(f"Created data folder: {data_folder}")

            # Generate filename: ticker_startdate_enddate.csv
            filename = (
                f"{self._last_ticker}_{self._last_start_date}_{self._last_end_date}.csv"
            )
            filepath = os.path.join(data_folder, filename)

            # Create a copy of the dataframe to add TICKER column
            df_export = self._last_fetched_df.copy()

            # Add TICKER column based on data type
            if self.data_type.currentText() == "Options":
                # Extract ticker components from the stored ticker
                ticker = self._last_ticker
                df_export.insert(0, "TICKER", ticker)
            else:
                # For spot data, use the symbol as ticker
                df_export.insert(0, "TICKER", self.symbol.currentText())

            # Save to CSV
            df_export.to_csv(filepath, index=False)

            self.log(f"Data saved to: {filepath}")
            self.set_status(f"Data downloaded to {filename}", "#26a69a")

        except Exception as e:
            self.log(f"Download failed: {str(e)}")
            self.set_status("Download failed", "#ef5350")

    def closeEvent(self, event):
        self.log("Shutting down...")
        self.thread_manager.stop_all()
        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Application starting...")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
