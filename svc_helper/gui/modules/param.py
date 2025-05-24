import os
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QCheckBox
from PyQt5.QtGui import QIntValidator, QDoubleValidator

class BaseParam(QWidget):
    def __init__(self, label="Param", id="param"):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(QLabel(label))
        self.layout.addWidget(self.field())
        self.id = id

    def field(self):
        self._field = QLineEdit()
        return self._field

class IntParam(BaseParam):
    def __init__(self,
        label="Param",
        id="param",
        default=0,
        min=0,
        max=100,
        on_change=lambda x: None
    ):
        self.validator = QIntValidator(min, max)
        super().__init__(label, id)
        self._field : QLineEdit
        self._field.setText(str(default))
        self._field.textChanged.connect(on_change)

    def field(self):
        line_edit = QLineEdit()
        line_edit.setValidator(self.validator)
        self._field = line_edit
        return line_edit

    def value(self):
        return int(self._field.text())

class DoubleParam(BaseParam):
    def __init__(self,
        label="Param",
        id="param",
        default=0.0,
        min=0.0,
        max=100.0,
        decimals=2,
        on_change=lambda x: None
    ):
        self.validator = QDoubleValidator(min, max, decimals)
        super().__init__(label, id)
        self._field : QLineEdit
        self._field.setText(str(default))
        self._field.textChanged.connect(on_change)
    
    def field(self):
        line_edit = QLineEdit()
        line_edit.setValidator(self.validator)
        self._field = line_edit
        return line_edit

    def value(self):
        return float(self._field.text())

class StringParam(BaseParam):
    def __init__(self,
        label="Param",
        id="param",
        default="",
        on_change=lambda x: None
    ):
        super().__init__(label, id)
        self._field : QLineEdit
        self._field.setText(default)
        self._field.textChanged.connect(on_change)

    def value(self):
        return self._field.text()

class BoolParam(BaseParam):
    def __init__(self,
        label="Param",
        id="param",
        default=False,
        on_change=lambda x: None
    ):
        self._field : QCheckBox
        super().__init__(label, id)
        self._field.setChecked(default)
        self._field.stateChanged.connect(lambda x: on_change(self.value()))

    def field(self):
        self._field = QCheckBox()
        return self._field

    def value(self):
        return self._field.isChecked()