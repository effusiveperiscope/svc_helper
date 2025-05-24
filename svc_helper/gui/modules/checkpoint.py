import os
from PyQt5.QtWidgets import QWidget, QComboBox, QVBoxLayout, QLabel
from typing import Callable
from pathlib import Path

class StandardCheckpointHandler:
    def __init__(self, config):
        self.config = config
        self.models_path = config.files.default_models_dir

        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        self._speakers = {}
        self.update()

    def update(self):
        speakers = [
            spk for spk in Path(self.models_path).iterdir() if spk.is_dir()]
        for spk in speakers:
            model_path = next(spk.glob("*.ckpt") or spk.glob("*.pth"))
            self._speakers[spk.name] = {'model_path': str(model_path)} 

    def speakers(self):
        return self._speakers

    def __call__(self):
        return [spk for spk in self._speakers.keys()]

class Checkpoint(QWidget):
    PLACEHOLDER = '--Select Checkpoint--'
    def __init__(self,
        get_checkpoints : Callable[[],str],
        load_checkpoint : Callable[[str],None],
        label: str = "Load Checkpoint"):
        super().__init__()

        # get checkpoints - returns list of checkpoint names which are passed to load_checkpoint
        self.get_checkpoints = get_checkpoints

        self.layout = QVBoxLayout(self)

        self.combo_box = QComboBox()
        self.layout.addWidget(QLabel(label))
        self.layout.addWidget(self.combo_box)

        self.update()
        # Avoid triggering load checkpoint on initialization
        self.combo_box.currentIndexChanged.connect(self._on_selected)
        self.load_checkpoint = load_checkpoint

    def _on_selected(self, index):
        if self.combo_box.itemText(index) == self.PLACEHOLDER:
            return
        self.load_checkpoint(self.combo_box.itemText(index))

    def value(self):
        return self.combo_box.currentText()

    def update(self):
        checkpoints = self.get_checkpoints()
        self.combo_box.clear()
        self.combo_box.addItem(self.PLACEHOLDER)
        for checkpoint in checkpoints:
            self.combo_box.addItem(checkpoint)