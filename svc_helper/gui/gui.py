from typing import Callable
from PyQt5.QtWidgets import QGroupBox, QWidget, QHBoxLayout, QVBoxLayout
from omegaconf import OmegaConf


from .widgets.recorder import AudioRecorder

from .modules.inference import Inference
from .modules.checkpoint import Checkpoint
from .modules.file_input import AudioFileInput
from .modules.param import BaseParam
from importlib import resources

def default_config():
    with resources.open_text("svc_helper.gui", "config.yaml") as f:
        return OmegaConf.load(f)

class VoiceGUI(QWidget):
    def __init__(self, config : OmegaConf = default_config(),
        recording = True): 
        super().__init__()
        self.config = config

        self.inference_box = QGroupBox()
        self.inference_box.setTitle("Inference")
        inference_layout = QVBoxLayout(self.inference_box)

        self.recording_box = QGroupBox()
        self.recording_box.setTitle("Recording")
        recording_layout = QVBoxLayout(self.recording_box)

        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.inference_box)
        if recording:
            self.layout.addWidget(self.recording_box)

        self.file_inputs = []
        self.checkpoints = []
        self.params = []
        self.inferences = []
        self.recording = recording

    def addCheckpoint(self, x : Checkpoint):
        self.checkpoints.append(x)
        return x

    def addFileInput(self, x : AudioFileInput):
        self.file_inputs.append(x)
        return x

    def addParam(self, x : BaseParam):
        self.params.append(x)
        return x

    def addInference(self, x : Inference):
        self.inferences.append(x)
        return x

    def build(self):
        for x in self.checkpoints:
            self.inference_box.layout().addWidget(x)
        for x in self.file_inputs:
            self.inference_box.layout().addWidget(x)
        for x in self.params:
            self.inference_box.layout().addWidget(x)
        for x in self.inferences:
            x: Inference
            x.gui_hook(self.get_params, self.config)
            self.inference_box.layout().addWidget(x)
        if self.recording:
            assert len(self.file_inputs) > 0, "Must have at least one file input"
            self.recorder = AudioRecorder(
                lambda f: self.file_inputs[0].setFiles([f]),
                record_dir=self.config.files.default_record_dir
            )
            self.recording_box.layout().addWidget(self.recorder)
        return self

    def get_params(self):
        ret = {x.id: x.value() for x in self.params}
        if len(self.file_inputs) > 0:
            ret["audio_files"] = {x.id: x.files() for x in self.file_inputs}
        if len(self.checkpoints) > 0:
            ret["model_labels"] = [x.value() for x in self.checkpoints]
        return ret