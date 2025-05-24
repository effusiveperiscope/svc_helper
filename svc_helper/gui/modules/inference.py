import os
from typing import Any, Callable
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import (QObject, QRunnable, QThreadPool, pyqtSignal, Qt)
from dataclasses import dataclass
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from ..utils import get_sanitized_filename
from ..widgets.stopwatch import Stopwatch

@dataclass
class AudioResult:
    label: str # used for file name
    audio: np.ndarray

@dataclass
class InferenceResult:
    audios: list[AudioResult]

@dataclass
class InferenceInfo:
    sr: int
    extension: str

class InferenceWorkerEmitters(QObject):
    finished = pyqtSignal(InferenceResult)

class InferenceWorker(QRunnable):
    def __init__(self,
        params: dict[str, Any],
        infer_action : Callable[[dict[str, Any]], AudioResult],
        ):
        super().__init__()
        self.params = params
        self.infer_action = infer_action
        self.emitters = InferenceWorkerEmitters()

    def run(self):
        try:
            result = self.infer_action(self.params)
        except Exception as e:
            print(e)
            result = InferenceResult([])
        self.emitters.finished.emit(result)

class Inference(QWidget):
    def __init__(self,
        info : InferenceInfo,
        infer_action : Callable[[dict[str, Any]], AudioResult],
        label="Infer",
        ):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.info = info

        self.push_button = QPushButton(label)
        self.push_button.clicked.connect(lambda: infer_action(self.get_params()))
        self.layout.addWidget(self.push_button)

        self.thread_pool = QThreadPool()
        self.stopwatch = Stopwatch()
        self.layout.addWidget(self.stopwatch)

    def infer(self, infer_action : Callable[[dict[str, Any]], AudioResult]):
        worker = InferenceWorker(
                self.get_params(), infer_action, )
        worker.emitters.finished.connect(self.infer_done)
        self.thread_pool.start(worker)
        self.stopwatch.stop_reset_stopwatch()
        self.stopwatch.start_stopwatch()

    def infer_done(self, result : InferenceResult):
        self.stopwatch.stop_reset_stopwatch()
        for audio in result.audios:
            base_output_path = os.path.join(
                self.config.files.default_output_dir, audio.label + "." + self.info.extension)
            output_path = get_sanitized_filename(base_output_path)
            sf.write(output_path, audio.audio, self.info.sr)
        
    def gui_hook(self, get_params : Callable[[], dict[str, Any]], config : OmegaConf):
        self.get_params = get_params
        self.config = config