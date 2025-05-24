import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
import numpy as np
from .audio_preview import AudioPreviewWidget
from ..utils import elide_string
import os

class AudioRecorder(QGroupBox):
    def __init__(self, push_fn=None, record_dir=None):
        super().__init__()
        self.setTitle("Audio recorder")
        self.setStyleSheet("padding:10px")
        self.layout = QVBoxLayout(self)
        self.push_fn = push_fn

        self.audio_settings = QAudioEncoderSettings()
        if os.name == "nt":
            self.audio_settings.setCodec("audio/pcm")
        else:
            self.audio_settings.setCodec("audio/x-raw")
        self.audio_settings.setSampleRate(44100)
        self.audio_settings.setBitRate(16)
        self.audio_settings.setQuality(QMultimedia.HighQuality)
        self.audio_settings.setEncodingMode(
            QMultimedia.ConstantQualityEncoding)

        self.preview = AudioPreviewWidget()
        self.layout.addWidget(self.preview)

        self.recorder = QAudioRecorder()
        self.input_dev_box = QComboBox()
        self.input_dev_box.setSizePolicy(QSizePolicy.Preferred,
            QSizePolicy.Preferred)
        if os.name == "nt":
            self.audio_inputs = self.recorder.audioInputs()
        else:
            self.audio_inputs = [x.deviceName() 
                for x in QAudioDeviceInfo.availableDevices(0)]

        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.toggle_record)
        self.layout.addWidget(self.record_button)

        for inp in self.audio_inputs:
            if self.input_dev_box.findText(elide_string(inp,60)) == -1:
                self.input_dev_box.addItem(elide_string(inp,60))
        self.layout.addWidget(self.input_dev_box)
        self.input_dev_box.currentIndexChanged.connect(self.set_input_dev)
        if len(self.audio_inputs) == 0:
            self.record_button.setEnabled(False) 
            print("No audio inputs found")
        else:
            self.set_input_dev(0) # Always use the first listed output
        # Doing otherwise on Windows would require platform-specific code

        self.probe = QAudioProbe()
        self.probe.setSource(self.recorder)
        self.probe.audioBufferProbed.connect(self.update_volume)
        self.volume_meter = QProgressBar()
        self.volume_meter.setTextVisible(False)
        self.volume_meter.setRange(0, 100)
        self.volume_meter.setValue(0)
        self.layout.addWidget(self.volume_meter)

        # RECORD_DIR
        self.record_dir = os.path.abspath(record_dir)
        self.record_dir_button = QPushButton("Change Recording Directory")
        self.layout.addWidget(self.record_dir_button)
        self.record_dir_label = QLabel("Recordings directory: "+str(
            self.record_dir))
        self.record_dir_button.clicked.connect(self.record_dir_dialog)

        self.last_output = ""

        self.sovits_button = QPushButton("Push last output")
        self.layout.addWidget(self.sovits_button)
        self.sovits_button.clicked.connect(lambda: self.push_fn(self.last_output))

        self.automatic_checkbox = QCheckBox("Send automatically")
        self.layout.addWidget(self.automatic_checkbox)

        self.layout.addStretch()

    def update_volume(self, buf):
        sample_size = buf.format().sampleSize()
        sample_count = buf.sampleCount()
        ptr = buf.constData()
        ptr.setsize(int(sample_size/8)*sample_count)

        samples = np.asarray(np.frombuffer(ptr, np.int16)).astype(float)
        rms = np.sqrt(np.mean(samples**2))
            
        level = rms / (2 ** 14)

        self.volume_meter.setValue(int(level * 100))

    def set_input_dev(self, idx):
        num_audio_inputs = len(self.audio_inputs)
        if idx < num_audio_inputs:
            self.recorder.setAudioInput(self.audio_inputs[idx])

    def record_dir_dialog(self):
        temp_record_dir = QFileDialog.getExistingDirectory(self,
            "Recordings Directory", self.record_dir, QFileDialog.ShowDirsOnly)
        if not os.path.exists(temp_record_dir): 
            return
        self.record_dir = temp_record_dir
        self.record_dir_label.setText(
            "Recordings directory: "+str(self.record_dir))
        
    def toggle_record(self):
        #print("toggle_record triggered at "+str(id(self)))
        if self.recorder.status() == QAudioRecorder.RecordingStatus:
            self.recorder.stop()
            self.record_button.setText("Record")
            self.last_output = self.recorder.outputLocation().toLocalFile()
            self.preview.from_file(self.last_output)
            self.preview.set_text("Preview - "+os.path.basename(
                self.last_output))
            if self.automatic_checkbox.isChecked():
                self.push_fn(self.last_output)
        else:
            self.record()
            self.record_button.setText("Recording to "+str(
                self.recorder.outputLocation().toLocalFile()))

    def record(self):
        unix_time = time.time()
        self.recorder.setEncodingSettings(self.audio_settings)
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir, exist_ok=True)
        output_name = "rec_"+str(int(unix_time))
        self.recorder.setOutputLocation(QUrl.fromLocalFile(os.path.join(
            self.record_dir,output_name)))
        self.recorder.setContainerFormat("audio/x-wav")
        self.recorder.record()