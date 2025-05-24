from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from ..widgets.file_button import FileButton
from ..widgets.audio_preview import AudioPreviewWidget

class AudioFileInput(QWidget):
    def __init__(self, id="files", label="Input Files"):
        super().__init__()
        self.layout = QVBoxLayout()

        self.file_button = FileButton()
        self.layout.addWidget(self.file_button)
        self.file_button.filesSelected.connect(self._onFilesSelected)
        
        self._files = []
        self.label = label
        self.files_label = QLabel()
        self._updateLabel()
        self.layout.addWidget(self.files_label)

        self.preview = AudioPreviewWidget()
        self.layout.addWidget(self.preview)

        super().__init__()
        self.setLayout(self.layout)

        self.id = id

    def files(self):
        return self._files

    def setFiles(self, files):
        self._onFilesSelected(files)

    def _onFilesSelected(self, files):
        self._files = files
        self._updateLabel()
        if not len(files): return
        self.preview.from_file(files[0])

    def _updateLabel(self):
        self.files_label.setText(f"{self.label}: {self.files()}")