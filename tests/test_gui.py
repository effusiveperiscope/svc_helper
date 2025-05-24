from svc_helper.gui import *
from PyQt5.QtWidgets import QMainWindow, QApplication

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice GUI")
        self.setGeometry(100, 100, 800, 600)

        gui = VoiceGUI()
        gui.addCheckpoint(Checkpoint(
            lambda: ['checkpoint 1', 'checkpoint 2'], lambda x: print(x)))
        gui.addFileInput(AudioFileInput())
        gui.addParam(IntParam(label="Int Param",
            id="int_param"))
        gui.addParam(DoubleParam(label="Double Param",
            id="double_param"))
        gui.addParam(StringParam(label="String Param",
            id="string_param"))
        gui.addParam(BoolParam(label="Bool Param",
            id="bool_param"))
        gui.addInference(Inference(
            info=InferenceInfo(sr=48000, extension="flac"),
            infer_action=lambda x: print(x)
        ))
        self.setCentralWidget(gui.build())

def test_gui():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()