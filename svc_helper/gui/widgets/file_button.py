from PyQt5.QtWidgets import QPushButton, QFileDialog
from PyQt5.QtCore import pyqtSignal

class FileButton(QPushButton):
    """
    A custom QPushButton that allows file selection through clicking or drag-and-drop.
    
    This button emits a signal with selected file paths when files are chosen either
    through a file dialog (when clicked) or through drag-and-drop operations.
    
    Attributes:
        filesSelected (pyqtSignal): Signal emitted with a list of selected file paths.
    """
    
    filesSelected = pyqtSignal(list)
    
    def __init__(self,
                 label="Files to Convert",
                 dialog_title="Select files",
                 dialog_filter="All Files (*);;Text Files (*.txt)"):
        """
        Initialize the FileButton.
        
        Args:
            label (str): The text displayed on the button. Defaults to "Files to Convert".
            dialog_title (str): The title of the file dialog window. Defaults to "Select files".
            dialog_filter (str): File filter for the dialog. Defaults to "All Files (*);;Text Files (*.txt)".
        """
        super().__init__(label)
        self.setAcceptDrops(True)
        self.clicked.connect(self.loadFileDialog)
        self.dialog_title = dialog_title
        self.dialog_filter = dialog_filter
        
    def loadFileDialog(self):
        """
        Open a file dialog to allow the user to select multiple files.
        
        Emits the filesSelected signal with the list of selected file paths.
        """
        filenames, _ = QFileDialog.getOpenFileNames(
            self, self.dialog_title, "", self.dialog_filter
        )
        self.filesSelected.emit(filenames)
    
    def dragEnterEvent(self, event):
        """
        Handle drag enter events to determine if dropped content should be accepted.
        
        Args:
            event (QDragEnterEvent): The drag enter event containing mime data.
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """
        Handle drop events to process dropped files.
        
        Extracts file paths from dropped URLs and emits the filesSelected signal
        with the list of valid local file paths.
        
        Args:
            event (QDropEvent): The drop event containing the dropped data.
        """
        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                if not url.toLocalFile():
                    continue
                files.append(url.toLocalFile())
            self.filesSelected.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()