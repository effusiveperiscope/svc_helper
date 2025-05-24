from PyQt5.QtWidgets import (QWidget,
    QVBoxLayout, QLabel, QFrame, QHBoxLayout, QSlider, QSizePolicy,
    QStyle, QPushButton)
from PyQt5.QtCore import (
    QMimeData, QUrl, Qt, QByteArray, QBuffer, QRunnable, QObject, pyqtSignal
)
from PyQt5.QtGui import (
    QDrag, QPainter, QColor, QPixmap, QCursor
)
from PyQt5.QtMultimedia import (
   QMediaContent, QAudio, QAudioDeviceInfo, QMediaPlayer)
import soundfile as sf
import os
import time
import numpy as np
from io import BytesIO
from gui.util import qshrink

class AudioPreviewWidget(QWidget):
    """
    A widget for audio playback with optional text display and dragging support.
    Allows pausing and seeking within the audio.
    """
    def __init__(self, button_only=False, drag_enabled=True, pausable=True):
        super().__init__()
        self.vlayout = QVBoxLayout(self)
        self.vlayout.setSpacing(0)
        self.vlayout.setContentsMargins(0,0,0,0)

        self.playing_label = QLabel("Preview")
        self.playing_label.setWordWrap(True)
        if not button_only:
            self.vlayout.addWidget(self.playing_label)

        self.player_frame = QFrame()
        self.vlayout.addWidget(self.player_frame)

        self.player_layout = QHBoxLayout(self.player_frame)
        self.player_layout.setSpacing(4)
        self.player_layout.setContentsMargins(0,0,0,0)

        self.player = QMediaPlayer()
        self.player.setNotifyInterval(500)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        if not button_only:
            self.player_layout.addWidget(self.seek_slider)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.player_layout.addWidget(self.play_button)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)

        if drag_enabled:
            self.play_button.mouseMoveEvent = self.drag_hook

        self.seek_slider.sliderMoved.connect(self.seek)
        self.player.positionChanged.connect(self.update_seek_slider)
        self.player.stateChanged.connect(self.state_changed)
        self.player.durationChanged.connect(self.duration_changed)

        self.local_file = ""
        self.pausable = pausable

    def set_text(self, text=""):
        """Set the text label displayed above the player."""
        if len(text) > 0:
            self.playing_label.show()
            self.playing_label.setText(text)
        else:
            self.playing_label.hide()

    def from_file(self, path):
        """Load an audio file from disk for playback."""
        try:
            self.player.stop()
            if hasattr(self, 'audio_buffer'):
                self.audio_buffer.close()

            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(path))))
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.local_file = path
        except Exception as e:
            print(e)
            pass

    def drag_hook(self, e):
        """Initiate drag-and-drop operation with the audio file."""
        if e.buttons() != Qt.LeftButton:
            return
        if not len(self.local_file):
            return

        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(os.path.abspath(self.local_file))])
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec_(Qt.CopyAction)

    def from_memory(self, data):
        """Load audio data from memory (e.g., in QByteArray format)."""
        self.player.stop()
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer.close()

        self.audio_data = QByteArray(data)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)
        self.player.setMedia(QMediaContent(), self.audio_buffer)

    def state_changed(self, state):
        """Update UI icon when player state changes."""
        if state in [QMediaPlayer.StoppedState, QMediaPlayer.PausedState]:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def duration_changed(self, dur):
        """Update slider range when media duration is known."""
        self.seek_slider.setRange(0, self.player.duration())

    def toggle_play(self):
        """Toggle between playing and paused/stopped state."""
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause() if self.pausable else self.player.stop()
        elif self.player.mediaStatus() != QMediaPlayer.NoMedia:
            self.player.play()
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def update_seek_slider(self, position):
        """Update slider position as the media plays."""
        self.seek_slider.setValue(position)

    def seek(self, position):
        """Seek to a position in the media."""
        self.player.setPosition(position)

class SmallAudioPreviewWidget(QWidget):
    """
    A minimal widget for audio playback with only a play/stop button.
    Loads and unloads media player on each playback.
    """
    def __init__(self, local_file: str):
        super().__init__()
        self.pb = QPushButton()
        self.pb.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        lay = QVBoxLayout(self)
        lay.addWidget(self.pb)
        self.pb.clicked.connect(self.toggle_play)
        self.pb.setFixedWidth(30)
        self.pb.setFixedHeight(20)
        self.pb.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.local_file = local_file
        self.is_playing = False

    def stop(self):
        """Stop playback and delete the player."""
        if hasattr(self, 'player'):
            self.player.stop()
            self.player.deleteLater()
            del self.player
        self.is_playing = False

    def play(self):
        """Start playback by creating a new media player."""
        self.is_playing = True
        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(self.local_file))))
        self.player.play()
        self.player.stateChanged.connect(self.state_changed)

    def state_changed(self, state):
        """Update button icon when playback stops."""
        if state == QMediaPlayer.StoppedState:
            self.pb.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.is_playing = False

    def toggle_play(self):
        """Toggle playback state between playing and stopped."""
        if self.is_playing:
            self.stop()
        else:
            self.play()
            self.pb.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

class WaveformDisplay(QFrame):
    """
    Displays a waveform visualization of audio and a play marker.
    """
    def __init__(self, player, parent=None, height=100):
        super().__init__(parent)
        self.setMinimumHeight(height)
        self.waveform = None
        self.marker_position = 0
        self.duration = 0
        self.player = player

    def load_waveform(self, audio_data, target_num_rects=100):
        """Load audio data and downsample it to render a waveform visualization."""
        samples, sample_rate = sf.read(BytesIO(audio_data))
        self.duration = len(samples) / sample_rate

        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)  # Convert to mono

        downsample_factor = len(samples) // target_num_rects
        downsampled_waveform = [samples[i:i+downsample_factor].max() for i in range(0, len(samples), downsample_factor)]

        self.waveform = ((np.array(downsampled_waveform) - np.min(downsampled_waveform)) /
                         np.ptp(downsampled_waveform) * (self.height() / 2)).astype(int)
        self.update()

    def set_marker_position(self, position):
        """Set the horizontal marker indicating playback position."""
        self.marker_position = position
        self.update()

    def paintEvent(self, event):
        if self.waveform is None:
            return

        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('green'))

        pixel_spacing = 2
        rect_width = (self.width() - pixel_spacing * (len(self.waveform) - 1)) / len(self.waveform)
        middle = self.height() // 2

        for i, amplitude in enumerate(self.waveform):
            rect_height = int(amplitude * 0.2)
            rect_x = int(i * (rect_width + pixel_spacing))
            painter.drawRect(rect_x, middle - rect_height, int(rect_width), rect_height * 2)

        if self.duration > 0:
            marker_x = int(self.marker_position * self.width() / self.duration)
            painter.setBrush(QColor('red'))
            painter.drawRect(marker_x - 1, 0, 2, self.height())

class RichAudioPreviewWidget(QWidget):
    """
    A QWidget that provides rich audio preview functionality, including:
    - Audio playback with play/pause control
    - Visual waveform display of audio file
    - Playhead marker synchronized with audio position
    - Optional drag-and-drop support for exporting the audio file

    This widget is suitable for displaying and interacting with audio data
    in a visually informative and interactive way.
    """
    def __init__(self, button_only=False, drag_enabled=True, pausable=True, height=100):
        """
        Initialize the rich audio preview widget.

        Parameters:
            button_only (bool): If True, hides the waveform and shows only the button.
            drag_enabled (bool): Enables drag-and-drop functionality for the audio file.
            pausable (bool): If False, playback stops on toggle rather than pausing.
        """
        super().__init__()
        hlayout = QHBoxLayout(self)
        frame = QFrame()
        hlayout.addWidget(frame)
        qshrink(hlayout)

        vlayout = QVBoxLayout(frame)
        vlayout.setSpacing(0)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setAlignment(Qt.AlignCenter)
        self.vlayout = vlayout

        self.player = QMediaPlayer()
        self.player.setNotifyInterval(50)

        self.waveform_display = WaveformDisplay(self.player, height=height)
        self.waveform_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        hlayout.addWidget(self.waveform_display)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)
        hlayout.addWidget(self.play_button)

        self.local_file = ""
        self.pausable = pausable
        self.mouse_over = False

        self.player.positionChanged.connect(self.update_marker)
        self.player.stateChanged.connect(self.state_changed)

        self.drag_pixmap = QPixmap("ts_cursor.png")

        if drag_enabled:
            self.play_button.mouseMoveEvent = self.drag_hook
        self.waveform_display.mouseMoveEvent = self.waveform_move_event
        self.waveform_display.mouseClickEvent = self.waveform_move_event

    def enterEvent(self, event):
        """
        Event handler triggered when the mouse enters the widget area.
        Enables key press detection (e.g., spacebar to toggle play).
        """
        self.mouse_over = True
        self.setFocus()

    def leaveEvent(self, event):
        """
        Event handler triggered when the mouse leaves the widget area.
        Disables keyboard focus context.
        """
        self.mouse_over = False

    def keyPressEvent(self, event):
        """
        Handle key press events. Toggles playback with the spacebar
        if the cursor is over the widget.

        Parameters:
            event (QKeyEvent): The key press event.
        """
        if self.mouse_over and event.key() == Qt.Key_Space:
            self.toggle_play()
            event.accept()
        else:
            super().keyPressEvent(event)

    def from_file(self, path):
        """
        Load audio from a file path and display its waveform.

        Parameters:
            path (str): The local path to the audio file.
        """
        self.load_audio_waveform(path)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(path))))
        self.local_file = path

    def from_memory(self, data):
        """
        Load audio from a memory buffer and display its waveform.

        Parameters:
            data (bytes): The raw audio data in memory.
        """
        self.load_audio_waveform(data, from_memory=True)
        self.audio_data = QByteArray(data)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)
        self.player.setMedia(QMediaContent(), self.audio_buffer)

    def load_audio_waveform(self, audio_data, from_memory=False):
        """
        Load and render the waveform for a given audio source.

        Parameters:
            audio_data (bytes or str): Audio data as bytes or file path.
            from_memory (bool): True if audio_data is bytes in memory.
        """
        if from_memory:
            self.waveform_display.load_waveform(audio_data)
        else:
            with open(audio_data, 'rb') as f:
                self.waveform_display.load_waveform(f.read())

    def state_changed(self, state):
        """
        Update the play button icon when the player's state changes.

        Parameters:
            state (QMediaPlayer.State): The new state of the media player.
        """
        if state in [QMediaPlayer.StoppedState, QMediaPlayer.PausedState]:
            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPlay')))

    def toggle_play(self):
        """
        Toggle between playing and paused/stopped state.
        """
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause() if self.pausable else self.player.stop()
        elif self.player.mediaStatus() != QMediaPlayer.NoMedia:
            self.player.play()
            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPause')))

    def seek(self, position):
        """
        Seek to a new position in the audio playback.

        Parameters:
            position (int): Position in milliseconds.
        """
        self.player.setPosition(position)

    def update_marker(self, position):
        """
        Update the position of the playback marker in the waveform.

        Parameters:
            position (int): Current playback position in milliseconds.
        """
        self.waveform_display.set_marker_position(position / 1000)

    def drag_hook(self, e):
        """
        Handle drag event to export audio file from the widget.

        Parameters:
            e (QMouseEvent): The mouse event triggering the drag.
        """
        if e.buttons() != Qt.LeftButton or not len(self.local_file):
            return

        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(os.path.abspath(self.local_file))])
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.setPixmap(self.drag_pixmap)
        drag.exec_(Qt.CopyAction)

    def waveform_move_event(self, event):
        """
        Seek within the waveform when the user clicks/moves mouse over it.

        Parameters:
            event (QMouseEvent): The mouse event on the waveform.
        """
        click_x = event.x()
        width = self.waveform_display.width()
        if click_x < 0 or click_x > width:
            return

        click_position_ratio = click_x / width
        new_position = int(click_position_ratio * float(self.waveform_display.duration) * 1000)
        self.player.setPosition(new_position)
        self.update()
