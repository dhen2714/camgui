import os
import asyncore
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QPushButton, 
    QVBoxLayout, 
    QApplication, 
    QSlider, 
    QLabel, 
    QLineEdit, 
    QFormLayout, 
    QInputDialog, 
    QSlider
)
from PyQt5.QtGui import QImage, QPixmap


class ExposureControlWidget(QWidget):
    """
    Get max exposure, min exposure for given camera
    """
    def __init__(self, camera=None, tick_interval=1):
        super().__init__()
        self.camera = camera
        self.tick_interval = tick_interval
        
        if self.camera:
            self.max = self.camera.max_exposure
            self.min = self.camera.min_exposure
        else:
            self.max = 10
            self.min = 0

        self.label = QLabel()
        self.label.setText('Exposure control:')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.min)
        self.slider.setMaximum(self.max)
        self.slider.setValue(self.max)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(self.tick_interval)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self.change_exposure)

    def change_exposure(self):
        if self.camera:
            exp = self.slider.value()
            self.camera.set_exposure(exp)


class ImageDisplayWidget(QLabel):
    """
    Displays frame from camera stream.
    If no camera connected, displays blank image.
    """
    def __init__(self, camera=None):
        super().__init__()
        self.camera = camera

        if self.camera:
            self.width = self.camera.width
            self.height = self.camera.height
            self.channels = self.camera.channels
            self.bitdepth = self.camera.bitdepth
            self.qformat = self.camera.qformat
        else:
            self.width = 640
            self.height = 480
            self.channels = 1
            self.bitdepth = 8
            self.qformat = QImage.Format_Grayscale8

        nullimg = np.zeros((self.height, self.width, self.channels))
        qimg = QImage(nullimg, self.width, self.height, self.qformat)
        self.setPixmap(QPixmap.fromImage(qimg))

    def update_image(self, qimage):
        self.setPixmap(QPixmap.fromImage(qimage))


class StartWindow(QMainWindow):
    """Main GUI window."""
    def __init__(self, camera=None):
        super().__init__()
        self.camera = camera

        self.central_widget = QWidget()
        self.image_display = ImageDisplayWidget(self.camera)
        # Button to save a single camera image.
        self.button_frame = QPushButton('Acquire single frame', 
                                        self.central_widget)
        # Button to acquire multiple frames.
        self.button_frames = QPushButton('Acquire frames', self.central_widget)
        # Slider for exposure control.
        self.widget_exp = ExposureControlWidget(self.camera)

        # Add save directory option.
        self.widget_savedir = QWidget()
        self.layout_savedir = QFormLayout(self.widget_savedir)
        self.line_edit_savedir = QLineEdit()
        self.label_savedir = QLabel()
        self.button_savedir = QPushButton('Directory for saved images:')
        self.layout_savedir.addRow(self.button_savedir, self.label_savedir)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.button_frame)
        self.layout.addWidget(self.button_frames)
        self.layout.addWidget(self.widget_savedir)
        self.layout.addWidget(self.widget_exp)
        self.layout.addWidget(self.image_display)
        self.setCentralWidget(self.central_widget)

        # Button clicked triggers.
        self.button_frame.clicked.connect(self.save_frame)
        self.button_frames.clicked.connect(self.save_frames)
        self.button_savedir.clicked.connect(self.get_savedir)

        # Thread for displaying camera feed.
        self.movie_thread = MovieThread(self.camera)
        self.movie_thread.send_frame.connect(self.update_frame)
        self.movie_thread.start()

        self.savedir = None # Directory to write images to.
        self.write = False # If writing images to disc.
        self.write_num = 0 # Number of images to write.

        self.frame = None # Current frame as a numpy array
        self.detector = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.total_count = 0

    @pyqtSlot(np.ndarray, int)
    def update_frame(self, frame, timestamp):
        """
        Pulls latest frame from the camera stream, updates image display.
        Writes to disk if option toggled.
        """
        self.frame = frame
        qimage = QImage(frame, self.camera.width, self.camera.height, 
                        self.camera.qformat)
        self.image_display.update_image(qimage)
        
        if self.write and self.write_num:
            outname = str(timestamp) + self.camera.savefmt
            if self.savedir:
                outpath = os.path.join(self.savedir, outname)
            else:
                outpath = outname
            # Color has been converted to RGB on the camera end.
            if self.camera.savefmt == '.png':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if cv2.imwrite(outpath, frame):
                print('Image saved:', outpath)
            else:
                print('Could not save to', outpath)
            self.write_num -= 1
        else:
            self.write = False

        # Report number of features every 100th frame.
        if (self.total_count % 100) == 0:
            c1, c2, cm = self.count_features() # Num keypoints in img1, img2 and matches
            print('\nKeypoints view 1:', c1)
            print('Keypoints view2:', c2)
            print('Matches:', cm)

        self.total_count += 1

    def count_features(self, ratio=0.6):
        """
        Performs sift feature detection, calculates number of matches.
        """
        frame = self.frame
        img1 = frame[:, 640:]
        img2 = frame[:, :640]
        k1, d1 = self.detector.detectAndCompute(img1, None)
        k2, d2 = self.detector.detectAndCompute(img2, None)

        matches = []
        match = self.matcher.knnMatch(d1, d2, k=2)
        for m1, m2 in match:
            if m1.distance < ratio*m2.distance:
                matches.append(m1)

        return len(k1), len(k2), len(matches)

    def update_write(self, write_num, savedir=None):
        """
        Updates number of frames to write, activated by 'write frames' button.
        """
        self.savedir = savedir
        if self.write:
            print('Already writing!')
        else:
            self.write_num = write_num
            self.write = True

    def save_frame(self):
        """Calls update_write() for a single frame."""
        self.update_write(1, self.savedir)

    def save_frames(self):
        """Calss update_write() for multiple frames."""
        numframes, ret = QInputDialog.getInt(self, 'Input', 
            'Enter number of frames to write (>0):')
        if ret and numframes > 0:
            self.update_write(numframes, self.savedir)

    def get_savedir(self):
        """Input for changing the directory to which images are saved."""
        savedir, ret = QInputDialog.getText(self, 'Input', 'Enter path to \
            directory in which images are to be saved:')
        if ret:
            self.savedir = str(savedir)
            self.line_edit_savedir.setText(str(savedir))
            self.label_savedir.setText(str(savedir))

    def change_exposure(self):
        """Triggered by moving exposure control slider."""
        exp = self.widget_exp.slider.value()
        self.camera.set_exposure(exp)

    def closeEvent(self, event):
        """
        Override of closeEvent, closes down all threads.
        """
        self.movie_thread.thread_close()
        event.accept() # Let the window close.


class MovieThread(QThread):
    send_frame = pyqtSignal(np.ndarray, int)
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._running = True

    def run(self):
        frame = np.zeros((480, 1280), dtype=np.uint8)
        while self._running:
            if self.camera:
                frame, t = self.camera.get_frame()

                self.send_frame.emit(frame, t)

    def thread_close(self):
        self._running = False
        if self.camera:
            self.camera.close()
        self.quit()
        self.wait()


if __name__ == '__main__':
    # from cameras import Webcam, LIOV7251Stereo
    # from trackers import GUIStereoTracker, DummyTracker
    # from scanner import UDPConnection
    from camwrappers import LIOV7251MIPI
    cam = LIOV7251MIPI('/dev/video0')
    app = QApplication([])
    # tracker = GUIStereoTracker()
    # udp = UDPConnection()
    # tracker.verbose = False
    window = StartWindow(cam)
    window.show()
    app.exit(app.exec_())