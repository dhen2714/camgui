import cv2
from PyQt5.QtGui import QImage
import numpy as np
from pyv4l2.camera import Camera
from pyv4l2.camera import MIPI_OV7251


class Webcam:
    def __init__(self, cid=0):
        self.cid = cid
        self.camera = cv2.VideoCapture(self.cid)

        ret, frame = self.camera.read()

        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.channels = frame.shape[2]
        self.bitdepth = frame.nbytes/(self.height*self.width*self.channels)

        self.nullimg = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        self.qformat = QImage.Format_RGB888
        self.frame_count = 0

        self.max_exposure = 10
        self.min_exposure = 0

        self.savefmt = '.png'

    def get_frame(self):
        ret, frame = self.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_count += 1
        return frame, self.frame_count

    def set_exposure(self):
        pass

    def close(self):
        self.camera.release()


class LIOV7251Stereo(Camera):
    def __init__(self, device_path):
        super().__init__()

        self.qformat = QImage.Format_Grayscale8
        self.max_exposure = 31
        self.min_exposure = 0
        self.channels = 1
        self.bitdepth = 8

        self.savefmt = '.pgm'


class LIOV7251MIPI(MIPI_OV7251):
    def __init__(self, device_path):
        super().__init__()

        self.qformat = QImage.Format_Grayscale8
        self.max_exposure = 20000
        self.min_exposure = 0
        self.channels = 1
        self.bitdepth = 8

        self.savefmt = '.pgm'


if __name__ == '__main__':
    cam = LIOV7251MIPI('/dev/video0')
    print(cam.width)
    for i in range(500):
        frame, t = cam.get_frame()
        print(t)
    cam.close()