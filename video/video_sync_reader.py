import cv2
import numpy as np
import multiprocessing as mp


class VideoSyncReader:

    def __init__(self):
        self.url = ''
        self.cap = None

    def open(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        self.init()

    def reopen(self):
        self.close()
        self.open(self.url)

    def close(self):
        self.cap.release()

    def init(self):
        self.fail_count = 0

    def read(self):
        if not self.isOpened():
            self.reopen()
            return False, np.zeros()
        ret, img = self.cap.read()
        if not ret:
            self.fail_count += 1
            if self.fail_count > 10:
                self.reopen()
        return ret, img

    def setBufferSize(self, size):
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def fps(self):
        if not self.isOpened():
            return -1
        return self.cap.get(cv2.CAP_PROP_FPS)

    def width(self):
        if not self.isOpened():
            return 0
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def height(self):
        if not self.isOpened():
            return 0
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def isOpened(self):
        if self.cap is None:
            return False
        return self.cap.isOpened()

    def find_available_interval(self):
        pass

    def __del__(self):
        self.close()

