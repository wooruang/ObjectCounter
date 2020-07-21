import cv2


class VideoViewer:
    V_NAME = 'name'
    V_CACHE = 'cache'
    v_FLAGS = 'flags'

    def __init__(self):
        self.viewers = []
        self.delay = 1

    # def initWindows(self, names):
    #     self.destroyWindows()
    #     self.win_names = names
    #     for n in self.win_names:
    #         cv2.namedWindow(n, cv2.WINDOW_KEEPRATIO)

    def addWindows(self, name, flags):
        for v in self.viewers:
            if v.name == name:
                return
        cv2.namedWindow(name, flags)
        view = {}
        view[self.V_NAME] = name
        view[self.V_CACHE] = None
        view[self.v_FLAGS] = flags
        self.viewers.append(view)

    def destroyWindows(self):
        for v in self.viewers:
            cv2.destroyWindow(v[self.V_NAME])

    def setImage(self, name, image):
        result = False
        for v in self.viewers:
            if v[self.V_NAME] == name:
                result = True
                v[self.V_CACHE] = image
                break
        return result

    def setDelay(self, delay):
        self.delay = delay

    def showAll(self, delay):
        for v in self.viewers:
            if v[self.V_CACHE] is None:
                continue
            h, w, c = v[self.V_CACHE].shape
            if h <= 0 or w <= 0 or c <= 0:
                continue
            cv2.imshow(v[self.V_NAME], v[self.V_CACHE])

        return cv2.waitKey(delay)

    def setMouseEvent(self, name, cb):
        cv2.setMouseCallback(name, cb)

    def __del__(self):
        self.destroyWindows()
