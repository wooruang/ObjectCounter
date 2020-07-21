import os
import time
import numpy as np
import argparse
import cv2
import pickle
from video.video_sync_reader import VideoSyncReader
from video.video_canvas import VideoCanvas


def parseArgs():
    parser = argparse.ArgumentParser(description="Object counter.")
    parser.add_argument("input", help="입력할 비디오")
    parser.add_argument(
        "-s", "--interval_seconds",
        type=float,
        help=
"""딥러닝 처리할 구간.\n
구간을 작게 설정할수록 변환이 오래 걸리는 대신 좀 더 높은 프레임 수의 결과물이 녹화된다.\n
반대로 구간을 크게 설정할수록 변환이 줄어드는 대신 좀 더 낮은 프레임 수의 결과물이 녹화된다.
(가능 초: 1, 0.9, 0.8, 0.7 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)"""
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="딥러닝 확률 임계치"
    )
    parser.add_argument(
        "--skip_zone",
        action="store_true",
        help="이전 영역을 사용."
    )

    args = parser.parse_args()
    return args


def setting_step(args):
    video = VideoSyncReader()
    video.open(args.input)

    fps = video.fps()
    print(fps)

    interval = setup_interval(fps)
    zone_info = setup_zone(video, args.skip_zone)


def setup_interval(fps):
    print("딥러닝 처리할 구간을 선택하십이오.")
    print("다음 중에 골라야 합니다.")
    print("1, 0.9, 0.8, 0.7 0.6, 0.5, 0.4, 0.3, 0.2, 0.1")
    try:
        interval =  float(input("> "))
    except Exception as e:
        print(e)
        print("숫자를 입력해야합니다.")
        exit(1)

    if interval not in [ i / 10 for i in range(1, 10)]:
        print("다음 중에 골라야 합니다.")
        print("1, 0.9, 0.8, 0.7 0.6, 0.5, 0.4, 0.3, 0.2, 0.1")
        exit(1)
    elif fps < 10 and interval <= 0.1:
        print(f"영상의 프레임 수 대비 너무 작은 구간을 선택하셨습니다. (영상 프레임 {fps})")
        exit(1)
    
    return interval

def setup_zone(video, skip_zone):
    WIN_NAME = 'select zone!'
    ZONE_FILE = '/Data/zone.pkl'
    fps = video.fps()

    def make_zone(canv):
        zone_info = {}
        for i, obj in enumerate(canv.obj_list):
            name = f'zone{i}'
            points = np.array(obj.getPoints()).reshape(-1, 2)
            zone_info[name] = points
        return zone_info

    canvas = VideoCanvas()
    canvas.setDrawType(canvas.DRAW_TYPE_POLY)

    if skip_zone:
        if os.path.exists(ZONE_FILE):
            with open(ZONE_FILE, 'r') as f:
                canvas.obj_list = pickle.load(f)

            return make_zone(canvas)
        else:
            print("이전 영역이 존재하지 않습니다.")
            exit(1)

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, canvas.mouseDrawingForCv)

    def drawObject(image, obj, color, thickness=3):
        if obj.getType() == VideoCanvas.DRAW_TYPE_RECT:
            points = obj.getPoints()
            p1 = tuple(points[0:2])
            p2 = tuple(points[2:4])
            return cv2.rectangle(image, p1, p2, color, thickness)
        elif obj.getType() == VideoCanvas.DRAW_TYPE_POLY:
            points = np.array(obj.getPoints()).reshape(-1, 2)
            return cv2.polylines(image, [points], True, color, thickness)

    while True:
        ret, img = video.read()

        for obj in canvas.obj_list:
            img = drawObject(img, obj, (255, 0, 0))
        if canvas.temp_obj is not None:
            drawObject(img, canvas.temp_obj, (0, 255, 0))

        cv2.imshow(WIN_NAME, img)
        key = cv2.waitKey(int(1000 / fps))
        canvas.keyDrawingForCv(key)

        if key == 27:  # 'ESC' key.
            print("프로그램을 종료합니다.")
            exit(1)
            break
        elif key == 255: # Del key.
            canvas.obj_list = []
        elif key == 32:  # 'space' key
            break

    with open(ZONE_FILE, 'w') as f:
        pickle.dump(canvas.obj_list, f)

    return make_zone(canvas)


def main():

    args = parseArgs()

    print(args.input)
    print(args.interval_seconds)

    setting_step(args)

    return
    # url = 'rtsp://admin:1234@192.168.0.71/stream1'
    url = args.input

    app = VideoApp()
    app.addChannel('cam1', url)
    app.addViewer('ch1')
    app.setChannelAtViewer('ch1', 'cam1')
    app.loopForMain()


class VideoChannel:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.reader = self.initReader(self.url)
        self.image = None
        self.times = []

    @staticmethod
    def initReader(url):
        reader = VideoReader()
        reader.open1(url)
        return reader

    def read(self):
        image = self.reader.getFrame()
        if image is not None:
            self.image = image
            current_time = time.time()
            self.times.append(current_time)
            self.times = [x for x in self.times if current_time - x < 1]
            print(len(self.times))


class VideoApp:
    CANVAS_STATE_NONE = 'none'
    CANVAS_STATE_RECT = 'rect'
    CANVAS_STATE_POLY = 'poly'

    def __init__(self):
        self.ch_infos = {}
        self.viewer = VideoViewer()

        self.viewer_ch = {}

        self.initCanvas()
        self.initEvent()

    def addChannel(self, name, url):
        self.ch_infos[name] = VideoChannel(name, url)

    def addViewer(self, name):
        self.viewer_ch[name] = None
        self.viewer.addWindows(name, cv2.WINDOW_KEEPRATIO)

    def setChannelAtViewer(self, viewer_name, ch_name):
        self.viewer_ch[viewer_name] = ch_name
        self.viewer.setMouseEvent(viewer_name, self.canvas.mouseDrawingForCv)

    def initCanvas(self):
        self.canvas = VideoCanvas()
        self.canvas_idx = 0

    def doCanvas(self):
        self.canvas.setDrawType(VideoCanvas.DRAW_TYPE_POLY)

    @staticmethod
    def drawObject(image, obj, color, thickness=3):
        if obj.getType() == VideoCanvas.DRAW_TYPE_RECT:
            points = obj.getPoints()
            p1 = tuple(points[0:2])
            p2 = tuple(points[2:4])
            return cv2.rectangle(image, p1, p2, color, thickness)
        elif obj.getType() == VideoCanvas.DRAW_TYPE_POLY:
            points = np.array(obj.getPoints()).reshape(-1, 2)
            return cv2.polylines(image, [points], True, color, thickness)

    def initEvent(self):
        self.events = {}

    def addEvent(self, event_obj):
        self.events[event_obj.name] = event_obj
        self.selected_event = event_obj.name

    def reloadEvents(self):
        for k in self.event:
            self.event[k].loadSetting()

    def loopForMain(self):
        while True:
            for k in self.ch_infos:
                self.ch_infos[k].read()

            self.doCanvas()

            for k in self.viewer_ch:
                if self.viewer_ch[k] is None:
                    continue

                if self.ch_infos[self.viewer_ch[k]].image is None:
                    continue
                h, w, c = self.ch_infos[self.viewer_ch[k]].image.shape
                if h <= 0 or w <= 0 or c <= 0:
                    continue

                ori_img = self.ch_infos[self.viewer_ch[k]].image

                drawed_img = ori_img.copy()

                # Event.
                drawed_img = self.events[self.selected_event].run(drawed_img)

                # ROI.
                if self.canvas.drawType() != VideoCanvas.DRAW_TYPE_NONE:
                    for obj in self.canvas.obj_list:
                        self.drawObject(drawed_img, obj, (255, 0, 0))

                    if self.canvas.temp_obj is not None:
                        self.drawObject(
                            drawed_img, self.canvas.temp_obj, (0, 255, 0))

                # print(ori_img)
                self.viewer.setImage(k, drawed_img)

            print(self.canvas.obj_list)

            key = self.viewer.showAll(1)
            self.canvas.keyDrawingForCv(key)
            print(key)

            if key == 27:  # 'ESC' key.
                break
            elif key == 93:  # ']' key.
                self.canvas_idx += 1
                if self.canvas_idx >= len(VideoCanvas.DRAW_TYPES):
                    self.canvas_idx = 0
            elif key == 91:  # '[' key.
                self.canvas_idx -= 1
                if self.canvas_idx < 0:
                    self.canvas_idx = len(VideoCanvas.DRAW_TYPES) - 1
            elif key == 114:  # 'r' key.
                self.reloadEvents()
            elif key == 49:  # '1' key.
                self.selected_event = list(self.events.keys())[0]
            elif key == 50:  # '2' key.
                if len(self.events.keys()) > 1:
                    self.selected_event = list(self.events.keys())[1]
            elif key == 51:  # '3' key.
                if len(self.events.keys()) > 2:
                    self.selected_event = list(self.events.keys())[2]
            elif key == 52:  # '4' key.
                if len(self.events.keys()) > 3:
                    self.selected_event = list(self.events.keys())[3]
            elif key == 32:  # 'space' key
                objects = [o.getPoints() for o in self.canvas.obj_list]
                self.events[self.selected_event].setSetting('roi', objects)


if __name__ == "__main__":
    main()
