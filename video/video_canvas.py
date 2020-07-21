import numpy as np
import cv2
from shapely.geometry import Polygon, Point


class DrawObject:
    DRAW_TYPE_RECT = 'rectangle'
    DRAW_TYPE_POLY = 'polygon'

    def __init__(self):
        self.__type = ''
        self.__points = []

    def setType(self, t):
        self.__type = t

    def getType(self):
        return self.__type

    def setPoints(self, points):
        self.__points = points

    def addPoints(self, points):
        self.__points.extend(points)

    def getPoints(self):
        return self.__points

    def move(self, distance_point):
        dx, dy = distance_point
        self.setPoints([p + dx if idx % 2 == 0 else p + dy for idx, p in enumerate(self.__points)])

    def contains(self, point):
        if self.__type == self.DRAW_TYPE_RECT:
            pts = self.getPoints()
            obj = Polygon([(pts[0], pts[1]), (pts[2], pts[1]), (pts[2], pts[3]), (pts[0], pts[3])])
        elif self.__type == self.DRAW_TYPE_POLY:
            pts = np.array(self.getPoints()).reshape(-1,2)
            obj = Polygon(pts)
        pt = Point(point)
        return obj.contains(pt)


class VideoCanvas:
    M_EVT_L_BTN_UP = 'L_BTN_UP'
    M_EVT_L_BTN_DOWN = 'L_BTN_DOWN'
    M_EVT_L_BTN_DBLCLK = 'L_BTN_DBLCLK'
    M_EVT_R_BTN_UP = 'R_BTN_UP'
    M_EVT_R_BTN_DOWN = 'R_BTN_DOWN'
    M_EVT_R_BTN_DBLCLK = 'R_BTN_DBLCLK'
    M_EVT_M_BTN_UP = 'M_BTN_UP'
    M_EVT_M_BTN_DOWN = 'M_BTN_DOWN'
    M_EVT_M_BTN_DBLCLK = 'M_BTN_DBLCLK'
    M_EVT_MOVE = 'MOVE'
    M_EVT_WHEEL = 'WHEEL'
    M_EVT_HWHEEL = 'HWHEEL'

    STATE_NORMAL = 'normal'
    STATE_MOVING = 'moving'
    STATE_CREATING = 'creating'

    DRAW_TYPE_NONE = 'none'
    DRAW_TYPE_RECT = 'rectangle'
    DRAW_TYPE_POLY = 'polygon'
    DRAW_TYPES = [DRAW_TYPE_NONE, DRAW_TYPE_RECT, DRAW_TYPE_POLY]

    OBJ_TYPE = 'type'
    OBJ_POINTS = 'points'

    UI_TYPE_CV = 'cv2'

    def __init__(self, ui_type=UI_TYPE_CV):
        self.event_dic = {}
        self.obj_list = []
        self.image = None
        self.temp_obj = None
        self.selected = -1
        self.select_point = ()
        self.state = self.STATE_NORMAL  # 'normal', 'moving', 'creating'
        self.draw_type = self.DRAW_TYPE_NONE

        if ui_type == self.UI_TYPE_CV:
            self.setMouseEventMapForCv()

    def setDrawType(self, t):
        self.draw_type = t
    
    def drawType(self):
        return self.draw_type

    def insertAnnoObject(self, obj):
        self.obj_list.append(obj)

    def selectObject(self, point):
        # type: (tuple) -> None
        px, py = point
        result = False
        for idx, obj in enumerate(self.obj_list):
            if obj.contains(point):
                self.selected = idx
                result = True
                break
        if not result:
            self.selected = -1
        return result

    def moveObject(self, x, y):
        if self.obj_list:
            self.obj_list[self.selected].move((x, y))

    def removeObject(self, idx):
        if idx != -1:
            self.obj_list.remove(self.obj_list[idx])

    def setMouseEventMapForCv(self):
        self.event_dic[self.M_EVT_L_BTN_UP] = cv2.EVENT_LBUTTONUP
        self.event_dic[self.M_EVT_L_BTN_DOWN] = cv2.EVENT_LBUTTONDOWN
        self.event_dic[self.M_EVT_L_BTN_DBLCLK] = cv2.EVENT_LBUTTONDBLCLK
        self.event_dic[self.M_EVT_R_BTN_UP] = cv2.EVENT_RBUTTONUP
        self.event_dic[self.M_EVT_R_BTN_DOWN] = cv2.EVENT_RBUTTONDOWN
        self.event_dic[self.M_EVT_R_BTN_DBLCLK] = cv2.EVENT_RBUTTONDBLCLK
        self.event_dic[self.M_EVT_M_BTN_UP] = cv2.EVENT_MBUTTONUP
        self.event_dic[self.M_EVT_M_BTN_DOWN] = cv2.EVENT_MBUTTONDOWN
        self.event_dic[self.M_EVT_M_BTN_DBLCLK] = cv2.EVENT_MBUTTONDBLCLK
        self.event_dic[self.M_EVT_MOVE] = cv2.EVENT_MOUSEMOVE
        self.event_dic[self.M_EVT_WHEEL] = cv2.EVENT_MOUSEWHEEL
        self.event_dic[self.M_EVT_HWHEEL] = cv2.EVENT_MOUSEHWHEEL

    def mouseDrawingForCv(self, event, x, y, flags, params):
        if self.draw_type == self.DRAW_TYPE_RECT:
            return self.mouseDrawingRectangle(event, x, y, flags, params)
        elif self.draw_type == self.DRAW_TYPE_POLY:
            return self.mouseDrawingPolygon(event, x, y, flags, params)


    def mouseDrawingRectangle(self, event, x, y, flags, params):
        if event == self.event_dic[self.M_EVT_L_BTN_DOWN]:
            if self.selectObject((x, y)):
                self.state = self.STATE_MOVING
                self.select_point = (x, y)
            else:
                self.state = self.STATE_CREATING
                self.select_point = (x, y)

        if event == self.event_dic[self.M_EVT_MOVE]:
            if self.state == self.STATE_MOVING:
                sx, sy = self.select_point
                dx = x - sx
                dy = y - sy
                self.moveObject(dx, dy)
                self.select_point = (x, y)
            elif self.state == self.STATE_CREATING:
                self.temp_obj = DrawObject()
                x1 = min(self.select_point[0], x)
                y1 = min(self.select_point[1], y)
                x2 = max(self.select_point[0], x)
                y2 = max(self.select_point[1], y)
                self.temp_obj.setType(self.DRAW_TYPE_RECT)
                self.temp_obj.setPoints([x1, y1, x2, y2])

        if event == self.event_dic[self.M_EVT_L_BTN_UP]:
            if self.state == self.STATE_CREATING:
                x1, y1, x2, y2 = self.temp_obj.getPoints()
                if x2 - x1 > 30 and y2 - y1 > 30:
                    self.obj_list.append(self.temp_obj)
                self.temp_obj = None

            self.state = self.STATE_NORMAL

    def mouseDrawingPolygon(self, event, x, y, flags, params):
        if event == self.event_dic[self.M_EVT_L_BTN_DOWN]:
            if self.state == self.STATE_NORMAL:
                if self.selectObject((x, y)):
                    self.state = self.STATE_MOVING
                    self.select_point = (x, y)

        if event == self.event_dic[self.M_EVT_MOVE]:
            if self.state == self.STATE_MOVING:
                sx, sy = self.select_point
                dx = x - sx
                dy = y - sy
                self.moveObject(dx, dy)
                self.select_point = (x, y)

        if event == self.event_dic[self.M_EVT_L_BTN_UP]:
            if self.state == self.STATE_NORMAL:
                self.state = self.STATE_CREATING
                self.select_point = (x, y)
            elif self.state == self.STATE_CREATING:
                if self.temp_obj is None:
                    self.temp_obj = DrawObject()
                    self.temp_obj.setType(self.DRAW_TYPE_POLY)
                    self.temp_obj.addPoints(self.select_point)

                self.select_point = (x, y)
                self.temp_obj.addPoints(self.select_point)

            elif self.state == self.STATE_MOVING:
                self.state = self.STATE_NORMAL

    def keyDrawingForCv(self, key):
        if self.draw_type == self.DRAW_TYPE_RECT:
            return self.keyDrawingRectangle(key)
        elif self.draw_type == self.DRAW_TYPE_POLY:
            return self.keyDrawingPolygon(key)

    def keyDrawingRectangle(self, key):
        pass
    
    def keyDrawingPolygon(self, key):
        if key == 13: # Enter key.
            self.state = self.STATE_NORMAL
            self.obj_list.append(self.temp_obj)
            self.temp_obj = None
        elif key == 255: # Del key.
            if self.state == self.STATE_CREATING:
                self.state = self.STATE_NORMAL
                self.temp_obj = None
                self.select_point = ()
            elif self.state == self.STATE_NORMAL:
                self.removeObject(self.selected)

