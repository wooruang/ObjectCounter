#-*- coding: utf-8 -*-
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from google.protobuf import text_format
from tqdm import tqdm
import cv2
from shapely.geometry import Polygon, Point

# # Make sure that caffe is on the python path:
# caffe_root = '/caffe'  # this file is expected to be in {caffe_root}/examples
# os.chdir(caffe_root)
# sys.path.insert(0, 'python')
import caffe
from caffe.proto import caffe_pb2
caffe.set_device(0)
caffe.set_mode_gpu()


class ObjectDetector:
    OUTPUT_FILE_FORMAT = '.csv'

    def __init__(self, input_path, interval, threshold, zone_info,
                 config='deploy.prototxt',
                 weights='VGG_car5_20190924T065602_SSD_512x512_iter_90000.caffemodel',
                 labelmap='labelmap_car5.prototxt'):
        self.initialize(input_path, interval, threshold,
                        zone_info, config, weights, labelmap)

    def initialize(self, input_path, interval, threshold, zone_info, config, weights, labelmap):
        self.initVariables(input_path, interval, threshold,
                           zone_info, config, weights, labelmap)
        self.initPlt()
        self.initLabelMap()
        self.initCaffeNet()
        self.initSelectedLabelIndexes()
        self.initColor()

    def initVariables(self, input_path, interval, threshold, zone_info, config, weights, labelmap):
        self.config_root = "/Data"
        self.input_path = os.path.join(self.config_root, input_path)
        input_name = os.path.basename(input_path).split('.')[0]
        input_result_name = '{}_result'.format(input_name)
        self.output_dir = os.path.join(self.config_root, input_result_name)
        self.output_video_path = os.path.join(
            self.output_dir, '{}.avi'.format(input_result_name))
        # Make output dir.
        os.mkdir(self.output_dir)

        self.zone_info = zone_info
        self.zone_path = {}
        self.zone_lines = {}

        for k in self.zone_info:
            out_name = '{}_{}{}'.format(input_name, k, self.OUTPUT_FILE_FORMAT)
            out_path = os.path.join(self.output_dir, out_name)
            self.zone_path[k] = out_path
            self.zone_lines = []

        self.labelmap_file = os.path.join(self.config_root, labelmap)
        self.model_config = os.path.join(self.config_root, config)
        self.model_weights = os.path.join(self.config_root, weights)
        self.threshold = threshold
        self.interval = interval
        self.image_resize = 512
        print("Config root        : {}".format(self.config_root))
        print("Input File         : {}".format(self.input_path))
        print("Labelmap File      : {}".format(self.labelmap_file))
        print("Model config File  : {}".format(self.model_config))
        print("Model weights File : {}".format(self.model_weights))
        print("Detect Zone info   : {}".format(self.zone_info))
        print("Detect Zone path   : {}".format(self.zone_path))
        print("Detect threshold   : {}".format(self.threshold))
        print("Detect Interval    : {}".format(self.interval))
        print("Image resize       : {}".format(self.image_resize))

    def initPlt(self):
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

    def initLabelMap(self):
        # load PASCAL VOC labels
        with open(self.labelmap_file, 'r') as f:
            self.labelmap = caffe_pb2.LabelMap()
            text_format.Merge(str(f.read()), self.labelmap)

    def initCaffeNet(self):
        self.net = caffe.Net(self.model_config,      # defines the structure of the model
                             self.model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean(
            'data', np.array([104, 117, 123]))  # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # set net to batch size of 1
        self.net.blobs['data'].reshape(
            8, 3, self.image_resize, self.image_resize)

    def initSelectedLabelIndexes(self):
        self.avaliable_label_indexes = [1, 3, 4, 6, 7, 9, 10, 11]

    def initColor(self):
        self.colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    def getFirstLine(self):
        line = '시간'
        for i in self.avaliable_label_indexes:
            line += ',{}'.format(self.labelmap.item[i].display_name)
        line += '\n'
        return line

    def detectAndParse(self, img):
        transformed_image = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = transformed_image
        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]
        return det_label, det_conf, det_xmin, det_ymin, det_xmax, det_ymax

    def run(self):
        cap = cv2.VideoCapture(self.input_path)

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # interval = fps / 10

        # print(w, h, total_frames, fps, interval)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(
            self.output_video_path, fourcc, fps / self.interval, (int(w), int(h)))

        log_writer = {}
        for k in self.zone_path:
            log_writer[k] = open(self.zone_path[k], 'w')
            log_writer[k].write(self.getFirstLine())

        def isWrong(ww, hh, xxmin, yymin, xxmax, yymax):
            if xxmin > ww or yymin > hh or xxmax < 0 or yymax < 0:
                return True
            CONF_W = 0.6
            CONF_H = 0.6

            limit_w = int(ww * CONF_W)
            limit_h = int(hh * CONF_H)

            o_w = xxmax - xxmin
            o_h = yymax - yymin

            if o_w > limit_w or o_h > limit_h:
                return True

            if o_w <= 20:
                return True
            if o_h <= 20:
                return True

            return False

        for a in tqdm(range(0, int(total_frames)), desc='inferencing'):
            # while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                break

            if a % self.interval == 0:
                overlay = img.copy()

                det_label, det_conf, det_xmin, det_ymin, det_xmax, det_ymax = self.detectAndParse(
                    img)

                # print(det_label)

                for k in self.zone_info:
                    poly = Polygon(self.zone_info[k])

                    cv2.polylines(img, [self.zone_info[k]], True, (255, 0, 0), 2)
                    poly_bound = poly.bounds

                    # cv2.rectangle(img, (int(poly_bound[0]), int(poly_bound[1])), (int(poly_bound[2]), int(poly_bound[3])), (0,255,0), 2)

                    self.drawTextAtBottomRight(img, int(poly_bound[0]), int(poly_bound[1]), k, (255, 0, 0), font_scale=1, thickness=3)



                    exist_objs = []

                    def filtering(idx):
                        xmin, ymin, xmax, ymax = self.filterBbox(
                            det_xmin[idx], det_ymin[idx], det_xmax[idx], det_ymax[idx], w, h)

                        if isWrong(w, h, xmin, ymin, xmax, ymax):
                            return False
                        temp_bbox = [xmin, ymin, xmax, ymax]
                        if temp_bbox in exist_objs:
                            return False
                        exist_objs.append([xmin, ymin, xmax, ymax])
                        return True

                    def intersect(idx):
                        xmin, ymin, xmax, ymax = self.filterBbox(
                            det_xmin[idx], det_ymin[idx], det_xmax[idx], det_ymax[idx], w, h)

                        point = Point(self.getCenterPoint([xmin, ymin, xmax, ymax]))
                        return poly.contains(point)

                    # Get detections with confidence higher than threshold.
                    top_indices = [i for i, conf in enumerate(
                        det_conf) if conf >= self.threshold and filtering(i) and intersect(i)]

                    # print('======')
                    # print(top_indices)

                    top_conf = det_conf[top_indices]
                    top_label_indices = det_label[top_indices].tolist()
                    top_labels = self.get_labelname(
                        self.labelmap, top_label_indices)
                    top_xmin = det_xmin[top_indices]
                    top_ymin = det_ymin[top_indices]
                    top_xmax = det_xmax[top_indices]
                    top_ymax = det_ymax[top_indices]

                    count_labels = [top_label_indices.count(
                        idx) for idx in self.avaliable_label_indexes]
                    
                    log_line = str(a / fps) if a != 0 else str(a)
                    for c in count_labels:
                        log_line += ',{}'.format(c)
                    log_line += '\n'
                    # print(log_line)
                    log_writer[k].write(log_line)

                    for i in range(top_conf.shape[0]):
                        xmin, ymin, xmax, ymax = self.filterBbox(
                            top_xmin[i], top_ymin[i], top_xmax[i], top_ymax[i], w, h)

                        score = top_conf[i]
                        label = int(top_label_indices[i])
                        label_name = top_labels[i]
                        display_txt = '%s: %.2f' % (label_name, score)
                        color = self.colors[label][:-1]
                        color = [int(c * 255) for c in color]

                        self.drawBboxWithoutText(
                            img, xmin, ymin, xmax, ymax, display_txt, color)

                    alpha = 0.5

                    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

                    for i in range(top_conf.shape[0]):
                        xmin, ymin, xmax, ymax = self.filterBbox(
                            top_xmin[i], top_ymin[i], top_xmax[i], top_ymax[i], w, h)

                        score = top_conf[i]
                        label = int(top_label_indices[i])
                        label_name = top_labels[i]
                        display_txt = '%s: %.2f' % (label_name, score)
                        color = self.colors[label][:-1]
                        color = [int(c * 255) for c in color]
                        self.drawBboxWithText(img, xmin, ymin, xmax, ymax, display_txt, color)

                cv2.imshow("Test", img)
                key = cv2.waitKey(0)
                
                if key == ord('a'):
                    exit(1)
                writer.write(img)
        writer.release()
        for k in log_writer:
            log_writer[k].close()

    @staticmethod
    def get_labelname(labelmap, labels):
        num_labels = len(labelmap.item)
        labelnames = []
        if type(labels) is not list:
            labels = [labels]
        for label in labels:
            found = False
            for i in range(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True
        return labelnames

    @staticmethod
    def filterBbox(xmin, ymin, xmax, ymax, w, h):
        rxmin = max(0, int(round(xmin * w)))
        rymin = max(0, int(round(ymin * h)))
        rxmax = min(int(w), int(round(xmax * w)))
        rymax = min(int(h), int(round(ymax * h)))
        return rxmin, rymin, rxmax, rymax

    @staticmethod
    def getCenterPoint(box):
        return (int((box[2] - box[0]) / 2 + box[0]), int((box[3] - box[1]) / 2 + box[1]))

    @staticmethod
    def drawBboxWithoutText(img, xmin, ymin, xmax, ymax, display_txt, color, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55, thickness=2):
        size = cv2.getTextSize(display_txt, font, font_scale, thickness)
        t_w = size[0][0]
        t_h = size[0][1]

        cv2.rectangle(img, (xmin, ymin - t_h - 20),
                      (xmin + t_w + 20, ymin), color, -1)

    @staticmethod
    def drawBboxWithText(img, xmin, ymin, xmax, ymax, display_txt, color, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55, thickness=2):
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        size = cv2.getTextSize(display_txt, font, font_scale, thickness)
        t_w = size[0][0]
        t_h = size[0][1]
        t_color = (0,0,0)

        cv2.rectangle(img, (xmin, ymin - t_h - 20),
                      (xmin + t_w + 20, ymin), t_color, 1)

        cv2.putText(img, display_txt, (xmin + 10, ymin - 10),
                    font, font_scale, t_color, thickness)

    @staticmethod
    def drawTextAtBottomRight(img, x, y, txt, color, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55, thickness=2):
        size = cv2.getTextSize(txt, font, font_scale, thickness)
        t_w = size[0][0]
        t_h = size[0][1]
        x_margin = 20
        y_margin = 10

        cv2.putText(img, txt, (x + x_margin, y + t_h + y_margin),
                    font, font_scale, color, thickness)



