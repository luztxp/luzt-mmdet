# -*- coding: utf-8 -*-
import os
import torch
import json
import codecs
import numpy as np
from PIL import Image
from collections import OrderedDict

from mmdet.apis import init_detector, inference_detector

from model_service.pytorch_model_service import PTServingBaseService

import time
from metric.metrics_manager import MetricsManager
import log
logger = log.getLogger(__name__)


class ObjectDetectionService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        config_file = os.path.join(os.path.dirname(__file__), 'config.py')
        checkpoint_file = os.path.join(os.path.dirname(__file__), 'model.pth')
        self.model = init_detector(config_file, checkpoint_file, device='cpu:0')

        # these three parameters are no need to modify
        self.input_image_key = 'images'
        self.class_names = ['red_stop', 'green_go', 'yellow_back', 'pedestrian_crossing', 'speed_limited', 'speed_unlimited']
        print('load weights file success')


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with codecs.open(classes_path, 'r', 'utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                preprocessed_data[k] = image
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        image = data[self.input_image_key]
        b, g, r = image.split()
        im = Image.merge("RGB", (r, g, b))
        result = inference_detector(self.model, np.array(im))
        bbox_result = result
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        inds = np.where(bboxes[:, -1] > 0.00001)[0]

        result = OrderedDict()
        if len(inds) > 0:
            detection_class_names = []
            out_boxes_list = []
            out_scores = []
            for ind in inds:
                class_id = labels[ind]
                class_name = self.class_names[int(class_id)]
                #  if class_name in ['red_stop', 'green_go', 'yellow_back']:
                #      continue
                detection_class_names.append(class_name)
                box = bboxes[ind][:4]
                score = bboxes[ind][4]
                box = [box[1], box[0], box[3], box[2]]
                out_boxes_list.append([round(float(v), 1) for v in box])  # v是np.float32类型，会导致无法json序列化，因此使用float(v)转为python内置float类型
                out_scores.append(score)
            result['detection_classes'] = detection_class_names
            result['detection_scores'] = [round(float(v), 4) for v in out_scores]
            result['detection_boxes'] = out_boxes_list
        else:
            result['detection_classes'] = []
            result['detection_scores'] = []
            result['detection_boxes'] = []

        return result


    def _postprocess(self, data):
        return data

    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        #  data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data


def parse_classify_rule(json_path=''):
    with codecs.open(json_path, 'r', 'utf-8') as f:
        rule = json.load(f)
    label_map = {}
    for super_label, labels in rule.items():
        for label in labels:
            label_map[label] = super_label
    return label_map
