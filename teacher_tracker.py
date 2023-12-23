import logging

import cv2
import numpy as np


class TrackObj:
    def __init__(self):
        pass

    def update(self):
        pass


class TeacherTracker:
    def __init__(self, max_len_frame_buffer=2):
        self.frame_buffer = list()
        self.max_len_frame_buffer = max_len_frame_buffer

    def infer(self, cur_frame):
        cur_frame_buff = cur_frame.copy()
        cur_frame_buff_gray = cv2.cvtColor(cur_frame_buff, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(cur_frame_buff_gray)
        if len(self.frame_buffer) < 2:
            return cur_frame
        self.frame_buffer = self.frame_buffer[-self.max_len_frame_buffer:]  # balance
        frame_absdiff = cv2.absdiff(self.frame_buffer[-1], self.frame_buffer[-2])
        algo_th, frame_absdiff_bin = cv2.threshold(frame_absdiff, 50, 255, cv2.THRESH_BINARY)
        # logging.info(algo_th)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        frame_absdiff_bin_dilate = cv2.dilate(frame_absdiff_bin, kernel, iterations=10)

        _num, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_absdiff_bin_dilate, connectivity=8)
        logging.info(('_num --> ', _num))
        if _num < 2:
            return cur_frame
        color = (0, 0, 255)
        max_area_idx = np.argmax(stats[1:, 4]) + 1
        max_area_rect = stats[max_area_idx, :4]
        cv2.rectangle(cur_frame, max_area_rect, color, 2)
        mask = labels == max_area_idx
        cur_frame[mask] = color
        return cur_frame
