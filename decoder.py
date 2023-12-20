import logging
import time

import cv2


def process_decoder(path_video, queue, buff_len=5):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        logging.error('cap is not opened !')

    t_last = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error('cap read failed !')

        idx_frame += 1
        if queue.qsize() > buff_len:
            queue.get()
            logging.info('dropping frame !')
        queue.put([idx_frame, frame, fc, fps, h, w])

        while time.time() - t_last < 1. / fps:
            time.sleep(0.001)
        t_last = time.time()

    cap.release()
