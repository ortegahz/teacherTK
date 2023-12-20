import argparse
import logging
from multiprocessing import Process, Queue

import cv2

from decoder import process_decoder
from utils import set_logging
from teacher_tracker import TeacherTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_video_in',
                        default='/media/manu/data/videos/vlc-record-2023-12-18-10h54m21s-rtsp___192.168.3.186_554_ch0_1-.mp4')
    parser.add_argument('--window_name', default='results')
    return parser.parse_args()


def run(args):
    ttracker = TeacherTracker()
    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(args.path_video_in, q_decoder), daemon=True)
    p_decoder.start()
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    while True:
        idx_frame, frame, fc, fps, h, w = q_decoder.get()

        frame = ttracker.infer(frame)
        cv2.putText(frame, f'{idx_frame} / {fc}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow(args.window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or (idx_frame > fc - 5 and fc > 0):
            break
    cv2.destroyAllWindows()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
