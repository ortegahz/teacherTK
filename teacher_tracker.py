import cv2


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
        self.frame_buffer = self.frame_buffer[-self.max_len_frame_buffer:]
        frame_absdiff = cv2.absdiff(self.frame_buffer[-1], self.frame_buffer[-2]) if len(
            self.frame_buffer) >= 2 else cur_frame_buff_gray
        mask = frame_absdiff > 10
        cur_frame[mask] = (0, 255, 255)
        return cur_frame
