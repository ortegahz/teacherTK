import logging


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def iou_calc(rect_a, rect_b):
    x1_a, y1_a, x2_a, y2_a = rect_a[0], rect_a[1], rect_a[0] + rect_a[2], rect_a[1] + rect_a[3]
    x1_b, y1_b, x2_b, y2_b = rect_b[0], rect_b[1], rect_b[0] + rect_b[2], rect_b[1] + rect_b[3]

    x_min, y_min = min(x1_a, x1_b), min(y1_a, y1_b)
    x_max, y_max = max(x2_a, x2_b), max(y2_a, y2_b)

    if x_max - x_min > rect_a[2] + rect_b[2] or y_max - y_min > rect_a[3] + rect_b[3]:
        return 0.0

    a_area = rect_a[2] * rect_a[3]
    b_area = rect_b[2] * rect_b[3]
    inter_area = (min(x2_a, x2_b) - max(x1_a, x1_b)) * (min(y2_a, y2_b) - max(y1_a, y1_b))
    iou = inter_area / (a_area + b_area - inter_area)
    return iou
