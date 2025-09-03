import cv2
import numpy as np
import sys
print(sys.path)


def get_lines(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_white = np.array([0, 0, 110]) # 168
    upper_white = np.array([172, 111, 255])
    mask = cv2.inRange(image_hsv, lower_white, upper_white)
    # edges = cv2.Canny(mask, 50, 200)
    h, w, _ = image.shape
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 60, maxLineGap=30)

    # Filter the lines
    center_line = None
    down_touch_line = None
    up_touch_line = None
    penalty_line = None
    length, length2 = 0, 0
    width1, width2 = 0, 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (y1 <= h // 2 <= y2) or (y2 <= h // 2 <= y1) or (abs(y1-y2) > abs(x1-x2)):
                # keep the longest vertical line (halfway Line)
                if abs(y2 - y1) > length:
                    length = abs(y2 - y1)
                    center_line = [(x1, y1), (x2, y2)]
                # Second longest vertical line (penalty area line)
                if abs(y2-y1) > length2 and abs(center_line[0][0]-x1) > 100 and abs(y1-y2) > abs(x1-x2):
                    length2 = abs(y2-y1)
                    penalty_line = [(x1, y1), (x2, y2)]
            # up touch line
            if (x2-x1 > y2-y1) and (y1 >= h // 2 <= y2):
                if abs(x2 - x1) > width1:
                    width1 = abs(x2 - x1)
                    down_touch_line = [(x1, y1), (x2, y2)]
            # down touch line
            if (x2-x1 > y2-y1) and (y1 <= h // 2 >= y2):
                if abs(x2 - x1) > width2:
                    width2 = abs(x2 - x1)
                    up_touch_line = [(x1, y1), (x2, y2)]

    return center_line, down_touch_line, up_touch_line, penalty_line

def get_intersection(line1, line2):
    '''Find the intersection point of two lines.'''
    (x1, y1), (x2, y2) = line1
    m = (y2-y1)/(x2-x1+1e-9)
    b = y2 - m*x2

    (l_x1, l_y1), (l_x2, l_y2) = line2
    m2 = (l_y2-l_y1)/(l_x2-l_x1+1e-9)
    b2 = l_y2 - m2*l_x2

    if m == m2:
        return (None, None)
    inter_x = (b-b2) / (m2-m)
    inter_y = m*inter_x + b
    return (int(inter_x), int(inter_y))