import sys
print(sys.path)

import cv2
import numpy as np
from .players_utils import get_last_players
from .line_detection_utils import get_lines, get_intersection


CENTER_LINE = None
CORNER_LINE_DOWN = None
CORNER_LINE_UP = None
CORNER_LINE = None

def draw_annotation(img, results, colors=None, annotate_ball=False):
    
    # Extract tracking data
    result = results
    bxs = result.boxes.xywh 
    cls = result.boxes.cls
    ids = result.boxes.id  

    img = img.copy()
    if colors is None:
        COLORS = {
            1: (0, 255, 255), 
            2: (0, 0, 255),  
            3: (255, 255, 0),  
            4: (255, 0, 0)
        }
    else:
        COLORS = {
            1: (0, 255, 255),  
            2: tuple(map(int,colors[0])),   
            3: (0, 0, 0),  
            4: tuple(map(int,colors[1]))
        }
    BALL_DETECTED = False
    for i, (x, y, w, h) in enumerate(bxs):
        x, y, w, h = int(x.detach().cpu().numpy()), int(y.detach().cpu().numpy()), \
                     int(w.detach().cpu().numpy()), int(h.detach().cpu().numpy())
        if cls[i] == 0 and not BALL_DETECTED and annotate_ball:
            BALL_DETECTED = True
            triangle_points = np.array([
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20]
            ], dtype=np.int32)
            triangle_points = triangle_points.reshape((-1, 1, 2))
            img = cv2.drawContours(img, [triangle_points], 0, (255, 0, 0), cv2.FILLED)
            img = cv2.drawContours(img, [triangle_points], 0, (0, 0, 0), 2)
            continue
        elif cls[i] == 0 and (BALL_DETECTED or not annotate_ball):
            continue
        
        center_x = int(x)
        center_y = int(y + (h / 2))
        cv2.ellipse(
            img=img,
            center=(center_x, center_y),
            axes=(int(w), int(0.35 * w)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=COLORS[int(cls[i].item())],
            thickness=4,
            lineType=cv2.LINE_4
        )
        
        if ids is not None:
            if ids[i] is not None:
                cv2.putText(
                    img,
                    f"ID: {int(ids[i])}",
                    (int(x - w / 2), int(y - h / 2 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
    return img

def draw_ball_annotation(img, results):
    
    img = img.copy()
    if len(results)!=0:
        result = results[0] # draw the ball with highst score
        bbox = result.bbox.to_xywh()
        x, y, w, h = bbox
        x, y = int(x), int(y)
        # x, y, w, h = int(x.detach().cpu().numpy()), int(y.detach().cpu().numpy()), \
        #              int(w.detach().cpu().numpy()), int(h.detach().cpu().numpy())
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ], dtype=np.int32)
        triangle_points = triangle_points.reshape((-1, 1, 2))
        img = cv2.drawContours(img, [triangle_points], 0, (255, 0, 0), cv2.FILLED)
        img = cv2.drawContours(img, [triangle_points], 0, (0, 0, 0), 2)

    return img
    
SIDE1, SIDE2 = None, None
def draw_farthest_players(image, results, draw_field_lines=True, draw_offside_lines=True):
    global SIDE1, SIDE2, CENTER_LINE, CORNER_LINE_DOWN, CORNER_LINE_UP, CORNER_LINE
    result = results
    bxs = result.boxes.xywh
    mask = result.boxes.cls == 2
    bxs1 = bxs[mask]

    mask2 = result.boxes.cls == 4
    bxs2 = bxs[mask2]
    center_line, corner_line_down, corner_line_up, corner_line = get_lines(image)

    CENTER_LINE = center_line if center_line is not None else CENTER_LINE
    CORNER_LINE_DOWN = corner_line_down if corner_line_down is not None else CORNER_LINE_DOWN
    CORNER_LINE_UP = corner_line_up if corner_line_up is not None else CORNER_LINE_UP
    CORNER_LINE = corner_line if corner_line is not None else CORNER_LINE

    players, max_idx, min_idx = get_last_players(bxs1, CENTER_LINE, image.shape[0])
    if players is not None:
        point, point1 = bxs1[max_idx], bxs1[min_idx]

    players2, max_idx2, min_idx2 = get_last_players(bxs2, CENTER_LINE, image.shape[0])
    if players2 is not None:
        point2, point3 = bxs2[max_idx2], bxs2[min_idx2]

    # Draw offside lines if farthest players are detected
    if players is not None and players2 is not None:

        # assign side to each team
        if SIDE1 is None or SIDE2 is None:
            SIDE1 = np.mean(players)
            SIDE2 = np.mean(players2)

        if SIDE1 > SIDE2:
            offside1 = True if players[min_idx] < players2[min_idx2] else False
            offside2 = True if players2[max_idx2] > players[max_idx] else False
        else:
            offside1 = True if players2[max_idx2] < players[max_idx] else False
            offside2 = True if players[min_idx] > players2[min_idx2] else False
            point1, point, min_idx, max_idx = point, point1, max_idx, min_idx
            point2, point3, max_idx2, min_idx2 = point3, point2, min_idx2, max_idx2
        
        cv2.rectangle(image, (int(point1[0] - point1[2]/2), int(point1[1] - point1[3]/2)), 
                      (int(point1[0] + point1[2]/2), int(point1[1] + point1[3]/2)), 
                      (0, 255, 0) if not offside1 else (255, 255, 0), 2)
   
        cv2.rectangle(image, (int(point2[0] - point2[2]/2), int(point2[1] - point2[3]/2)),
                     (int(point2[0] + point2[2]/2), int(point2[1] + point2[3]/2)),
                     (0, 255, 0) if not offside2 else (255, 255, 0), 2)
        
        h = image.shape[0]
        (x1, y1), (x2, y2) = CENTER_LINE
        cm = (y2-y1)/((x2-x1)+1e-9)
        b = y2 - cm*x2
        y = min(y1, y2, get_intersection(CENTER_LINE, CORNER_LINE_UP)[1])-500
        if CENTER_LINE is not None and CORNER_LINE is not None:
            y = get_intersection(CENTER_LINE, CORNER_LINE)[1]
        
        if y is None:
            y = min(y1,y2)
        x = int((y-b)/cm)
        
        if offside1:
            cv2.line(image, (int(players[min_idx]), int(h)), (x, int(y)), (255, 255, 0), 2)
            cv2.line(image, (int(players2[min_idx2]), int(h)), (x, int(y)), (0, 255, 0), 2)

        if offside2:
            cv2.line(image, (int(players[max_idx]), int(h)), (x, int(y)), (0, 255, 0), 2)
            cv2.line(image, (int(players2[max_idx2]), int(h)), (x, int(y)), (255, 255, 0), 2)


    if draw_field_lines:
        image = draw_lines(image, CENTER_LINE, CORNER_LINE_DOWN, CORNER_LINE_UP, CORNER_LINE)
    if draw_offside_lines and players is not None:
        h = image.shape[0]
        (x1, y1), (x2, y2) = CENTER_LINE
        cm = (y2-y1)/((x2-x1)+1e-9)
        b = y2 - cm*x2
        y = min(y1, y2, get_intersection(CENTER_LINE, CORNER_LINE_UP)[1])-1000
        x = int((y-b)/cm)
        cv2.line(image, (int(players[max_idx]), int(h)), (x, int(y)), (0, 255, 255), 4)
        cv2.line(image, (int(players[min_idx]), int(h)), (x, int(y)), (0, 255, 255), 4)
    return image

def draw_lines(image, center_line, corner_line_down, corner_line_up, corner_line):
    # Draw detected lines
    if center_line is not None:
        cv2.line(image, center_line[0], center_line[1], (0, 0, 255), 3)
    if corner_line_down is not None:
        cv2.line(image, corner_line_down[0], corner_line_down[1], (0, 0, 255), 3)
    if corner_line_up is not None:
        cv2.line(image, corner_line_up[0], corner_line_up[1], (0, 0, 255), 3)
    if corner_line is not None:
        cv2.line(image, corner_line[0], corner_line[1], (0, 0, 255), 3)

    intersection1, intersection2 = None, None
    if center_line is not None:
        if corner_line_down is not None:
            intersection1 = get_intersection(center_line, corner_line_down)
            cv2.circle(image, intersection1, 10, (255, 0, 255), -1)
        if corner_line_up is not None:
            intersection2 = get_intersection(center_line, corner_line_up)
            cv2.circle(image, intersection2, 10, (255, 0, 255), -1)
    return image

def draw_keypoints(frame, results):
    points = results.keypoints.xy.squeeze(0)
    for i, point in enumerate(points):
        x, y = point
        if x!=0 and y!=0:
            cv2.rectangle(frame, (int(x) - 10, int(y) - 10),
                  (int(x) + 10, int(y) + 10),
                  (0,255,255), -1)
            frame = cv2.putText(frame, f'{i}', (int(x)-10, int(y)+5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3,)
    return frame