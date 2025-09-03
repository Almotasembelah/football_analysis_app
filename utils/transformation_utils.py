import numpy as np
import cv2
import torch
import sys
print(sys.path)


class Transform:
    def __init__(self):
        self.M = None
        # Keypoints on Image './input/pitch.jpg' with size (600, 300)
        self.pitch_keypoints = [ # Left Side
           [25,10],[25,70],[25,115],[25,185],[25,230],[25,290], # Goal Line
           [55,115],[55,185],
           [85,150],  # Penalty Spot
           [113,70],[113,115],[113,185],[113,230],
           # Halfway Line
           [300,10],[300,115],[300,185],[300,290],
           # Right Side
           [492,70],[492,115],[492,185],[492,230],
           [520,150], # Penalty Spot
           [550,115],[550,185],           
           [580,10],[580,70],[580,115],[580,185],[580,230],[580,290], # Goal Line

           [252, 150], # Center Circle Left Side
           [348,150], # Center Circle Right Side
        ]

        self.d_kpts = None

    def find_homography(self, detected_keypoints):
        if isinstance(detected_keypoints, torch.Tensor):
            detected_keypoints = detected_keypoints.detach().cpu().numpy()

        pts1, pts2 = [], []
        for p1, p2 in zip(detected_keypoints, self.pitch_keypoints):
            if p1[0]>0 and p1[1]>0:
                pts1.append(list(map(int, p1)))
                pts2.append(p2)
                
        pts1, pts2 = np.float32(pts1), np.float32(pts2)
        self.d_kpts = pts1

        self.M, _ = cv2.findHomography(pts1, pts2)

    def transform(self, points):
        if len(points)==0:
            return []
        
        if isinstance(points, list):
            points = np.array(points)
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        assert self.M is not None, 'Call find_homography'

        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.M)
        points = points.reshape(-1, 2).astype(np.int32)
        return points
    
    def get_feet_position(self, boxes):
        feet = np.zeros(shape=(boxes.shape[0], 2))
        feet[...,0] = boxes[...,0]
        feet[...,1] = boxes[...,1] + (boxes[...,3]//2)
        return feet

    def draw_tactical_board(self, frame, result, team1_color=(0, 0, 255), team2_color=(255, 0, 0)):
        board = cv2.imread('./input/pitch.jpg')
        board = cv2.resize(board, (600, 300))

        bxs = result.boxes.xywh
        mask1 = result.boxes.cls==2
        mask2 = result.boxes.cls==4
        mask3 = result.boxes.cls==1 # Goalkeepers

        team1_pos = bxs[mask1]
        team2_pos = bxs[mask2]
        gks = bxs[mask3]

        team1_pos = self.transform(self.get_feet_position(team1_pos))
        team2_pos = self.transform(self.get_feet_position(team2_pos))
        gks = self.transform(self.get_feet_position(gks))

        for x, y in team1_pos:
            board = cv2.circle(board, (x, y), 10, tuple(map(int,team1_color)), -1)

        for x, y in team2_pos:
            board = cv2.circle(board, (x, y), 10, tuple(map(int,team2_color)), -1)

        for x, y in gks:
            board = cv2.circle(board, (x, y), 10, (0, 255, 255), -1)

        board = cv2.resize(board, (frame.shape[1]//5, frame.shape[0]//5))
        c_img = frame[frame.shape[0]-(board.shape[0])-50:frame.shape[0]-50, 
                      frame.shape[1]//2-(board.shape[1]//2):(frame.shape[1]//2)+(board.shape[1]//2), :]
        board = cv2.addWeighted(c_img, 0.3, board, 0.7, 1)
        frame[frame.shape[0]-(board.shape[0])-50:frame.shape[0]-50, 
              frame.shape[1]//2-(board.shape[1]//2):(frame.shape[1]//2)+(board.shape[1]//2), :] = board
        return frame
    
    

