from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
import torch
from collections import Counter

class Player2TeamAssigner:
    def __init__(self):
        self.referee_color = None

        self.referee_color_idx = None
        self.team1_color_idx = None
        self.team2_color_idx = None

        self.team1_color = None
        self.team2_color = None
        self.counts = None

        self.kmeans = MiniBatchKMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
        self.i = 0
        self.colors = []

    def get_player_color(self, image):

        # lower_green = np.array([36, 25, 25])
        # upper_green = np.array([70, 255, 255])

        # # Create the mask
        # mask = cv2.inRange(cv2.cvtColor(image,cv2.COLOR_RGB2HSV), lower_green, upper_green)
        mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 1],127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        mask_ln = np.where(mask==0)
        mask_ln = len(mask_ln[0])
        if mask_ln/(image.shape[0]*image.shape[1]) > 0.8:
            image = cv2.bitwise_and(image, image, mask = mask)
        else:
            image = cv2.bitwise_and(image, image, mask = cv2.bitwise_not(mask))

        # Reshape the image to a 2D array of pixels and convert to float32
        obj_img = image.reshape(-1, 3).astype(np.float32)

        # Define criteria and apply kmeans
        middle_pixel = image[image.shape[0] // 2, image.shape[1] // 2]  # Middle of the image
        corner_pixel = image[image.shape[0] // 2, 0]  

        # Create centers array (shape: (k, channels))
        centers = np.array([[middle_pixel], [corner_pixel]], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2
        _, labels, centers = cv2.kmeans(obj_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS, centers=())

        # Reshape the labels to the image shape
        clustered_image = labels.reshape((image.shape[0], image.shape[1]))

        # Get the player cluster based on corner pixels
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = centers[player_cluster]
        return player_color
    
    def collect_colors(self, frame, results):
        bxs = results.boxes.xyxy
        labels = results.boxes.cls
        colors = []
        for i, (box, label) in enumerate(zip(bxs, labels)):
            if label == 0 or label == 1:
                continue

            x1, y1, x2, y2= box.tolist()
        
            # Crop object
            img = frame[int(y1):int(y2-((y2-y1)//2)), int(x1):int(x2)]
            color = self.get_player_color(img)
            color = (color[0], color[1], color[2])
            colors.append(color)

            if label == 3: # Referee
                self.referee_color = color
            
        return colors
    
    def fit(self, frame, results, partial_fit=False):
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colors = self.collect_colors(frame, results)
        if partial_fit:
            self.kmeans.partial_fit(np.array(colors).reshape(-1, 3))
        else:
            self.kmeans.fit(np.array(colors).reshape(-1, 3))
        labels = self.kmeans.labels_
        # Count instances per cluster
        self.counts = Counter(labels)
        keys = list(self.counts.keys())
        values = list(self.counts.values())
        self.referee_color_idx = keys[np.argmin(np.array(values))]
        if self.referee_color_idx == 0:
            self.team1_color_idx = 1
            self.team2_color_idx = 2
        elif self.referee_color_idx == 1:
            self.team1_color_idx = 0
            self.team2_color_idx = 2
        else:
            self.team1_color_idx = 0
            self.team2_color_idx = 1
        self.team1_color = tuple(self.kmeans.cluster_centers_[self.team1_color_idx].astype(np.uint8))
        self.team2_color = tuple(self.kmeans.cluster_centers_[self.team2_color_idx].astype(np.uint8))
        self.referee_color = tuple(self.kmeans.cluster_centers_[self.referee_color_idx].astype(np.uint8))

    @torch.inference_mode()
    def predict(self, frame, results):
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bxs = results.boxes.xyxy
        labels = results.boxes.cls
        for i, (box, label) in enumerate(zip(bxs, labels)):
            if label == 0 or label == 1:
                continue

            x1, y1, x2, y2= box.tolist()

            # Crop object
            img = frame[int(y1):int(y2-((y2-y1)//2)), int(x1):int(x2)]
            
            color = self.get_player_color(img)
            color_idx = self.kmeans.predict(np.array(color).reshape(-1, 3).astype(np.float32))[0]
            labels = self.kmeans.labels_
            # Count instances per cluster
            self.counts = Counter(labels)
            
            if color_idx==self.team2_color_idx and label==2.0:
                results.boxes.cls[i] = 4.0

        return results
    
    def fit_predict(self, frame, results, i=1):
        if self.i < i:
            if i == 1:
                self.fit(frame, results)
            else:
                # partial training to make the prediction more stable
                self.fit(frame, results, partial_fit=True)
            self.i += 1
        return self.predict(frame, results)