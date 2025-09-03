import numpy as np
import torch
import numpy as np
import torch
from collections import defaultdict

class ImprovedPlayers:
    """
    This class stabilizes player tracking by reassigning "lost" player IDs 
    to new detections based on proximity, ensuring consistent ID usage 
    throughout a game or match.

    Since there are only 22 players on the field, any detected ID greater 
    than 22 (24 in this implementation as a safety margin) indicates that 
    an existing ID has been lost. In such cases, the new ID is reassigned 
    to the lost one. If multiple IDs are lost, the closest match based 
    on position is selected.
    """
    def __init__(self, transform=None):
        self.ids = set()
        self.all_ids = {i+1 for i in range(23)}
        self.snapshots = []
        self.last_positions = {}  # Track last known center positions of each ID
        self.positions = defaultdict(list)
        self.frame_shape = None
        self.transform = transform
        self.map_ids = {} # Map newly detected IDs (greater than 24) to the nearest lost ID

    def _get_center(self, box):
        """Compute center point of a bounding box."""
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @torch.inference_mode()
    def _assign_id(self, current_id, box, ids, result):
        """Reassigns a high ID to a lost one based on distance matching."""
        cx, cy = self._get_center(box)
        lost_ids = self.all_ids - set(ids.tolist())  # IDs that disappeared
        
        if not lost_ids:
            return  # Nothing to reassign

        # Compute distances from this detection to all lost IDs
        distances = [
            (lid, np.linalg.norm(abs(np.array(self.last_positions.get(lid, (0, 0))) - (np.array([cx, cy]) if self.transform is None else self.transform.transform([cx, cy])[0]))))
            for lid in lost_ids
        ]
        distances = [d for d in distances if d[1] is not None]

        if distances:
            best_id = min(distances, key=lambda x: x[1])[0]
            idx = ids.tolist().index(current_id.item())
            result.boxes.id[idx] = best_id
            # assign the new high id with the lost ID
            self.map_ids[current_id.item()] = best_id
            # update the position for the lost ID
            self.positions[best_id].append((cx, cy) if self.transform is None else tuple(i for i in self.transform.transform([cx, cy])[0]))

        return result
    
    @torch.inference_mode()
    def fit(self, frame, result):
        if self.frame_shape is None:
            self.frame_shape = frame.shape[:2]

        ids = result.boxes.id
        bxs = result.boxes.xyxy

        mask = result.boxes.cls == 2

        bxs_masked = bxs[mask]
        ids_masked = ids[mask]
        for id_, box in zip(ids_masked, bxs_masked):
            box = box.tolist()
            cx, cy = self._get_center(box)

            if id_.item() in self.ids:
                # Known player: just update position
                self.last_positions[id_.item()] = (cx, cy) if self.transform is None else tuple(i for i in self.transform.transform([cx, cy])[0])
                self.positions[id_.item()].append((cx, cy) if self.transform is None else tuple(i for i in self.transform.transform([cx, cy])[0]))
                continue

            if id_.item() <= 24:
                # New valid ID: register it
                x1, y1, x2, y2 = map(int, box)
                self.ids.add(id_.item())
                self.snapshots.append(frame[y1:y2, x1:x2])
                self.last_positions[id_.item()] = (cx, cy) if self.transform is None else tuple(i for i in self.transform.transform([cx, cy])[0])
                self.positions[id_.item()].append((cx, cy) if self.transform is None else tuple(i for i in self.transform.transform([cx, cy])[0]))
            else:
                # High ID:
                # if old 
                if self.map_ids.get(id_.item(), 25)<=24:
                    # update the position
                    self.last_positions[id_.item()] = (cx, cy) if self.transform is None else tuple(i for i in self.transform.transform([cx, cy])[0])
                    self.positions[id_.item()].append((cx, cy) if self.transform is None else tuple(i for i in self.transform.transform([cx, cy])[0]))
                    # Change the ID in result for the annotation
                    idx = ids.tolist().index(id_.item())
                    result.boxes.id[idx] = self.map_ids[id_.item()] 
                else:
                    # if new
                    self._assign_id(id_, box, ids, result)

        return result

    def get_heatmap(self, player_id, bins=100):
        """Generate a heatmap for a single player."""
        pos = self.positions.get(player_id, [])
        if not pos:
            print(f"No positions for player {player_id}")
            return None
        
        # Convert to numpy arrays
        x = [p[0] for p in pos]
        y = [p[1] for p in pos]
        
        # Create 2D histogram
        heatmap, _, _ = np.histogram2d(x, y, bins=bins,
                                                range=[[0, 600], [0, 300]])
        
        # Normalize
        heatmap = heatmap.T  # transpose for correct orientation
        heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

        return heatmap

def get_last_players(boxes, center_line, h):
    """
    Get the farthest and nearest players relative to their goalkeeper.

    This function helps determine whether the farthest player 
    might be in an offside position.
    """
    (x1, y1), (x2, y2) = center_line
    cm = (y2-y1)/((x2-x1)+1e-9)
    b = y2 - cm*x2
    y = min(y1, y2)-1000
    x = int((y-b)/cm)
    players = []
    for box in boxes:
        x1, y1, _, _ = box
        m = (y1-y)/(x1-x+1e-9)
        b1 = y1 - m*x1
        x2 = int((h - b1)/m)
        players.append(x2)
    
    if len(players) == 0:
        return None, None, None
    idx_max = np.argmax(np.array(players))
    idx_min = np.argmin(np.array(players))
    return players, idx_max, idx_min