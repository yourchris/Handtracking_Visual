import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- 3D Projection Logic ---
def get_cube_points(center, size, angle_x, angle_y):
    s = size / 2
    # 8 vertices of a cube
    points = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ])

    # Rotation Matrices
    rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    rotated_points = points @ rx @ ry
    
    projected = []
    for p in rotated_points:
        x = int(p[0] + center[0])
        y = int(p[1] + center[1])
        projected.append((x, y))
    return projected

def draw_cube(img, points, color=(255, 255, 255)):
    # Connect the 12 edges
    for i in range(4):
        cv2.line(img, points[i], points[(i+1)%4], color, 2)
        cv2.line(img, points[i+4], points[((i+1)%4)+4], color, 2)
        cv2.line(img, points[i], points[i+4], color, 2)

# --- Main Vision System ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def run_3d_controller():
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    
    with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7) as hands:
        angle_x, angle_y = 0, 0
        cube_size = 150
        
        while True:
            frame = vs.read()
            if frame is None: continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Focus only on ONE hand to avoid the "sticky cube" on both hands
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0] # Take only the first hand
                
                # 1. Get Landmarks
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                index_mcp = hand_landmarks.landmark[5] # Knuckle for anchoring
                
                # 2. Calculate Pinch Distance for "Zoom"
                # (Euclidean distance between thumb and index)
                dist = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
                
                # If pinched, adjust cube size; otherwise, just rotate
                cube_color = (255, 255, 255)
                if dist < 0.05:
                    cube_color = (0, 255, 0) # Turn green when scaling
                    cube_size = int(dist * 3000) # Scale based on distance
                
                # 3. Update Rotation based on hand position
                angle_y = (index_mcp.x - 0.5) * np.pi * 2
                angle_x = (index_mcp.y - 0.5) * np.pi * 2
                
                # 4. Render Tracking and Cube
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                finger_pos = (int(index_mcp.x * w), int(index_mcp.y * h))
                cube_pts = get_cube_points(finger_pos, max(50, cube_size), angle_x, angle_y)
                draw_cube(frame, cube_pts, color=cube_color)

            cv2.putText(frame, "Pinch to Scale | Move to Rotate", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            cv2.imshow("3D Object Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                vs.stop()
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_3d_controller()