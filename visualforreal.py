import cv2
import mediapipe as mp
import time
from threading import Thread

# 1. Background Camera Stream
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
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# 2. Processor Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def run_vision_system():
    # Standard source initialization
    vs = VideoStream(src=0).start()
    time.sleep(1.0) 
    
    with mp_hands.Hands(
        model_complexity=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        p_time = 0
        
        while True:
            frame = vs.read()
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            
            # Optimization: Non-writeable flag for processing
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            frame.flags.writeable = True

            # Tracking Overlay
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Performance Metrics
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            
            # Discreet UI
            cv2.putText(frame, f"System FPS: {int(fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow("Vision Application", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                vs.stop()
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_system()