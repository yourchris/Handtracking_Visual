import cv2
import mediapipe as mp

print(f"OpenCV Version: {cv2.__version__}")
print(f"MediaPipe Version: {mp.__version__}")

# Check if OpenCV can see your MacBook's camera
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera is accessible!")
    cap.release()
else:
    print("❌ Camera not found (Check Privacy/Permissions in Sequoia)")