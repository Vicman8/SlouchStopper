import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import collections
import winsound  # Windows beep
import sys
import os

if getattr(sys, 'frozen', False):
    # Running in PyInstaller exe
    base_path = sys._MEIPASS
else:
    # Running as normal script
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "pose_landmarker_full.task")
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Posture thresholds
SLOUCH_RATIO = 0.5       # ratio below this = slouch
ALERT_SECONDS = 5         # slouch duration before alert
SMOOTHING_FRAMES = 10      # number of frames to average for smoothing

# Posture tracking variables
upright_time = 0
total_time = 0
slouch_start_time = None
alert_triggered = False

# For smoothing
neck_ratio_history = collections.deque(maxlen=SMOOTHING_FRAMES)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = pose_landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    if results.pose_landmarks:
        landmark_list = results.pose_landmarks[0]

        nose = landmark_list[0]
        left_shoulder = landmark_list[11]
        right_shoulder = landmark_list[12]

        # Midpoint of shoulders
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        h, w, _ = frame.shape

        # Convert normalized coordinates to pixels
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        shoulder_x, shoulder_y = int(mid_shoulder_x * w), int(mid_shoulder_y * h)

        # Neck length
        neck_length = math.hypot(nose_x - shoulder_x, nose_y - shoulder_y)

        # Shoulder width
        shoulder_width = math.hypot(int(left_shoulder.x * w) - int(right_shoulder.x * w),
                                    int(left_shoulder.y * h) - int(right_shoulder.y * h))

        # Relative neck ratio
        relative_neck = neck_length / shoulder_width

        # Add to history for smoothing
        neck_ratio_history.append(relative_neck)
        smoothed_neck = sum(neck_ratio_history) / len(neck_ratio_history)

        # Determine posture using smoothed value
        if smoothed_neck < SLOUCH_RATIO:
            color = (0, 0, 255)  # Red = slouch
            is_upright = False

            # Start slouch timer if not already started
            if slouch_start_time is None:
                slouch_start_time = time.time()
            else:
                # Trigger alert if slouched too long
                if time.time() - slouch_start_time >= ALERT_SECONDS and not alert_triggered:
                    alert_triggered = True
                    # Visual alert: thick red border
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
                    # Beep alert (Windows)
                    winsound.Beep(1000, 500)
        else:
            color = (0, 255, 0)  # Green = upright
            is_upright = True
            slouch_start_time = None
            alert_triggered = False

        # Draw neck line
        cv2.line(frame, (shoulder_x, shoulder_y), (nose_x, nose_y), color, 5)

        # Draw shoulders
        cv2.circle(frame, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 8, (255, 0, 0), -1)
        cv2.circle(frame, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 8, (255, 0, 0), -1)

        # Posture scoring
        frame_duration = time.time() - frame_start
        total_time += frame_duration
        if is_upright:
            upright_time += frame_duration

        posture_score = (upright_time / total_time) * 100

        # Display posture score
        cv2.putText(frame, f"Posture Score: {posture_score:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show smoothed neck ratio for debugging
        cv2.putText(frame, f"Neck ratio: {smoothed_neck:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("SlouchStopper", frame)

    key = cv2.waitKey(1) & 0xFF

    if cv2.getWindowProperty("SlouchStopper", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
