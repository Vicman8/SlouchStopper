import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import collections
import winsound
import sys
import os
import tkinter as tk
from PIL import Image, ImageTk

# -----------------------------
# Handle PyInstaller path
# -----------------------------
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "pose_landmarker_full.task")

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# -----------------------------
# Settings
# -----------------------------
SLOUCH_RATIO = 0.5
ALERT_SECONDS = 5
SMOOTHING_FRAMES = 10

upright_time = 0
total_time = 0
slouch_start_time = None
alert_triggered = False

neck_ratio_history = collections.deque(maxlen=SMOOTHING_FRAMES)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# Tkinter Setup (Dark Theme)
# -----------------------------
root = tk.Tk()
root.title("SlouchStopper")
root.geometry("900x700")
root.minsize(600, 400)
root.configure(bg="#1e1e1e")

is_pinned = False
is_paused = False

def toggle_pin():
    global is_pinned
    is_pinned = not is_pinned
    root.attributes("-topmost", is_pinned)

def toggle_pause():
    global is_paused
    is_paused = not is_paused

    if is_paused:
        pause_button.config(text="▶ Resume")
    else:
        pause_button.config(text="⏸ Pause")

# Top bar
top_frame = tk.Frame(root, bg="#1e1e1e")
top_frame.pack(fill="x")

pin_button = tk.Button(
    top_frame,
    text="📌 Pin",
    command=toggle_pin,
    bg="#333333",
    fg="white",
    activebackground="#444444",
    activeforeground="white",
    bd=0,
    padx=10,
    pady=5
)
pin_button.pack(side="right", padx=10, pady=5)

pause_button = tk.Button(
    top_frame,
    text="⏸ Pause",
    command=toggle_pause,
    bg="#333333",
    fg="white",
    activebackground="#444444",
    activeforeground="white",
    bd=0,
    padx=10,
    pady=5
)
pause_button.pack(side="right", padx=5, pady=5)

video_label = tk.Label(root, bg="black")
video_label.pack(fill="both", expand=True)
# -----------------------------
# Main Loop
# -----------------------------
def update_frame():
    global upright_time, total_time
    global slouch_start_time, alert_triggered

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame_start = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    if not is_paused:
        results = pose_landmarker.detect_for_video(
            mp_image,
            int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )
    else:
        results = None

    if results and results.pose_landmarks:
        landmark_list = results.pose_landmarks[0]

        nose = landmark_list[0]
        left_shoulder = landmark_list[11]
        right_shoulder = landmark_list[12]

        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        h, w, _ = frame.shape

        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
        shoulder_x, shoulder_y = int(mid_shoulder_x * w), int(mid_shoulder_y * h)

        neck_length = math.hypot(nose_x - shoulder_x, nose_y - shoulder_y)

        shoulder_width = math.hypot(left_x - right_x, left_y - right_y)

        relative_neck = neck_length / shoulder_width

        neck_ratio_history.append(relative_neck)
        smoothed_neck = sum(neck_ratio_history) / len(neck_ratio_history)

        if smoothed_neck < SLOUCH_RATIO:
            color = (0, 0, 255)
            is_upright = False

            if slouch_start_time is None:
                slouch_start_time = time.time()
            else:
                if time.time() - slouch_start_time >= ALERT_SECONDS and not alert_triggered:
                    alert_triggered = True
                    winsound.Beep(1000, 500)
        else:
            color = (0, 255, 0)
            is_upright = True
            slouch_start_time = None
            alert_triggered = False

        # Draw neck line
        cv2.line(frame, (shoulder_x, shoulder_y), (nose_x, nose_y), color, 5)

        # Draw shoulder dots
        cv2.circle(frame, (left_x, left_y), 8, (255, 0, 0), -1)
        cv2.circle(frame, (right_x, right_y), 8, (255, 0, 0), -1)

        # Red border if slouched too long
        if alert_triggered:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)

        # Timing
        frame_duration = time.time() - frame_start
        total_time += frame_duration
        if is_upright:
            upright_time += frame_duration

        posture_score = (upright_time / total_time) * 100 if total_time > 0 else 0

        # Posture score text
        cv2.putText(frame,
                    f"Posture Score: {posture_score:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        # Neck ratio debug text
        cv2.putText(frame,
                    f"Neck Ratio: {smoothed_neck:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

    # Resize frame to fit window while keeping aspect ratio
    label_width = video_label.winfo_width()
    label_height = video_label.winfo_height()

    if label_width > 1 and label_height > 1:
        h, w, _ = frame.shape
        aspect_ratio = w / h

        if label_width / label_height > aspect_ratio:
            new_height = label_height
            new_width = int(label_height * aspect_ratio)
        else:
            new_width = label_width
            new_height = int(label_width / aspect_ratio)

        resized_frame = cv2.resize(frame, (new_width, new_height))
    else:
        resized_frame = frame

    # Convert for Tkinter
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# -----------------------------
# Clean Exit
# -----------------------------
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

update_frame()
root.mainloop()