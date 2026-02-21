# SlouchStopper

## Description
SlouchStopper is a real-time posture tracker that monitors neck alignment, gives a posture score, and alerts you if you slouch for too long.

This is for windows
Make sure your webcam is enabled

## Option 1: Run With Python (Requires venv)
1. Clone the repo:

2. Create a virtual environment: python -m venv venv

3. Activate it: venv\Scripts\activate

4. Install dependencies: pip install -r requirements.txt

5. Run the program: python main.py

# Option 2: Build the Windows exe
1. Install pyinstaller: pip install pyinstaller
2. Build the exe: pyinstaller --windowed --collect-all mediapipe --collect-all cv2 --add-data "pose_landmarker_full.task;." main.py
3. The exe will be located in: dist/main/
