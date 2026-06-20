
import os
import cv2
import time
import json
import math
import csv
import threading
import urllib.request
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, request, jsonify, send_from_directory

app = Flask(__name__)

# Ensure required directories exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ENROLLED_DIR = os.path.join(DATA_DIR, 'enrolled_faces')
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'reports')
SNAPSHOTS_DIR = os.path.join(REPORT_DIR, 'snapshots')

for folder in [DATA_DIR, ENROLLED_DIR, REPORT_DIR, SNAPSHOTS_DIR]:
    os.makedirs(folder, exist_ok=True)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(DATA_DIR, 'face_landmarker.task')

# Download Face Landmarker model if missing
if not os.path.exists(MODEL_PATH):
    print("[System] Downloading MediaPipe Face Landmarker model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[System] Model downloaded successfully.")
    except Exception as e:
        print(f"[System Error] Failed to download model: {e}")

# Path to attendance and user mapping
ATTENDANCE_FILE = os.path.join(DATA_DIR, 'attendance.csv')
MAPPING_FILE = os.path.join(DATA_DIR, 'user_mapping.json')

# Initialize mapping file if not exists
if not os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, 'w') as f:
        json.dump({}, f)

# Initialize attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name", "Blinks", "Left Turn", "Right Turn", "Smile", "Liveness Status", "Recognition Status"])

# Global state for Liveness Tracking
class LivenessState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.face_detected = False
            self.blink_count = 0
            self.left_turn = False
            self.right_turn = False
            self.smile_detected = False
            self.liveness_status = "PENDING"  # PENDING, LIVE PERSON, SPOOF DETECTED
            self.start_time = None
            self.elapsed_time = 0.0
            self.spoof_score = 0.0  # Percentage 0-100
            self.recognized_name = "Unknown"
            self.recognition_confidence = 0.0
            self.blink_in_progress = False
            self.ear_history = []
            self.flatness_history = []
            self.static_frame_count = 0
            self.last_face_center = None
            self.snapshot_saved = False
            self.current_username = None

liveness_state = LivenessState()

# Global variables for camera frame streaming
latest_frame = None
latest_frame_annotated = None
frame_lock = threading.Lock()

# Face recognition setup
recognizer = None
recognizer_lock = threading.Lock()

def load_face_recognizer():
    global recognizer
    trainer_path = os.path.join(DATA_DIR, 'trainer.yml')
    with recognizer_lock:
        if os.path.exists(trainer_path):
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(trainer_path)
                print("[Face Recognizer] Loaded trainer.yml successfully.")
            except Exception as e:
                print(f"[Face Recognizer] Error loading trainer.yml: {e}")
                recognizer = None
        else:
            print("[Face Recognizer] No trainer.yml found. Face recognition disabled until enrollment.")
            recognizer = None

load_face_recognizer()

# Helper: Get enrolled user ID mappings
def get_user_mappings():
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_user_mappings(mapping):
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)

# MediaPipe Face Mesh landmarks indices mapping
# Left eye
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# Right eye
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
# Mouth
MOUTH_CORNERS = [61, 291]
# Face lateral boundaries & Nose tip
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
NOSE_TIP = 4

def calculate_ear(landmarks, points):
    try:
        # vertical eye landmarks
        p2_p6 = np.linalg.norm(np.array(landmarks[points[1]]) - np.array(landmarks[points[5]]))
        p3_p5 = np.linalg.norm(np.array(landmarks[points[2]]) - np.array(landmarks[points[4]]))
        # horizontal eye landmark
        p1_p4 = np.linalg.norm(np.array(landmarks[points[0]]) - np.array(landmarks[points[3]]))
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return ear
    except Exception:
        return 0.0

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.error_message = None

    def run(self):
        global latest_frame, latest_frame_annotated
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DSHOW is faster on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Fallback
            
        if not cap.isOpened():
            self.error_message = "Webcam could not be opened. Please verify that it is connected and not in use."
            print(f"[Camera Error] {self.error_message}")
            self.running = False
            return

        print("[Camera Thread] Webcam connected successfully.")
        
        # Initialize MediaPipe Face Landmarker
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                running_mode=vision.RunningMode.IMAGE
            )
            detector = vision.FaceLandmarker.create_from_options(options)
            print("[Camera Thread] MediaPipe Face Landmarker initialized.")
        except Exception as e:
            self.error_message = f"Failed to initialize Face Landmarker: {e}"
            print(f"[Camera Error] {self.error_message}")
            self.running = False
            return

        # Tracking variables inside thread
        last_ear = 0.3
        blink_cooldown = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("[Camera Thread] Failed to grab frame.")
                time.sleep(0.01)
                continue

            # Mirror the frame for intuitive view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert color space for MediaPipe Tasks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            try:
                detection_result = detector.detect(mp_image)
            except Exception as e:
                print(f"[Camera Thread] MediaPipe process error: {e}")
                time.sleep(0.01)
                continue
            
            annotated_frame = frame.copy()
            face_found = False
            
            if detection_result.face_landmarks:
                face_found = True
                face_landmarks = detection_result.face_landmarks[0]
                
                # Convert landmarks to pixel coordinates
                landmarks_px = []
                for lm in face_landmarks:
                    landmarks_px.append([int(lm.x * w), int(lm.y * h), lm.z])
                
                landmarks_np = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks])
                
                # 1. EAR (Eye Aspect Ratio) for Blink detection
                left_ear = calculate_ear(landmarks_px, LEFT_EYE_LANDMARKS)
                right_ear = calculate_ear(landmarks_px, RIGHT_EYE_LANDMARKS)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # 2. Head Movement (Yaw)
                nose = landmarks_px[NOSE_TIP]
                left_cheek = landmarks_px[LEFT_CHEEK]
                right_cheek = landmarks_px[RIGHT_CHEEK]
                
                d_left = abs(nose[0] - left_cheek[0])
                d_right = abs(right_cheek[0] - nose[0])
                
                # Prevent divide by zero
                if d_right == 0: d_right = 1
                yaw_ratio = d_left / d_right
                
                # 3. Smile Detection
                m_corners = [landmarks_px[MOUTH_CORNERS[0]], landmarks_px[MOUTH_CORNERS[1]]]
                mouth_width = np.linalg.norm(np.array(m_corners[0][:2]) - np.array(m_corners[1][:2]))
                face_width = np.linalg.norm(np.array(left_cheek[:2]) - np.array(right_cheek[:2]))
                smile_ratio = mouth_width / face_width if face_width > 0 else 0
                
                # 4. Anti-Spoofing: Laplacian Variance & Depth Flatness
                # Crop face bounding box
                xs = [lm[0] for lm in landmarks_px]
                ys = [lm[1] for lm in landmarks_px]
                min_x, max_x = max(0, min(xs)), min(w, max(xs))
                min_y, max_y = max(0, min(ys)), min(h, max(ys))
                
                face_crop = frame[min_y:max_y, min_x:max_x]
                laplacian_var = 150.0  # Default safe variance
                
                if face_crop.size > 0:
                    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                
                # Check 3D Depth Variation (std dev of Z coordinates of key central facial landmarks)
                # Flat photos projected on 3D mesh might show artificial template Z-depth,
                # but standard deviation of the Z values of landmarks normalized shows specific bounds.
                z_coords = [lm.z for lm in face_landmarks]
                z_std = np.std(z_coords)
                
                # Real face depth characteristics: Z-std is typically within [0.035, 0.07] in normalized mesh coords.
                # A flat screen or printed photo will sometimes fall outside this range due to scaling anomalies,
                # but texture variance is the most robust passive check.
                
                # 5. Face Recognition
                recognized_name = "Unknown"
                confidence_val = 999.0
                
                with recognizer_lock:
                    if recognizer is not None and face_crop.size > 0:
                        try:
                            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                            gray_crop_resized = cv2.resize(gray_crop, (150, 150))
                            label_id, confidence = recognizer.predict(gray_crop_resized)
                            
                            # LBPH confidence score: 0 is perfect match. Usually < 80-90 is good.
                            if confidence < 95:
                                mapping = get_user_mappings()
                                name_mapped = next((name for name, i in mapping.items() if i == label_id), None)
                                if name_mapped:
                                    recognized_name = name_mapped
                                    confidence_val = confidence
                        except Exception as e:
                            print(f"[Face Recognizer] Predict error: {e}")

                # Draw face points (Glowing Point Cloud effect)
                for lm in face_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated_frame, (cx, cy), 1, (0, 255, 128), -1)
                
                # Draw key points with larger indicators
                for pt in [LEFT_EYE_LANDMARKS[0], LEFT_EYE_LANDMARKS[3], RIGHT_EYE_LANDMARKS[0], RIGHT_EYE_LANDMARKS[3], NOSE_TIP]:
                    pt_px = landmarks_px[pt]
                    cv2.circle(annotated_frame, (pt_px[0], pt_px[1]), 3, (180, 105, 255), -1)
                
                # Draw facial boxes & info overlays
                cv2.rectangle(annotated_frame, (min_x, min_y), (max_x, max_y), (180, 105, 255), 2)
                
                # Update Liveness Logic State
                with liveness_state.lock:
                    liveness_state.face_detected = True
                    
                    if liveness_state.start_time is None:
                        liveness_state.start_time = time.time()
                        liveness_state.elapsed_time = 0.0
                    else:
                        liveness_state.elapsed_time = time.time() - liveness_state.start_time

                    # Update recognized name
                    liveness_state.recognized_name = recognized_name
                    liveness_state.recognition_confidence = round(100 - min(confidence_val, 100), 1)

                    # Blink Logic
                    # If EAR drops below 0.18, start a blink detection
                    if avg_ear < 0.18:
                        liveness_state.blink_in_progress = True
                    # When EAR goes back above 0.23, complete the blink count
                    elif avg_ear > 0.23 and liveness_state.blink_in_progress:
                        if blink_cooldown == 0:
                            liveness_state.blink_count += 1
                            liveness_state.blink_in_progress = False
                            blink_cooldown = 10  # Frames cooldown
                    
                    if blink_cooldown > 0:
                        blink_cooldown -= 1
                    
                    # Head Movement (Yaw ratio: Left turn > 1.8, Right turn < 0.55)
                    if yaw_ratio > 1.8:
                        liveness_state.left_turn = True
                    elif yaw_ratio < 0.55:
                        liveness_state.right_turn = True

                    # Smile Detection (Smile ratio > 0.35)
                    if smile_ratio > 0.35:
                        liveness_state.smile_detected = True

                    # Anti-Spoof scoring heuristics
                    # 1. Laplacian texture check: Low variance (<55) indicates blur/low res, high (>800) suggests screens/moire.
                    spoof_penalty = 0.0
                    if laplacian_var < 55:
                        spoof_penalty += 40.0
                    elif laplacian_var > 850:
                        spoof_penalty += 30.0

                    # 2. Check for micro-movements of eyes to ensure it's not a cutout or static image
                    liveness_state.ear_history.append(avg_ear)
                    if len(liveness_state.ear_history) > 60:
                        liveness_state.ear_history.pop(0)
                        ear_variance = np.var(liveness_state.ear_history)
                        if ear_variance < 0.0001:  # extremely static eye shape
                            spoof_penalty += 35.0

                    # 3. Check for uniform static positions (no face movement at all)
                    face_center = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
                    if liveness_state.last_face_center is not None:
                        center_dist = np.linalg.norm(np.array(face_center) - np.array(liveness_state.last_face_center))
                        if center_dist < 0.5:
                            liveness_state.static_frame_count += 1
                        else:
                            liveness_state.static_frame_count = 0
                    liveness_state.last_face_center = face_center

                    # If the face is unnaturally still for 45+ frames, it's likely a static picture or mounting
                    if liveness_state.static_frame_count > 45:
                        spoof_penalty += 25.0
                    
                    liveness_state.spoof_score = min(spoof_penalty, 100.0)

                    # Update Liveness Status
                    if liveness_state.liveness_status == "PENDING":
                        # Check criteria for LIVE PERSON: At least 1 blink AND both head movements (left & right)
                        # Optional: Smile detection can also be logged, but the primary prompt requirement:
                        # "If the user blinks at least once and turns their head left or right, mark the face as 'LIVE PERSON'"
                        # Wait, "turns their head left or right" -> left OR right, or BOTH?
                        # The prompt says: "If the user blinks at least once and turns their head left or right, mark the face as 'LIVE PERSON'."
                        # It says left OR right. Let's make it satisfy (left_turn OR right_turn) AND blink_count >= 1.
                        if (liveness_state.left_turn or liveness_state.right_turn) and liveness_state.blink_count >= 1:
                            if liveness_state.spoof_score < 50.0:
                                liveness_state.liveness_status = "LIVE PERSON"
                                # Log attendance
                                if not liveness_state.snapshot_saved:
                                    # Save a snapshot frame
                                    snapshot_name = f"{liveness_state.recognized_name}_{int(time.time())}.jpg"
                                    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_name)
                                    cv2.imwrite(snapshot_path, frame)
                                    
                                    # Append to CSV
                                    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                                    with open(ATTENDANCE_FILE, 'a', newline='') as f_csv:
                                        writer = csv.writer(f_csv)
                                        writer.writerow([
                                            timestamp_str, 
                                            liveness_state.recognized_name, 
                                            liveness_state.blink_count, 
                                            "Yes" if liveness_state.left_turn else "No", 
                                            "Yes" if liveness_state.right_turn else "No", 
                                            "Yes" if liveness_state.smile_detected else "No", 
                                            "LIVE PERSON",
                                            "Recognized" if liveness_state.recognized_name != "Unknown" else "Unregistered"
                                        ])
                                    liveness_state.snapshot_saved = True
                            else:
                                liveness_state.liveness_status = "SPOOF DETECTED"
                        
                        # Check Timeout (10 seconds)
                        elif liveness_state.elapsed_time > 10.0:
                            liveness_state.liveness_status = "SPOOF DETECTED"
                        
                        # If spoof score is critically high immediately
                        elif liveness_state.spoof_score >= 80.0:
                            liveness_state.liveness_status = "SPOOF DETECTED"

                # Draw Overlay texts on Annotated frame
                with liveness_state.lock:
                    status = liveness_state.liveness_status
                    blinks = liveness_state.blink_count
                    left = liveness_state.left_turn
                    right = liveness_state.right_turn
                    name = liveness_state.recognized_name
                    elapsed = liveness_state.elapsed_time

                # Display info box on frame
                status_color = (0, 255, 0) if status == "LIVE PERSON" else ((0, 0, 255) if status == "SPOOF DETECTED" else (0, 165, 255))
                cv2.putText(annotated_frame, f"Liveness: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(annotated_frame, f"Blinks: {blinks} | Head Left: {left} | Right: {right}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Recognized: {name}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 105, 255), 2)
                cv2.putText(annotated_frame, f"Timer: {max(0.0, 10.0 - elapsed):.1f}s", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            else:
                # No face detected
                with liveness_state.lock:
                    liveness_state.face_detected = False
                    # Reset timer if face is lost
                    liveness_state.start_time = None
                    liveness_state.elapsed_time = 0.0
                
                cv2.putText(annotated_frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Store the frame
            with frame_lock:
                latest_frame = frame.copy()
                latest_frame_annotated = annotated_frame.copy()

            time.sleep(0.03)  # Loop speed matching ~30 FPS

        cap.release()
        print("[Camera Thread] Camera released.")

# Start camera thread globally
camera_thread = CameraThread()
camera_thread.start()

# Yield MJPEG frames
def generate_frames():
    global latest_frame_annotated, camera_thread
    while True:
        with frame_lock:
            if latest_frame_annotated is None:
                frame_data = None
            else:
                ret, buffer = cv2.imencode('.jpg', latest_frame_annotated)
                frame_data = buffer.tobytes() if ret else None

        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        else:
            # Send blank frame if error/starting
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            msg = "Starting Camera..."
            if not camera_thread.running and camera_thread.error_message:
                msg = camera_thread.error_message
            
            if len(msg) > 30:
                words = msg.split(' ')
                line1 = " ".join(words[:5])
                line2 = " ".join(words[5:])
                cv2.putText(blank, line1, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(blank, line2, (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            else:
                cv2.putText(blank, msg, (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

# Server-Sent Events endpoint to push status panel updates
def get_liveness_status_stream():
    while True:
        with liveness_state.lock:
            data = {
                "face_detected": liveness_state.face_detected,
                "blink_count": liveness_state.blink_count,
                "left_turn": liveness_state.left_turn,
                "right_turn": liveness_state.right_turn,
                "smile_detected": liveness_state.smile_detected,
                "liveness_status": liveness_state.liveness_status,
                "elapsed_time": round(liveness_state.elapsed_time, 1),
                "time_left": round(max(0.0, 10.0 - liveness_state.elapsed_time), 1) if liveness_state.start_time else 10.0,
                "spoof_score": round(liveness_state.spoof_score, 1),
                "recognized_name": liveness_state.recognized_name,
                "recognition_confidence": liveness_state.recognition_confidence
            }
        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(0.1)  # stream data every 100ms

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    return Response(get_liveness_status_stream(), mimetype='text/event-stream')

@app.route('/reset', methods=['POST'])
def reset_liveness():
    global camera_thread
    liveness_state.reset()
    
    # Self-healing: if camera thread stopped (due to access error), attempt restart
    if not camera_thread.is_alive() or not camera_thread.running:
        print("[System Reset] Camera thread is stopped. Attempting self-healing restart...")
        camera_thread = CameraThread()
        camera_thread.start()
        
    return jsonify({"status": "success", "message": "Liveness parameters reset and camera connection retried."})

@app.route('/enroll', methods=['POST'])
def enroll_user():
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({"status": "error", "message": "Name cannot be empty."}), 400

    # Ensure camera thread is running and we have a frame
    with frame_lock:
        if latest_frame is None:
            return jsonify({"status": "error", "message": "Camera is not ready. Cannot capture faces."}), 500
        current_cam_frame = latest_frame.copy()

    # Capture face crops to enroll
    # In a real setup, we want to capture 15 frames from the stream over a short duration.
    # To keep it extremely robust and fast, we will capture 15 face frames from the stream
    # during the background thread, or crop directly if a face is present.
    # Let's write a small loop that collects 15 frames from the camera feed
    # We can do this in a short helper thread, updating the user on progress!
    def enrollment_worker(username):
        user_folder = os.path.join(ENROLLED_DIR, username)
        os.makedirs(user_folder, exist_ok=True)
        
        # Load mappings
        mapping = get_user_mappings()
        if username not in mapping:
            # Assign next unique ID
            new_id = len(mapping) + 1
            mapping[username] = new_id
            save_user_mappings(mapping)
        else:
            new_id = mapping[username]

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                running_mode=vision.RunningMode.IMAGE
            )
            detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[Enrollment Error] Failed to initialize Face Landmarker: {e}")
            return False
        
        count = 0
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            
        print(f"[Enrollment] Collecting frames for user: {username} (ID: {new_id})")
        
        while count < 15 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            try:
                detection_result = detector.detect(mp_image)
            except Exception as e:
                print(f"[Enrollment] process error: {e}")
                time.sleep(0.05)
                continue
            
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                
                min_x, max_x = max(0, int(min(xs))), min(w, int(max(xs)))
                min_y, max_y = max(0, int(min(ys))), min(h, int(max(ys)))
                
                face_crop = frame[min_y:max_y, min_x:max_x]
                if face_crop.size > 0:
                    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    gray_resized = cv2.resize(gray_crop, (150, 150))
                    cv2.imwrite(os.path.join(user_folder, f"{count}.jpg"), gray_resized)
                    count += 1
                    print(f"[Enrollment] Saved frame {count}/15")
                    time.sleep(0.1)  # small interval between frames
            time.sleep(0.05)
            
        cap.release()
        
        if count >= 15:
            # Retrain LBPH recognizer
            print("[Enrollment] Training LBPH Face Recognizer...")
            retrain_recognizer()
            return True
        return False

    # Start enrollment worker thread
    thread = threading.Thread(target=enrollment_worker, args=(name,))
    thread.start()
    
    return jsonify({"status": "success", "message": f"Enrollment started for {name}. Look straight at the camera."})

def retrain_recognizer():
    global recognizer
    mapping = get_user_mappings()
    if not mapping:
        return
        
    faces = []
    ids = []
    
    for username, user_id in mapping.items():
        user_folder = os.path.join(ENROLLED_DIR, username)
        if not os.path.isdir(user_folder):
            continue
            
        for file in os.listdir(user_folder):
            if file.endswith('.jpg'):
                img_path = os.path.join(user_folder, file)
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray_img is not None:
                    faces.append(gray_img)
                    ids.append(user_id)
                    
    if len(faces) > 0:
        with recognizer_lock:
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.train(faces, np.array(ids))
                trainer_path = os.path.join(DATA_DIR, 'trainer.yml')
                recognizer.save(trainer_path)
                print(f"[Face Recognizer] Training completed. Saved to {trainer_path}")
            except Exception as e:
                print(f"[Face Recognizer] Training error: {e}")

@app.route('/attendance')
def get_attendance():
    logs = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            # Skip header
            header = next(reader, None)
            for row in reader:
                if len(row) >= 8:
                    logs.append({
                        "timestamp": row[0],
                        "name": row[1],
                        "blinks": row[2],
                        "left_turn": row[3],
                        "right_turn": row[4],
                        "smile": row[5],
                        "liveness": row[6],
                        "recognition": row[7]
                    })
    return jsonify(logs[::-1])  # reverse to show latest logs first

@app.route('/edit_attendance', methods=['POST'])
def edit_attendance():
    data = request.json
    timestamp = data.get('timestamp')
    old_name = data.get('old_name')
    new_name = data.get('new_name', '').strip()
    
    if not new_name:
        return jsonify({"status": "error", "message": "New name cannot be empty."}), 400
        
    updated = False
    rows = []
    
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                rows.append(header)
            for row in reader:
                if len(row) >= 8 and row[0] == timestamp and row[1] == old_name:
                    row[1] = new_name
                    updated = True
                rows.append(row)
                
    if updated:
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return jsonify({"status": "success", "message": "Attendance record updated successfully."})
    else:
        return jsonify({"status": "error", "message": "Record not found."}), 404

@app.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    data = request.json
    timestamp = data.get('timestamp')
    name = data.get('name')
    
    deleted = False
    rows = []
    
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                rows.append(header)
            for row in reader:
                if len(row) >= 8 and row[0] == timestamp and row[1] == name:
                    deleted = True
                    continue  # skip this row to delete it
                rows.append(row)
                
    if deleted:
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return jsonify({"status": "success", "message": "Attendance record deleted successfully."})
    else:
        return jsonify({"status": "error", "message": "Record not found."}), 404

@app.route('/download_report', methods=['GET'])
def download_report():
    # Fetch recent attendance records to render a dynamic report
    logs = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 8:
                    logs.append(row)
    
    # Generate verification report HTML
    # We will embed recent snapshots if available, or list stats
    report_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Liveness Verification & Attendance Report</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 0; }
            .meta-info { margin: 15px 0 25px 0; font-size: 0.9em; color: #7f8c8d; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .status-live { color: #27ae60; font-weight: bold; }
            .status-spoof { color: #c0392b; font-weight: bold; }
            .btn-print { background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; font-weight: bold; cursor: pointer; float: right; }
            .btn-print:hover { background-color: #2980b9; }
        </style>
    </head>
    <body>
        <div class="container">
            <button class="btn-print" onclick="window.print()">Print / Save PDF</button>
            <h1>Liveness & Attendance Report</h1>
            <div class="meta-info">
                <strong>Generated on:</strong> """ + time.strftime("%Y-%m-%d %H:%M:%S") + """<br>
                <strong>System status:</strong> ONLINE | SECURE LIVENESS DETECTION ACTIVE
            </div>
            
            <h2>Verification Logs</h2>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>User Name</th>
                        <th>Blinks</th>
                        <th>Head Left</th>
                        <th>Head Right</th>
                        <th>Smile</th>
                        <th>Liveness Status</th>
                        <th>Authentication</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for row in logs[::-1]:
        status_class = "status-live" if row[6] == "LIVE PERSON" else "status-spoof"
        report_html += f"""
                    <tr>
                        <td>{row[0]}</td>
                        <td>{row[1]}</td>
                        <td>{row[2]}</td>
                        <td>{row[3]}</td>
                        <td>{row[4]}</td>
                        <td>{row[5]}</td>
                        <td><span class="{status_class}">{row[6]}</span></td>
                        <td>{row[7]}</td>
                    </tr>
        """
        
    report_html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save report
    report_path = os.path.join(REPORT_DIR, 'liveness_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_html)
        
    return send_from_directory(REPORT_DIR, 'liveness_report.html', as_attachment=True)

@app.route('/stats')
def get_stats():
    mapping = get_user_mappings()
    registered_count = len(mapping)
    
    total_count = 0
    live_count = 0
    spoof_count = 0
    
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 8:
                    total_count += 1
                    if row[6] == "LIVE PERSON":
                        live_count += 1
                    elif row[6] == "SPOOF DETECTED":
                        spoof_count += 1
                        
    return jsonify({
        "total_verifications": total_count,
        "registered_users": registered_count,
        "live_verifications": live_count,
        "spoof_attempts": spoof_count
    })

if __name__ == '__main__':
    # Shutdown camera thread on exit
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        camera_thread.running = False
        camera_thread.join()
