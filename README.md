# VSB FaceGuard AI: Secure AI Liveness Detection & Biometrics Dashboard

VSB FaceGuard is a secure, real-time Presentation Attack Detection (PAD) and Face Recognition web application. It integrates active liveness challenges (blinking, head turning) with passive texture-based anti-spoofing algorithms to prevent photo printouts and mobile screen replay attacks. 

Developed with a modern cybersecurity slate dashboard theme, it uses **Python Flask, OpenCV, MediaPipe Tasks, and LBPH Face Recognition**.

---

## Key Features

1. **Active Liveness Verification**:
   - **Eye Blink Detection**: Tracks the Eye Aspect Ratio (EAR) dynamically.
   - **Head Rotation Tracker**: Computes scale-invariant horizontal rotation ratios to register Left and Right turns.
   - **Smile Detection**: Tracks mouth deformation ratios relative to face boundaries.
2. **Passive Anti-Photo Spoofing Guards**:
   - **Texture & Sharpness Check**: Measures Laplacian variance on the cropped face region to block high-frequency screen Moire patterns or low-resolution paper printouts.
   - **Eye Micro-movement Verification**: Analyzes the standard deviation of EAR values over a sliding window. Static photos have zero eye dynamics, triggering spoof warnings.
   - **Stillness Tracker**: Flags presentation attacks if the facial mesh coordinates show unnatural stillness (e.g. static printed masks).
3. **On-the-Fly Profile Enrollment & Recognition**:
   - Enroll new users directly from the UI. The camera collects 15 grayscale frames of the user's face, crops them, and trains an **LBPH (Local Binary Patterns Histograms) Face Recognizer** on the fly.
   - Saves the classifier to `data/trainer.yml` and recognizes registered users in subsequent runs.
4. **Attendance Logging & Dynamic KPI Statistics**:
   - Automatically registers attendance entries in `data/attendance.csv` once liveness is successfully verified.
   - Displays real-time database stats on 4 glowing KPI cards: **Total Verifications**, **Registered Profiles**, **Live Verifications**, and **Spoof Intercepted**.
5. **Upgraded Professional Dashboard UI**:
   - Premium glassmorphic interface styled with slate dark theme colors (`#0F172A`/`#1E293B`).
   - SVG Circular Progress ring showing the **Spoof Threat Index** (updates color: Green 🟢, Amber 🟡, Red 🔴 dynamically).
   - Biometric fingerprint scanner profile card with scanning bar animations.
   - Search/Filter log table permitting instant client-side keyword filtration.
   - Dynamic laser scan overlay and AI Focus corners on the active webcam feed.
   - Self-healing camera thread connection that auto-restarts the camera on Reset click.
6. **Audit Report Export**:
   - Exports printable HTML verification reports containing session logs and snapshots.

---

## Project Structure

```text
face-liveness-detection/
├── app.py                  # Flask server, background camera thread, Tasks API, & CSV logger
├── requirements.txt        # Python library dependencies
├── README.md               # Upgraded project documentation
├── templates/
│   └── index.html          # Professional cybersecurity dashboard HTML5 template
├── static/
│   ├── style.css           # Slate theme styles, animations, circular gauges, & responsive grid
│   └── script.js           # SSE client stream, log search, enrollment, & UI controllers
└── data/                   # Generated on launch (attendance logs, mappings, model task, trainer)
```

---

## Installation & Running

Ensure you are using Python 3.8 to 3.14 on a Windows system with a webcam.

### 1. Install Dependencies
Open your command terminal in the project root directory and run:

```bash
python -m pip install --user -r requirements.txt
```

### 2. Launch the Application
Start the Flask web server:

```bash
python app.py
```

*Note: On your first launch, the application will automatically download the required Google MediaPipe model file (`face_landmarker.task` - 5.6 MB) into your `data/` folder.*

### 3. Open in Browser
Open your browser and navigate to:
- **`http://localhost:5000`** (to run on the same laptop)
- **`http://<your-local-network-ip>:5000`** (to view the live dashboard remotely on your smartphone/tablet connected to the same Wi-Fi)

---

## Core Algorithms Explained

### 1. Eye Aspect Ratio (EAR)
Tracks eye blink compression:
\[EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}\]
If the EAR drops below `0.18` and subsequently recovers above `0.23`, a blink event is registered.

### 2. Head Yaw Ratio (Symmetry Check)
We measure the distance from the nose tip (4) to the left cheek boundary (234) and the right cheek boundary (454).
\[Ratio = \frac{Distance(Nose, Left Cheek)}{Distance(Right Cheek, Nose)}\]
- A ratio **greater than 1.8** signifies a **Left Turn**.
- A ratio **less than 0.55** signifies a **Right Turn**.
- Normal centered values reside between `0.6` and `1.6`.

### 3. Texture-Based Spoof Detection
By computing the Laplacian of the cropped face bounding box, we determine the variance in image sharpness:
\[Variance = \sigma^2( \nabla^2 I_{gray} )\]
- **Extremely Low Variance (< 55)**: Indicates out-of-focus prints, printed texture paper, or blur.
- **Extremely High Variance (> 850)**: Indicates halftone printing dots or computer screen grid Moire patterns.

---

## Usage Guide for Demonstrations

1. **Verify Liveness**: Look into the camera. Perform a blink, then rotate your head left or right. Verify that the checklist boxes turn into green checks and the banner transitions to a pulsing green **LIVE PERSON VERIFIED** badge.
2. **Enroll a User**: Type a name (e.g. `Gokul B`) and click **Enroll**. Keep your head steady as the scanner collects 15 frames. Once trained, the profile card will read `Identified Profile: Gokul B`.
3. **Verify Spoof Warning**: Remain completely static for 10 seconds or turn away from the camera. The system will trigger `SPOOF DETECTED` (flashing red state).
4. **Logs Filtering**: Type any character in the table search bar; the logs list will filter instantly.
5. **Edit / Delete entries**: Click the pencil icon next to a log to rename the entry, or the trash icon to permanently remove it.
