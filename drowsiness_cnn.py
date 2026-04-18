from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import pygame
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import mediapipe as mp
import math

# ==================== SETTINGS ====================
IMG_SIZE = 32          # must match both CNN trainings

# ---- EYE / DROWSINESS TUNING ----
EYE_ALPHA = 0.20              # smoother for eye CNN score
EYE_DROWSY_HIGH = 0.70        # above this → likely drowsy

#  RELAXED recovery: easier to go back to Non Drowsy
EYE_AWAKE_LOW = 0.52          # below this → likely awake (was 0.45)
EYE_DROWSY_FRAMES = 10        # frames to confirm drowsy
EYE_AWAKE_FRAMES = 5          # frames to confirm awake (was 8)

# ---- EYE VISIBILITY (HAAR) ----
NO_EYE_FRAMES_THRESH = 12     # many frames without eyes -> drowsy

# ---- MOUTH / YAWN TUNING ----
MOUTH_ALPHA = 0.25            # smoothing for yawn score
YAWN_THRESH = 0.65            # treat >0.65 as yawn (on smoothed combined score)
YAWN_MIN_FRAMES = 5           # continuous frames
YAWN_COOLDOWN_FRAMES = 35     # frames to wait before next yawn alert

# geometric MAR → 0..1 normalisation (rough)
MAR_MIN = 0.20   # closed mouth approx
MAR_MAX = 0.70   # wide open mouth approx

# ---------- EMAIL ALERT SETUP ----------
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "receiver_email@gmail.com"
#[email alert stopped , i will resume it later]
screenshot_path = "alert.jpg"
email_sent = False  # to avoid spamming emails


def send_email_alert(image_path, timestamp, reason="Drowsiness / Yawning"):
    if not os.path.exists(image_path):
        print("❌ Email: screenshot file not found:", image_path)
        return False

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = f"🚨 Driver Alert: {reason}"

    body = f"""
    Warning: The driver showed signs of {reason} at {timestamp}.
    The attached image shows the condition at that moment.
    """

    msg.attach(MIMEText(body, "plain"))

    try:
        with open(image_path, "rb") as attachment:
            img = MIMEBase('application', 'octet-stream')
            img.set_payload(attachment.read())
            encoders.encode_base64(img)
            img.add_header(
                'Content-Disposition',
                'attachment; filename="alert.jpg"'
            )
            msg.attach(img)
    except Exception as e:
        print("❌ Error reading screenshot for email:", e)
        return False

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("📧 Email alert sent successfully.")
        return True
    except Exception as e:
        print("❌ Email sending failed:", e)
        return False


# ---------- BEEP SOUND SETUP ----------
SOUND_ENABLED = False
sound = None
sound_flag = False  

try:
    pygame.mixer.init()
    if os.path.exists("beep.mp3"):
        sound = pygame.mixer.Sound("beep.mp3")
        SOUND_ENABLED = True
        print("🔊 Beep sound loaded.")
    else:
        print("⚠ beep.mp3 not found. Sound will be disabled.")
except Exception as e:
    print("⚠ Sound init error, sound disabled:", e)
    SOUND_ENABLED = False


# ---------- MODELS ----------
# YOLO model
if not os.path.exists("yolov8n.pt"):
    print("❌ yolov8n.pt not found in current directory.")
    raise SystemExit(1)

try:
    yolo_model = YOLO("yolov8n.pt")
    print("✅ YOLO model loaded.")
except Exception as e:
    print("❌ Error loading YOLO model:", e)
    raise SystemExit(1)

# Eye CNN model (32x32 gray)
if not os.path.exists("eye_cnn.h5"):
    print("❌ eye_cnn.h5 not found in current directory.")
    raise SystemExit(1)

try:
    eye_model = tf.keras.models.load_model("eye_cnn.h5")
    print("✅ Eye CNN model loaded.")
except Exception as e:
    print("❌ Error loading eye CNN model:", e)
    raise SystemExit(1)

# Mouth / Yawn CNN model (32x32 gray)
if not os.path.exists("mouth_cnn.h5"):
    print("❌ mouth_cnn.h5 not found in current directory.")
    raise SystemExit(1)

try:
    mouth_model = tf.keras.models.load_model("mouth_cnn.h5")
    print("✅ Mouth CNN model loaded.")
except Exception as e:
    print("❌ Error loading mouth CNN model:", e)
    raise SystemExit(1)

# Haar cascade for eyes
eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
if not os.path.exists(eye_cascade_path):
    print("❌ Haar cascade file not found:", eye_cascade_path)
    raise SystemExit(1)

eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
if eye_cascade.empty():
    print("❌ Failed to load Haar cascade for eyes.")
    raise SystemExit(1)
else:
    print("✅ Eye Haar cascade loaded.")

# ---------- MEDIAPIPE FACE MESH FOR MOUTH ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# standard lip landmarks for MAR computation
LM_LEFT_CORNER = 61
LM_RIGHT_CORNER = 291
LM_UPPER_MID = 13
LM_LOWER_MID = 14
LM_UPPER_LEFT = 82
LM_LOWER_LEFT = 87
LM_UPPER_RIGHT = 312
LM_LOWER_RIGHT = 317

# for bounding box
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]


# ---------- CNN HELPERS ----------
def get_eye_drowsy_prob(face_region):
    """Return Drowsy probability from eye CNN given full face crop."""
    if face_region is None or face_region.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    preds = eye_model.predict(gray, verbose=0)[0]
    non_d, drowsy = preds
    print(f"[EYE]  NonD={non_d:.3f}, Drowsy={drowsy:.3f}")
    return float(drowsy)


def get_yawn_prob(mouth_region):
    """Return Yawn probability from mouth CNN given mouth crop."""
    if mouth_region is None or mouth_region.size == 0:
        return 0.5  # neutral if nothing
    gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    preds = mouth_model.predict(gray, verbose=0)[0]
    no_yawn, yawn = preds
    return float(yawn)


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def extract_mouth_region_with_facemesh(face_region):
    """
    Use MediaPipe FaceMesh on the face_region to get a tight mouth box
    and compute Mouth Aspect Ratio (MAR).
    Returns (mouth_region, mar) or (None, 0.0) if not found.
    """
    if face_region is None or face_region.size == 0:
        return None, 0.0

    h, w, _ = face_region.shape
    rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_face)

    if not result.multi_face_landmarks:
        return None, 0.0

    landmarks = result.multi_face_landmarks[0]

    # ----- MAR computation points -----
    def lp(idx):
        lm = landmarks.landmark[idx]
        return (lm.x * w, lm.y * h)

    p_left = lp(LM_LEFT_CORNER)
    p_right = lp(LM_RIGHT_CORNER)

    p_up_mid = lp(LM_UPPER_MID)
    p_lo_mid = lp(LM_LOWER_MID)
    p_up_left = lp(LM_UPPER_LEFT)
    p_lo_left = lp(LM_LOWER_LEFT)
    p_up_right = lp(LM_UPPER_RIGHT)
    p_lo_right = lp(LM_LOWER_RIGHT)

    horiz = dist(p_left, p_right)
    vert1 = dist(p_up_mid, p_lo_mid)
    vert2 = dist(p_up_left, p_lo_left)
    vert3 = dist(p_up_right, p_lo_right)
    mar = 0.0
    if horiz > 0:
        mar = (vert1 + vert2 + vert3) / (3.0 * horiz)

    # ----- bounding box for mouth crop -----
    xs = []
    ys = []
    for idx in MOUTH_LANDMARKS:
        lm = landmarks.landmark[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x_min = max(min(xs), 0)
    x_max = min(max(xs), w - 1)
    y_min = max(min(ys), 0)
    y_max = min(max(ys), h - 1)

    # add small margin
    dx = int(0.20 * (x_max - x_min + 1))
    dy = int(0.20 * (y_max - y_min + 1))

    x1 = max(0, x_min - dx)
    y1 = max(0, y_min - dy)
    x2 = min(w - 1, x_max + dx)
    y2 = min(h - 1, y_max + dy)

    if x2 <= x1 or y2 <= y1:
        return None, mar

    mouth_region = face_region[y1:y2, x1:x2]

    # draw box on face for visualization
    cv2.rectangle(face_region, (x1, y1), (x2, y2), (0, 255, 255), 1)

    return mouth_region, mar


# ==================== MAIN LOOP ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam (index 0).")
    raise SystemExit(1)

eye_state = "Non Drowsy"
eye_score = 0.0
eye_drowsy_frames = 0
eye_awake_frames = 0
no_eye_frames = 0

yawn_state = "Normal"
yawn_score = 0.0     # smoothed combined score (MAR + CNN)
yawn_frames = 0
yawn_cooldown = 0

print("🚗 Driver Drowsiness + Yawn Detection Active — Press 'Q' to exit\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Failed to read frame from camera.")
            break

        saved_frame = frame.copy()  # for email screenshot

        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes

        face_region = None
        eyes_detected = False

        # -------- YOLO: get face crop ----------
        for box in boxes:
            cls = int(box.cls[0])
            if cls != 0:  # we only care about 'person'
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            # tighten box slightly to upper body/face
            x1 += w // 8
            x2 -= w // 8
            y1 += h // 8
            y2 -= h // 8

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_region = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break  # only first person

        if face_region is not None and face_region.size != 0:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            fh, fw = face_gray.shape

            # ------ Eye region (top ~55% of face) ------
            eye_h_end = int(fh * 0.55)
            roi_gray = face_gray[0:eye_h_end, :]
            roi_color = face_region[0:eye_h_end, :]
            roi_small = cv2.resize(roi_gray, None, fx=0.5, fy=0.5)

            eyes = eye_cascade.detectMultiScale(
                roi_small,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(18, 18)
            )

            if len(eyes) > 0:
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    ex *= 2
                    ey *= 2
                    ew *= 2
                    eh *= 2
                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        (255, 0, 0),
                        2
                    )

            if eyes_detected:
                no_eye_frames = 0
            else:
                no_eye_frames += 1

            # Eye CNN
            raw_eye_prob = get_eye_drowsy_prob(face_region)
            eye_score = (1 - EYE_ALPHA) * eye_score + EYE_ALPHA * raw_eye_prob

            # ------ Mouth region via FaceMesh (MAR + CNN) ------
            mouth_region, mar = extract_mouth_region_with_facemesh(face_region)

            if mouth_region is not None:
                raw_yawn_prob = get_yawn_prob(mouth_region)

                # Normalize MAR
                mar_norm = (mar - MAR_MIN) / (MAR_MAX - MAR_MIN)
                mar_norm = max(0.0, min(1.0, mar_norm))

                # Combine CNN + MAR scores
                combined_raw = 0.7 * mar_norm + 0.3 * raw_yawn_prob

                # Smooth the output
                yawn_score = (1 - MOUTH_ALPHA) * yawn_score + MOUTH_ALPHA * combined_raw

                # ---- NEW LOGIC ----
                if yawn_score >= YAWN_THRESH and yawn_cooldown == 0:
                    yawn_frames += 1
                else:
                    yawn_frames = 0

                # Confirm yawning
                if yawn_frames >= YAWN_MIN_FRAMES:
                    yawn_state = "Yawning"
                    yawn_cooldown = YAWN_COOLDOWN_FRAMES

                # EARLY RECOVERY CHECK — RESET STATE IF USER CLOSED MOUTH
                if yawn_score < (YAWN_THRESH * 0.55):  # adjustable sensitivity
                    yawn_state = "Normal"
                    yawn_cooldown = 0

                # Reduce cooldown if active
                if yawn_cooldown > 0:
                    yawn_cooldown -= 1

        else:
            # no face
            no_eye_frames += 1
            eye_score = (1 - EYE_ALPHA) * eye_score
            yawn_score = (1 - MOUTH_ALPHA) * yawn_score

        # -------- Eye-based state decision (FIX 1 APPLIED) ----------
        if no_eye_frames >= NO_EYE_FRAMES_THRESH:
            # eyes not visible for many frames → treat as drowsy
            eye_state = "Drowsy"
        else:
            if eye_score >= EYE_DROWSY_HIGH:
                # strong evidence of drowsy
                eye_drowsy_frames += 1
                eye_awake_frames = 0

            elif eye_score <= EYE_AWAKE_LOW:
                # strong evidence of awake
                eye_awake_frames += 1
                eye_drowsy_frames = 0

            else:
                # MID-RANGE: use it as RECOVERY if we are already drowsy
                if eye_state == "Drowsy":
                    eye_awake_frames += 1      # count recovery frames
                    eye_drowsy_frames = 0
                else:
                    # uncertain but previously Non Drowsy → don't accumulate anything
                    eye_drowsy_frames = 0
                    eye_awake_frames = 0

            # final decision
            if eye_drowsy_frames >= EYE_DROWSY_FRAMES:
                eye_state = "Drowsy"
            elif eye_awake_frames >= EYE_AWAKE_FRAMES:
                eye_state = "Non Drowsy"
            # else: keep previous state

        # -------- ALERTS (Drowsy or Yawning) ----------
        dangerous = (eye_state == "Drowsy") or (yawn_state == "Yawning")

        if dangerous:
            # Start alarm if danger detected and not already playing
            if SOUND_ENABLED and not sound_flag:
                try:
                    sound.play(-1)   # loop alarm
                    sound_flag = True
                    print("🔔 Continuous alarm started.")
                except Exception as e:
                    print("⚠ Error playing sound:", e)
                    SOUND_ENABLED = False

            # Send email only once per danger phase
            if not email_sent:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.imwrite(screenshot_path, saved_frame)
                reason = f"{eye_state}"
                if yawn_state == "Yawning":
                    reason += " + Yawning"
                if send_email_alert(screenshot_path, timestamp, reason):
                    email_sent = True

        else:
            # SAFETY RECOVERY MODE — STOP SOUND IMMEDIATELY
            if sound_flag:
                try:
                    sound.fadeout(600)   # smoother stop
                except:
                    sound.stop()
                sound_flag = False

            # Reset email trigger for future dangerous episodes
            email_sent = False

            # Reset yawn cooldown immediately if no longer yawning
            if yawn_state != "Yawning":
                yawn_cooldown = 0

        # -------- DISPLAY ----------
        eye_color = (0, 255, 0) if eye_state == "Non Drowsy" else (0, 255, 255)

        cv2.putText(frame, f"EyeState: {eye_state}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, eye_color, 2)

        cv2.putText(frame, f"EyeScore: {eye_score:.3f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"NoEyeFrames: {no_eye_frames}", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(frame, f"YawnState: {yawn_state}", (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255) if yawn_state == "Yawning" else (255, 255, 255), 2)

        cv2.putText(frame, f"YawnScore: {yawn_score:.3f}", (30, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if dangerous:
            cv2.putText(frame, "ALERT !!!", (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        cv2.imshow("Driver Drowsiness + Yawn Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        if sound_flag:
            sound.stop()
        pygame.mixer.quit()
    except:
        pass
    print("👋 Exited cleanly.")
