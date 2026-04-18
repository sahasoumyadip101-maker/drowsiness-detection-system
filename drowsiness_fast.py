import cv2
import numpy as np

"""
FAST & STABLE DROWSINESS DETECTION (Python 3.11)

- Uses ONLY OpenCV Haar cascades (face + eyes)
- No YOLO, no CNN => much smoother and less laggy
- Logic:
    - If eyes are missing for N consecutive frames => Drowsy
    - Else => Non Drowsy
"""

# Haar cascade paths (bundled with OpenCV)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"

# TUNABLE PARAMETERS
NO_EYE_FRAMES_THRESH = 12   # frames without eyes -> Drowsy (increase = slower, decrease = faster)
FRAME_DOWNSCALE = 0.7       # resize factor to speed up processing (0.5–0.8 is good)

def main():
    # Load cascades
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    if face_cascade.empty() or eye_cascade.empty():
        print("Error: Could not load Haar cascades.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    no_eye_frames = 0
    stable_state = "Non Drowsy"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Downscale frame for speed
        h, w = frame.shape[:2]
        new_w = int(w * FRAME_DOWNSCALE)
        new_h = int(h * FRAME_DOWNSCALE)
        small_frame = cv2.resize(frame, (new_w, new_h))

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces on smaller frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        eyes_detected = False

        # Take first face only
        for (x, y, fw, fh) in faces[:1]:
            # Draw face rect on original frame (scale coords back)
            x_big = int(x / FRAME_DOWNSCALE)
            y_big = int(y / FRAME_DOWNSCALE)
            fw_big = int(fw / FRAME_DOWNSCALE)
            fh_big = int(fh / FRAME_DOWNSCALE)

            cv2.rectangle(frame, (x_big, y_big), (x_big + fw_big, y_big + fh_big), (0, 255, 0), 2)

            # Use upper half of face for eyes (in SMALL frame)
            roi_gray = gray[y : y + fh // 2, x : x + fw]
            roi_color_small = small_frame[y : y + fh // 2, x : x + fw]

            # Detect eyes in ROI
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            if len(eyes) > 0:
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    # Scale eye box back to original full frame
                    ex_big = int((x + ex) / FRAME_DOWNSCALE)
                    ey_big = int((y + ey) / FRAME_DOWNSCALE)
                    ew_big = int(ew / FRAME_DOWNSCALE)
                    eh_big = int(eh / FRAME_DOWNSCALE)

                    cv2.rectangle(
                        frame,
                        (ex_big, ey_big),
                        (ex_big + ew_big, ey_big + eh_big),
                        (255, 0, 0),
                        2
                    )

            break  # only first face

        # Update no-eye frames counter
        if eyes_detected:
            no_eye_frames = 0
        else:
            no_eye_frames += 1

        # Decide state
        if no_eye_frames >= NO_EYE_FRAMES_THRESH:
            stable_state = "Drowsy"
        else:
            stable_state = "Non Drowsy"

        # Draw info
        color = (0, 255, 0) if stable_state == "Non Drowsy" else (0, 255, 255)
        cv2.putText(frame, f"State: {stable_state}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"NoEyeFrames: {no_eye_frames}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if stable_state == "Drowsy":
            cv2.putText(frame, "DROWSY ALERT !!!", (30, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("FAST Driver Drowsiness Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
