import ctypes
import os
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_kr_font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)


def put_text_kr(frame, text, pos, color_bgr):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    b, g, r = color_bgr
    draw.text(pos, text, font=_kr_font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def ctrl_space_pressed():
    ctrl = ctypes.windll.user32.GetAsyncKeyState(0x11)
    space = ctypes.windll.user32.GetAsyncKeyState(0x20)
    return ctrl and space


def count_fingers(lm):
    count = 0 

    margin = 0.03

    for tip in (8, 12, 16, 20):
        if lm[tip].y < lm[tip - 2].y - margin:
            count += 1

    thumb_dir = lm[2].x - lm[17].x
    if (lm[4].x - lm[3].x) * thumb_dir > 0:
        count += 1

    return count


def read_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def build_face_db(root="faces_db"):
    db = {}
    if not os.path.isdir(root):
        return db
    for name in os.listdir(root):
        person_dir = os.path.join(root, name)
        if not os.path.isdir(person_dir):
            continue
        feats = []
        for fn in os.listdir(person_dir):
            img = read_unicode(os.path.join(person_dir, fn))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            yunet.setInputSize((iw, ih))
            _, fs = yunet.detect(img)
            if fs is None:
                continue
            f = max(fs, key=lambda r: r[2] * r[3])
            aligned = recognizer.alignCrop(img, f)
            feats.append(recognizer.feature(aligned))
        if feats:
            db[name] = feats
            print("enrolled " + name + ": " + str(len(feats)) + " faces")
    return db


def identify(frame, face_row):
    aligned = recognizer.alignCrop(frame, face_row)
    feat = recognizer.feature(aligned)
    best_name, best_score = "Unknown", 0.0
    for name, feats in face_db.items():
        for f in feats:
            s = recognizer.match(feat, f, cv2.FaceRecognizerSF_FR_COSINE)
            if s > best_score:
                best_score, best_name = s, name
    label = best_name if best_score > COSINE_THRESHOLD else "Unknown"
    return label, best_score


yunet = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    score_threshold=0.6,
)

recognizer = cv2.FaceRecognizerSF.create(
    "face_recognition_sface_2021dec.onnx",
    "",
)
COSINE_THRESHOLD = 0.45

face_db = build_face_db()
print("DB people: " + str(list(face_db.keys())))

with open("hand_landmarker.task", "rb") as _f:
    _hand_model = _f.read()

hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=_hand_model),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
)

camera = cv2.VideoCapture(0)

alt_tab_done = False
reset_key_pressed = False

hand_return_done = False
open_hand_frames = 0
OPEN_HAND_THRESHOLD = 5

while True:
    success, frame = camera.read()

    if not success:
        break

    h, w = frame.shape[:2]
    recognized = []

    yunet.setInputSize((w, h))
    _, yn_faces = yunet.detect(frame)
    if yn_faces is not None:
        for f in yn_faces:
            bx, by, bw, bh = f[:4].astype(int)
            name, score = identify(frame, f)
            recognized.append(((bx, by, bw, bh), name, score))

    face_count = len(recognized)

    for (x, y, fw, fh), name, score in recognized:
        known = name != "Unknown"
        color = (0, 255, 0) if known else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        frame = put_text_kr(frame, name + " " + str(round(score, 2)), (x, y - 30), color)

    cv2.putText(
        frame,
        "Faces: " + str(face_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)
    fingers = 0
    if result.hand_landmarks:
        fingers = count_fingers(result.hand_landmarks[0])

    cv2.putText(
        frame,
        "Fingers: " + str(fingers),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    if fingers == 5:
        open_hand_frames += 1
    else:
        open_hand_frames = 0
        hand_return_done = False

    if open_hand_frames >= OPEN_HAND_THRESHOLD and not hand_return_done and alt_tab_done:
        pyautogui.hotkey("alt", "tab")
        hand_return_done = True

    if ctrl_space_pressed() and reset_key_pressed == False:
        alt_tab_done = False
        reset_key_pressed = True

    if not ctrl_space_pressed():
        reset_key_pressed = False

    unknown_present = any(name == "Unknown" for _, name, _ in recognized)
    if unknown_present and alt_tab_done == False:
        pyautogui.hotkey("alt", "tab")
        alt_tab_done = True

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()