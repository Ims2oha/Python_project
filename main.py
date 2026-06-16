import ctypes
import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def ctrl_space_pressed():
    ctrl = ctypes.windll.user32.GetAsyncKeyState(0x11)
    space = ctypes.windll.user32.GetAsyncKeyState(0x20)
    return ctrl and space


def merge_boxes(boxes, iou_threshold=0.3):
    # 겹치는 박스를 하나로 합쳐서 중복 카운트를 막는다
    merged = []
    for box in boxes:
        x, y, w, h = box
        duplicate = False
        for mx, my, mw, mh in merged:
            ix1 = max(x, mx)
            iy1 = max(y, my)
            ix2 = min(x + w, mx + mw)
            iy2 = min(y + h, my + mh)   
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = w * h + mw * mh - inter
            if union > 0 and inter / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            merged.append((x, y, w, h))
    return merged


def count_fingers(lm):
    # 펴진 손가락 개수를 센다 (mediapipe 21개 landmark 기준)
    count = 0

    # margin: 경계에서 깜빡임/오인식 방지 (이만큼 확실히 위에 있어야 펴짐)
    margin = 0.03

    # 검지~새끼: tip(8,12,16,20)이 pip(6,10,14,18)보다 margin 이상 위(y 작음)면 펴짐
    for tip in (8, 12, 16, 20):
        if lm[tip].y < lm[tip - 2].y - margin:
            count += 1

    # 엄지: 방향 기반 (좌우손/미러 무관, 임계값 튜닝 불필요)
    # 엄지 바깥 방향 = 새끼뿌리(17) -> 엄지뿌리(2) 쪽.
    # tip(4)이 ip(3)보다 그 바깥으로 더 나가면 펴짐, 안쪽(손바닥)으로 들어가면 접힘.
    thumb_dir = lm[2].x - lm[17].x
    if (lm[4].x - lm[3].x) * thumb_dir > 0:
        count += 1

    return count


# Haar Cascade (기존 방식)
face_cascade = cv2.CascadeClassifier("face.xml")

# YuNet (딥러닝 방식, 추가)
yunet = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    score_threshold=0.6,
)

# MediaPipe Hands (손 관절 검출, Tasks API)
# 모델을 bytes(buffer)로 읽어 넘긴다 → 한글 경로(얼굴인식)를 C++ 로더에 안 넘겨 오류 회피
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

faces = []
alt_tab_done = False
reset_key_pressed = False

hand_return_done = False   # 중복 실행 방지
open_hand_frames = 0       # 연속 5개 카운터
OPEN_HAND_THRESHOLD = 5    # N프레임 연속이어야 동작

while True:
    success, frame = camera.read()

    if not success:
        break

    h, w = frame.shape[:2]
    boxes = []

    # 1) YuNet 검출
    yunet.setInputSize((w, h))
    _, yn_faces = yunet.detect(frame)
    if yn_faces is not None:
        for f in yn_faces:
            bx, by, bw, bh = f[:4].astype(int)
            boxes.append((bx, by, bw, bh))

    # 2) Haar 검출 (보조)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    haar_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60),
    )
    for x, y, fw, fh in haar_faces:
        boxes.append((int(x), int(y), int(fw), int(fh)))

    # 3) 두 결과 합치기 (겹치는 박스 제거)
    faces = merge_boxes(boxes)

    face_count = len(faces)

    for x, y, fw, fh in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    cv2.putText(
        frame,
        "Faces: " + str(face_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # 손 검출 + 손가락 개수
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

    # 손가락 5개 N프레임 연속 → 직전 창으로 복귀 (alt+tab 토글)
    # 단, 얼굴 2개로 화면이 한번 바뀐 뒤(alt_tab_done=True)에만 동작
    if fingers == 5:
        open_hand_frames += 1
    else:
        open_hand_frames = 0
        hand_return_done = False   # 손 내리면 리셋 → 다시 펴면 또 동작

    if open_hand_frames >= OPEN_HAND_THRESHOLD and not hand_return_done and alt_tab_done:
        pyautogui.hotkey("alt", "tab")
        hand_return_done = True

    if ctrl_space_pressed() and reset_key_pressed == False:
        alt_tab_done = False
        reset_key_pressed = True

    if not ctrl_space_pressed():
        reset_key_pressed = False

    if face_count >= 2 and alt_tab_done == False:
        pyautogui.hotkey("alt", "tab")
        alt_tab_done = True

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
