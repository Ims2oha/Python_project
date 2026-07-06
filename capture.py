import os
import cv2
import numpy as np

# 등록할 사람 이름 (실행 시 물어봄)
person = input("등록할 사람 이름: ").strip()
save_dir = os.path.join("faces_db", person)
os.makedirs(save_dir, exist_ok=True)

# 기존 파일 개수부터 이어서 번호 매김
start = len([f for f in os.listdir(save_dir) if f.lower().endswith(".jpg")])

yunet = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    score_threshold=0.6,
)

camera = cv2.VideoCapture(0)
count = 0

print("SPACE=현재 얼굴 저장, Q=종료. 정면/좌우/위/아래 각도로 여러 장 찍어.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    yunet.setInputSize((w, h))
    _, faces = yunet.detect(frame)

    face_ok = faces is not None and len(faces) > 0
    if face_ok:
        for f in faces:
            x, y, fw, fh = f[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    cv2.putText(
        frame,
        "saved: " + str(count) + "  SPACE=save Q=quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if face_ok else (0, 0, 255),
        2,
    )

    cv2.imshow("capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" ") and face_ok:
        # 한글 경로 저장은 imencode + tofile
        path = os.path.join(save_dir, "cap_" + str(start + count) + ".jpg")
        cv2.imencode(".jpg", frame)[1].tofile(path)
        count += 1
        print("saved", path)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("총 " + str(count) + "장 추가됨 →", save_dir)
