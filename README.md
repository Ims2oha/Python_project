# 얼굴 인식(누구인지) + 손동작 창 전환

- **프로젝트명**: 유튜브 보고싶다
- **프로젝트 주제**: 야자 때 딴짓하다가 선생님에게 들키는 걸 막기
- **선정 배경**: 4월 쯤 몰래 딴 짓을 하다가 서은주 선생님께 들켜서 좀 혼났는 데 다시는 들키지 않는다는 다짐으로 만들 게 되었습니다
- **프로젝트 목표**: 선생님이 야자감독으로 노트북을 감지하러 복도로 오시면 그걸 감지해서 바로 화면이 바뀌게 하여 딴짓을 해도 걸리지 않게 하기

---

웹캠으로 **얼굴이 누구인지**와 **손가락 수**를 실시간 인식해서 Windows 창을 자동 전환하는 프로그램.

- **등록 안 된 얼굴(모르는 사람) 감지** → 다른 창으로 전환 (`Alt+Tab`)
- **손가락 5개(하이파이브)** → 직전 창으로 복귀 (`Alt+Tab`) — *단, 얼굴로 한번 전환된 뒤에만 동작*

> 이전 버전은 "얼굴이 **2개 이상**이면 전환"이었으나, 이제는 **등록된 본인만 있으면 전환 안 하고, 모르는 사람(선생님)이 잡혀야 전환**한다. 훨씬 정확.

---

## 동작 흐름

```
카메라 프레임
   │
   ├─ 얼굴 검출 (YuNet) ── 얼굴마다 SFace로 신원 인식 ── 이름 or Unknown
   │        └─ Unknown(등록 안 됨) 1명이라도 있으면 → Alt+Tab (다른 창으로)  [alt_tab_done = True]
   │
   └─ 손 검출 (MediaPipe) ── 손가락 수 카운트
            └─ 5개를 N프레임 연속 + alt_tab_done → Alt+Tab (직전 창 복귀)
```

핵심 상태값:

| 변수 | 역할 |
|------|------|
| `face_db` | `{이름: [얼굴 embedding들]}`. 시작 시 `faces_db/` 폴더로 만듦 |
| `alt_tab_done` | 얼굴로 화면 전환이 일어났는지. 손동작 복귀의 **전제 조건** |
| `open_hand_frames` | 손가락 5개가 연속으로 잡힌 프레임 수 (오작동 방지) |
| `hand_return_done` | 손동작 복귀가 이미 실행됐는지 (한 번만 동작) |

---

## 얼굴 인식 방식 — OpenCV SFace

- **YuNet** (`FaceDetectorYN`): 얼굴 **위치 + landmark 5점**(눈/코/입) 검출.
- **SFace** (`FaceRecognizerSF`): landmark로 얼굴을 정렬(`alignCrop`)한 뒤 **128차원 embedding** 추출.
- 두 embedding의 **코사인 유사도**가 `COSINE_THRESHOLD`를 넘으면 같은 사람.

```
YuNet detect → 얼굴 box + landmark
   └─ recognizer.alignCrop(frame, face) → 정렬된 얼굴
       └─ recognizer.feature() → 128-d embedding
           └─ 등록된 사람들 embedding과 코사인 비교
               └─ max 유사도 > COSINE_THRESHOLD → 그 사람 이름
                  else → "Unknown"
```

> **Haar Cascade는 제거됨.** SFace는 landmark가 있어야 하는데 Haar는 landmark를 안 준다. 또 YuNet+Haar 병합이 같은 얼굴을 2개로 세는 문제도 있어서 **YuNet 단독**으로 바꿨다. (`face.xml`, `merge_boxes()` 더 이상 안 씀)

---

## 등록 (누구인지 학습)

사람별 폴더에 얼굴 사진을 넣어두면 **시작할 때 자동 학습**된다.

```
faces_db/
  이겸이/
    xxx.jpg
    yyy.jpg ...
  임서하/
    ...
```

- 사진은 **정면 + 좌우 살짝 + 위/아래로 볼 때**까지 여러 장(5~12장) 권장.
- **실제 야자 조명**에서 찍은 것 포함해야 밤에 본인을 놓치지 않음.
- **완전 옆모습(90도)·고개 푹 숙임·흐린 사진은 빼기** — 눈 하나가 안 보이면 정렬이 깨져 오히려 방해된다.

### 캡처 도구 — `capture.py`

런타임과 **같은 웹캠**으로 각도별 사진을 바로 찍어 폴더에 넣는 보조 스크립트.

```bash
python capture.py
```

1. 콘솔에 이름 입력 (예: `이겸이`)
2. 창에서 얼굴 각도를 바꿔가며 **SPACE** = 저장 (초록 박스일 때만 저장됨)
3. **Q** = 종료 → `faces_db/이름/`에 `cap_N.jpg`로 이어붙음
4. 사람마다 반복 후 `python main.py` 재실행

---

## 필요 파일

| 파일 | 설명 |
|------|------|
| `main.py` | 본체 (검출 + 인식 + 트리거) |
| `capture.py` | 등록용 얼굴 캡처 도구 |
| `face_detection_yunet_2023mar.onnx` | YuNet 얼굴 검출 모델 |
| `face_recognition_sface_2021dec.onnx` | SFace 얼굴 **인식** 모델 (~37MB) |
| `hand_landmarker.task` | MediaPipe 손 관절 검출 모델 (약 7.8MB) |
| `faces_db/<이름>/*.jpg` | 등록할 사람별 얼굴 사진 |

> - `hand_landmarker.task`: [MediaPipe 모델 페이지](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
> - `face_recognition_sface_2021dec.onnx`: [OpenCV Zoo](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)

## 설치 / 실행

```bash
pip install opencv-contrib-python mediapipe==0.10.14 pyautogui pillow numpy
python main.py
```

- 종료: 영상 창에서 **`q`** 키
- 리셋: **`Ctrl + Space`** → `alt_tab_done`을 풀어 얼굴 전환을 다시 가능하게 함

> **주의 — MediaPipe 버전**: `mediapipe==0.10.14`로 고정. 최신(0.10.35)은 손가락 인식에 쓰는 `solutions` API가 빠져 있고, 또 모델을 **bytes로 읽어 넘긴다**(한글 폴더 경로 `얼굴인식`을 C++ 로더가 못 읽는 문제 회피).

> **주의 — 한글 경로**: 프로젝트 폴더명이 한글(`얼굴인식`)이라 OpenCV C++ 로더가 경로를 못 읽는다. 그래서:
> - onnx 모델은 **파일명만**(상대경로) 넘긴다 — 폴더 안에서 실행하므로 경로에 한글이 안 들어감.
> - 사진은 `cv2.imread` 대신 `np.fromfile` + `cv2.imdecode`로 읽고, 저장은 `cv2.imencode` + `tofile`로 한다.

---

## 코드 해석

### 1. 키 입력 감지 — `ctrl_space_pressed()`

```python
def ctrl_space_pressed():
    ctrl = ctypes.windll.user32.GetAsyncKeyState(0x11)   # 0x11 = Ctrl
    space = ctypes.windll.user32.GetAsyncKeyState(0x20)  # 0x20 = Space
    return ctrl and space
```

Windows API(`GetAsyncKeyState`)로 Ctrl과 Space가 동시에 눌렸는지 본다. 리셋 트리거로 사용.

### 2. 한글 경로 이미지 읽기 — `read_unicode()`

```python
def read_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)     # 한글 경로도 OK
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
```

`cv2.imread`는 한글 경로(`얼굴인식`)를 못 읽어서 우회한다.

### 3. 등록 DB 만들기 — `build_face_db()`

`faces_db/<이름>/` 폴더를 순회하며 각 사진에서 얼굴을 찾아 embedding을 뽑아 `{이름: [embedding...]}`을 만든다. 시작 시 한 번 실행.

```python
_, fs = yunet.detect(img)                 # 사진에서 얼굴 검출
f = max(fs, key=lambda r: r[2] * r[3])    # 가장 큰 얼굴 선택
aligned = recognizer.alignCrop(img, f)    # landmark로 정렬
feats.append(recognizer.feature(aligned)) # 128-d embedding
```

### 4. 신원 인식 — `identify()`

현재 프레임의 얼굴 하나를 등록 DB 전체와 비교해 가장 높은 유사도의 이름을 고른다. threshold 못 넘으면 `Unknown`.

```python
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
    return label, best_score      # 점수도 반환 (화면에 표시해 튜닝용)
```

### 5. 한글 이름 그리기 — `put_text_kr()`

`cv2.putText`는 한글을 `?????`로 그린다. PIL + 맑은고딕(`malgun.ttf`)으로 그려서 되돌린다.

```python
_kr_font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)

def put_text_kr(frame, text, pos, color_bgr):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    ImageDraw.Draw(pil).text(pos, text, font=_kr_font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
```

### 6. 손가락 개수 세기 — `count_fingers(lm)`

MediaPipe가 주는 손 관절 21개 좌표(`lm`)로 펴진 손가락을 센다.

**검지~새끼 (4개)** — 손끝(tip)이 두 번째 관절(pip)보다 위(y가 작음)에 있으면 펴짐:
```python
margin = 0.03   # 경계에서 깜빡임 방지: 이만큼 확실히 위일 때만 인정
for tip in (8, 12, 16, 20):
    if lm[tip].y < lm[tip - 2].y - margin:
        count += 1
```

**엄지 (1개)** — 방향 기반 판정 (좌우손·카메라 미러 무관):
```python
thumb_dir = lm[2].x - lm[17].x              # 엄지 바깥 방향 (새끼뿌리→엄지뿌리)
if (lm[4].x - lm[3].x) * thumb_dir > 0:     # 끝(4)이 그 바깥으로 더 나가면 펴짐
    count += 1
```

> MediaPipe 손 관절 번호: 4=엄지끝, 8=검지끝, 12=중지끝, 16=약지끝, 20=새끼끝. 각 손끝에서 −2 하면 그 손가락의 pip 관절.

### 7. 메인 루프 — 검출 → 인식 → 표시

```python
yunet.setInputSize((w, h))
_, yn_faces = yunet.detect(frame)
if yn_faces is not None:
    for f in yn_faces:
        bx, by, bw, bh = f[:4].astype(int)
        name, score = identify(frame, f)              # 누구인지
        recognized.append(((bx, by, bw, bh), name, score))

for (x, y, fw, fh), name, score in recognized:
    known = name != "Unknown"
    color = (0, 255, 0) if known else (0, 0, 255)      # 등록=초록, 모름=빨강
    cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
    frame = put_text_kr(frame, name + " " + str(round(score, 2)), (x, y - 30), color)
```

### 8. 트리거 로직

**모르는 얼굴 → 다른 창:**
```python
unknown_present = any(name == "Unknown" for _, name, _ in recognized)
if unknown_present and alt_tab_done == False:
    pyautogui.hotkey("alt", "tab")
    alt_tab_done = True          # 한 번만. 다시 하려면 Ctrl+Space로 리셋
```

**손가락 5개 → 직전 창 복귀:**
```python
if fingers == 5:
    open_hand_frames += 1        # 연속 카운트
else:
    open_hand_frames = 0
    hand_return_done = False     # 손 내리면 리셋 → 다시 펴면 또 동작

if open_hand_frames >= OPEN_HAND_THRESHOLD and not hand_return_done and alt_tab_done:
    pyautogui.hotkey("alt", "tab")   # 직전 창으로 토글백
    hand_return_done = True
```

---

## threshold 튜닝 — `COSINE_THRESHOLD`

등록 사진으로 측정한 실측값:

- **같은 사람** 유사도: 최소 0.606 ~ 평균 0.756
- **다른 사람** 유사도: 최대 0.411 ~ 평균 0.26

두 분포 사이(0.411 ~ 0.606)에 경계를 두면 된다. 현재 **`0.45`** 사용.

박스에 **이름 + 점수**가 같이 표시되므로, 실제로 찍히는 점수를 보고 조정한다.

| 증상 | 조정 |
|------|------|
| 본인인데 `Unknown` (점수가 threshold 근처) | `COSINE_THRESHOLD` ↓ (0.42까지) |
| 남인데 본인 이름 뜸 | `COSINE_THRESHOLD` ↑ (0.5~0.55) |
| **특정 각도(옆/위/아래)에서만 Unknown** | 그 각도 사진을 `capture.py`로 더 추가 (근본 해결) |
| 어두운 야자 조명서 놓침 | 그 조명에서 사진 추가 |

## 손동작 튜닝 포인트

| 증상 | 조정 |
|------|------|
| 손가락 깜빡임/오인식 | `count_fingers`의 `margin` ↑ (0.04~0.05) |
| 엄지가 거꾸로 (폈는데 안 셈 / 접었는데 셈) | 엄지 판정 부등호 `> 0` ↔ `< 0` |
| 손동작이 너무 민감/둔함 | `OPEN_HAND_THRESHOLD` 조정 |
