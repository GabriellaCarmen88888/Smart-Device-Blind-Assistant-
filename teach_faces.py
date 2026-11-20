"""
FACE TEACHER  -  run THIS file (not main.py) to teach the MaixCam who is who.

How to use
1. Write the names of the people at the top (NAMES). They are taught in this order.
2. Run this file from MaixVision.
3. Person number 1 stands about 50 cm from the camera, looks straight at it, in good light.
4. TAP THE TOUCHSCREEN once. The screen says "LEARNED: <name>".
5. Next person: stand in front of the camera and tap again.
6. The faces are saved in /root/faces.bin automatically.
7. Test it: the boxes show the name when the camera knows the face.
8. Stop this app. Now run main.py. It loads /root/faces.bin by itself.

Use the SAME model settings here and in main.py (they are already the same).
"""
import os
import time

from maix import app, camera, display, image, nn, touchscreen

# ================= settings =================
NAMES = ["Carmen", "Nuel", "Niko", "Andrew"]   # people to teach, in this order (same as FACE_NAMES in main.py)
FACES_FILE = "/root/faces.bin"
FACE_DETECT_MODEL = "/root/models/yolov8n_face.mud"
FACE_FEATURE_MODEL = "/root/models/insghtface_webface_r50.mud"
FACE_MATCH_TH = 0.85                  # higher = stricter match
CLEAR_ALL_FIRST = False               # True = forget everyone saved before and start again
# ============================================

if CLEAR_ALL_FIRST and os.path.exists(FACES_FILE):
    os.remove(FACES_FILE)
    print("old faces deleted")

rec = nn.FaceRecognizer(detect_model=FACE_DETECT_MODEL,
                        feature_model=FACE_FEATURE_MODEL, dual_buff=False)
if os.path.exists(FACES_FILE):
    rec.load_faces(FACES_FILE)
print("known people:", rec.labels[1:])      # labels[0] is "unknown"

cam = camera.Camera(640, 480)
disp = display.Display()
ts = touchscreen.TouchScreen()

msg, msg_t = "", 0.0
was_pressed = False

while not app.need_exit():
    img = cam.read()
    faces = rec.recognize(img, 0.5, 0.45, FACE_MATCH_TH, True)

    # a "tap" = the finger just touched the screen
    _x, _y, pressed = ts.read()
    tapped = bool(pressed) and not was_pressed
    was_pressed = bool(pressed)

    known_count = len(rec.labels) - 1
    next_name = NAMES[known_count] if known_count < len(NAMES) else None

    for f in faces:
        known = f.class_id != 0
        color = image.COLOR_GREEN if known else image.COLOR_RED
        img.draw_rect(f.x, f.y, f.w, f.h, color=color, thickness=2)
        img.draw_string(f.x, max(f.y - 14, 0), rec.labels[f.class_id] if known else "unknown", color=color)

    if tapped:
        if next_name is None:
            msg = "everyone in NAMES is already taught"
        elif len(faces) != 1:
            msg = "need exactly ONE face in view (I see %d)" % len(faces)
        elif faces[0].class_id != 0:
            msg = "this face is already known as " + rec.labels[faces[0].class_id]
        else:
            rec.add_face(faces[0], next_name)
            rec.save_faces(FACES_FILE)
            msg = "LEARNED: " + next_name
            print(msg, "-> saved to", FACES_FILE)
        msg_t = time.time()

    if next_name is not None:
        img.draw_string(4, 4, "next to teach: %s  (tap the screen)" % next_name, color=image.COLOR_WHITE)
    else:
        img.draw_string(4, 4, "all names taught. You can stop this app.", color=image.COLOR_WHITE)
    if msg and time.time() - msg_t < 4:
        img.draw_string(4, 24, msg, color=image.COLOR_WHITE)
    disp.show(img)
