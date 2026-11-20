"""
MaixCam obstacle guide  -  sends the CUE and a SCENE summary to the ESP32.

What it does
1. Measures the distance (in metres) to whatever is in front of the camera, in three zones:
   LEFT / CENTRE / RIGHT (YOLO26-depth, Depth-Anything or the YOLO fallback, as before).
2. Decides the cue: "left", "right" or "straight" (or "none" if SAY_STRAIGHT_WHEN_CLEAR is False
   and the path is clear).
3. Sends the cue to the ESP32 over UART about 10 times per second (and immediately when it
   changes). The ESP32 turns the cue into the 3 vibration motors. If the ESP32 stops receiving
   the cue it blinks all motors, so the user knows the camera link is lost.
4. After every object / face detection it also sends a short SCENE line: who / what is in front,
   on which side, and how far. The ESP32 keeps the latest one. When the user presses the button
   the ESP32 reads it out loud to the Bluetooth earphone, e.g. "left nuel, front carmen,
   right table two point five meters".
   The MaixCam does NOT send any audio now. The words are stored inside the ESP32.

Messages (text lines, 115200 baud, NMEA-style checksum = XOR of every character between $ and *):
   $CUE,straight*5B
   $SCN;L,P,nuel,-1;F,P,carmen,22;R,O,table,25*XX
   SCN items:  side (L/F/R), kind (P = person, O = object), label, distance in tenths of a
   metre (-1 = unknown).

Faces: run teach_faces.py first (names Carmen, Nuel, Niko, Andrew, in that order). Only these
names are spoken by name; any other face is spoken as "person".

Wiring: MaixCam pin U2T (TX) -> ESP32 D33,  MaixCam GND -> ESP32 GND.   (U2R is not needed.)
"""
import os
import re
import time

import numpy as np
from maix import app, camera, display, image, nn, pinmap, touchscreen, uart

# ======================= SETTINGS (edit these) =======================
# ---- distance / decision ----
WARN_DIST_M = 3.0             # closer than this (metres) counts as "in the way"
HYSTERESIS_M = 0.3            # a zone that is "in the way" is only cleared again when it is
                              # WARN_DIST_M + this far away. Stops the cue flickering when a
                              # wall sits right at the limit. 0 = off (old behaviour).
SAY_STRAIGHT_WHEN_CLEAR = True  # True: say "straight" when nothing is in the way.
                              # False: stay silent when the path is clear.
CONFIRM_FRAMES = 3            # a new cue must be seen this many frames in a row (avoids flicker)
CUE_SEND_EVERY_S = 0.1        # the cue is re-sent this often (the ESP32 needs it as a heartbeat)

# ---- where to look in the image (fractions of the image) ----
LANE = (0.33, 0.67)           # left | centre | right split
ROW_BAND = (0.25, 0.70)       # ignore top (ceiling) and bottom (floor) of the picture
NEAR_PERCENTILE = 10          # depth: "nearest" = the 10th-percentile distance in a zone
SMOOTH = 0.5                  # 0..1, higher = reacts faster, lower = steadier numbers

# ---- camera ----
CAM_W, CAM_H = 640, 480

# ---- object names on screen (what is in front) ----
OBJECTS_ENABLED = True
OBJ_MODEL_CLASS = "YOLO11"    # use "YOLOv5" for a model you trained on MaixHub
OBJ_MODEL_PATH = "/root/models/yolo11n.mud"
OBJ_CONF = 0.5

# ---- face recognition ----
FACES_ENABLED = True
FACE_DETECT_MODEL = "/root/models/yolov8n_face.mud"
FACE_FEATURE_MODEL = "/root/models/insghtface_webface_r50.mud"
FACES_FILE = "/root/faces.bin"
FACE_NAMES = ["Carmen", "Nuel", "Niko", "Andrew"]   # given in order, one per screen tap
FACE_MATCH_TH = 0.85          # higher = stricter match
LEARN_NAME = None             # To teach a face WITHOUT the touchscreen: write a name here,
                              # e.g. "Mum", run the app and look at the camera.
                              # The first new face is saved with that name.
                              # Then set it back to None.

RUN_AI_EVERY = 3              # run object/face detection every N frames (keeps it fast)

# ---- what the ESP32 may say when the button is pressed ----
# Only faces with a name in FACE_NAMES are spoken by name; every other face is "person".
# SPOKEN: object label from the detector -> word the ESP32 says. Objects NOT in this list are
# not announced. The words (values) are also the audio clips: tools/make_clips.py reads this
# table, so after you add a word here, run make_clips.py again and re-upload the ESP32 sketch.
SPOKEN = {
    "person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle",
    "bus": "bus", "truck": "truck", "fire hydrant": "hydrant", "bench": "bench",
    "dog": "dog", "cat": "cat", "umbrella": "umbrella", "backpack": "backpack",
    "suitcase": "suitcase", "chair": "chair", "couch": "sofa", "potted plant": "plant",
    "bed": "bed", "dining table": "table", "tv": "television", "laptop": "laptop",
    "refrigerator": "fridge", "toilet": "toilet", "sink": "sink", "bottle": "bottle",
}
MAX_SPEAK_M = 9.5             # things farther than this are not announced
SCENE_MAX_ITEMS = 4           # at most this many items per announcement (nearest first)

# ---- backend: "auto" tries depth first, then YOLO ----
BACKEND = "auto"              # "auto", "depth" (YOLO26-depth), "depthanything" or "yolo"
DEPTH_MODEL_PATH = "/root/models/yolo26n_depth_640x480.mud"
CAL_A, CAL_B = 1.0, 0.0       # depth calibration: d_real = exp(CAL_A * log(d_raw) + CAL_B)
DA_MODEL_PATH = "/root/models/depth_anything_v2_vits.mud"
DA_HIGHER_IS_NEARER = True    # Depth-Anything raw values are normally "bigger = closer".
                              # If the on-screen raw number goes UP when you walk AWAY, set False.
DA_K = 6.0                    # distance_m = DA_K / raw.  CALIBRATE: stand 2 m from a wall,
                              # read "raw" on screen, set DA_K = 2.0 * raw.
YOLO_MODEL_PATH = "/root/models/yolov5s.mud"
FOCAL_PX = 600.0              # YOLO fallback only: calibrate (dist = real_width * FOCAL_PX / box_px)
DEFAULT_WIDTH_M = 0.5
REAL_WIDTH_M = {
    "person": 0.5, "chair": 0.5, "bicycle": 0.6, "motorcycle": 0.8,
    "car": 1.8, "bus": 2.5, "truck": 2.5, "bench": 1.2, "dog": 0.4,
    "couch": 1.8, "bed": 1.5, "potted plant": 0.4, "dining table": 1.2,
    "tv": 1.0, "refrigerator": 0.7, "toilet": 0.4, "sink": 0.6,
}

# ---- UART to the ESP32 (pins reported valid by this device) ----
UART_DEV = "/dev/ttyS2"       # the header pins U2T / U2R = UART2 = /dev/ttyS2
UART_TX_PIN = "B0"            # printed "U2T" on the board
UART_RX_PIN = "B1"            # printed "U2R" on the board (not used yet)
BAUD = 115200                 # must match MAIX_BAUD in the ESP32 sketch
# =====================================================================

FAR_M = 99.0

# ---------------- UART ----------------
def _uart_function(pin, kind):
    """Ask the camera which UART function this pin has ("UART2_TX" ...). kind = "TX" or "RX"."""
    try:
        funcs = list(pinmap.get_pin_functions(pin))
    except Exception as e:
        print("pinmap: cannot list functions of", pin, e)
        funcs = []
    for f in funcs:
        if "UART" in f.upper() and kind in f.upper():
            return f
    print("pinmap: no UART %s function found on %s. Functions: %s" % (kind, pin, funcs))
    return "UART2_" + kind


for _pin, _kind in ((UART_TX_PIN, "TX"), (UART_RX_PIN, "RX")):
    _func = _uart_function(_pin, _kind)
    try:
        pinmap.set_pin_function(_pin, _func)
        print("pin", _pin, "->", _func)
    except Exception as e:
        print("pinmap warning:", _pin, _func, e)
serial = uart.UART(UART_DEV, BAUD)


# ---------------- UART sender ----------------
def write_all(data):
    """Send every byte. If the UART is busy, wait a moment and try again."""
    sent, tries = 0, 0
    while sent < len(data):
        n = serial.write(data[sent:])
        if n is None or n < 0:
            n = 0
        sent += n
        if sent < len(data):
            tries += 1
            if tries > 400:
                raise RuntimeError("UART is stuck")
            time.sleep(0.005)


def make_frame(body):
    """'CUE,left' -> b'$CUE,left*HH\n'  (HH = XOR of every character of the body, in hex)."""
    cs = 0
    for ch in body.encode():
        cs ^= ch
    return ("$%s*%02X\n" % (body, cs)).encode()


def send_line(body):
    try:
        write_all(make_frame(body))
    except Exception as e:                       # never let a UART problem stop the camera loop
        print("send error:", e)


def send_cue(cue):
    send_line("CUE," + (cue if cue else "none"))


# ---------------- what / who is in front, as a short list for the ESP32 ----------------
def side_of(cx, img_w):
    """L / F / R from the horizontal position (same lanes as the distance zones)."""
    return "L" if cx < img_w * LANE[0] else ("R" if cx > img_w * LANE[1] else "F")


def build_scene(obj_items, face_items, dist_map, img_w, img_h, box_dist):
    """Merge object boxes and face boxes into [(side, kind, label, metres or None), ...].
    kind: "P" = person (name or "person"), "O" = object. Nearest first, at most SCENE_MAX_ITEMS."""
    names = [n.lower() for n in FACE_NAMES]

    def known(label):
        return label.lower() if label.lower() in names else None

    found, used_faces = [], set()
    for (x, y, w, h, label) in obj_items:                 # 1) people seen by the object detector
        if label != "person":
            continue
        who = None
        for i, (fx, fy, fw, fh, fl) in enumerate(face_items):
            if x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h:
                used_faces.add(i)                         # this face belongs to this person
                who = who or known(fl)
        found.append((side_of(x + w / 2, img_w), "P", who or "person",
                      box_dist(dist_map, x, y, w, h, img_w, img_h)))
    for i, (fx, fy, fw, fh, fl) in enumerate(face_items):  # 2) faces without a person box
        if i not in used_faces:
            found.append((side_of(fx + fw / 2, img_w), "P", known(fl) or "person",
                          box_dist(dist_map, fx, fy, fw, fh, img_w, img_h)))
    for (x, y, w, h, label) in obj_items:                 # 3) everyday objects
        word = SPOKEN.get(label)
        if word and label != "person":
            found.append((side_of(x + w / 2, img_w), "O", word,
                          box_dist(dist_map, x, y, w, h, img_w, img_h)))

    best = {}                                              # same thing twice on one side -> keep nearest
    for side, kind, label, d in found:
        if d is not None and d > MAX_SPEAK_M:
            continue
        label = re.sub(r"[^a-z0-9_]", "", label.lower().replace(" ", "_"))
        if not label:
            continue
        key = (side, label)
        if key not in best or (d if d is not None else 1e9) < (best[key][3] if best[key][3] is not None else 1e9):
            best[key] = (side, kind, label, d)
    items = sorted(best.values(), key=lambda t: t[3] if t[3] is not None else 1e9)
    return items[:SCENE_MAX_ITEMS]


def scene_body(items):
    parts = ["SCN"]
    for side, kind, label, d in items:
        parts.append("%s,%s,%s,%d" % (side, kind, label, -1 if d is None else int(round(d * 10))))
    return ";".join(parts)


# ---------------- decision logic ----------------
_blocked = [False, False, False]      # remembers which zones were "in the way" (for hysteresis)
_last_side = None                     # last left/right choice (for hysteresis)


def decide(d):
    """d = (left, centre, right) distances in metres. Returns the direction to GO."""
    global _last_side
    dl, dc, dr = d
    for i in range(3):
        limit = WARN_DIST_M + (HYSTERESIS_M if _blocked[i] else 0.0)
        _blocked[i] = d[i] < limit
    L, C, R = _blocked
    if not (L or C or R):
        return "straight" if SAY_STRAIGHT_WHEN_CLEAR else None
    if C:                                   # something right in front: go to the freer side
        if _last_side and abs(dl - dr) < HYSTERESIS_M:
            return _last_side               # both sides about equally free: keep the old choice
        _last_side = "left" if dl >= dr else "right"
        return _last_side
    if R and not L:                         # obstacle on the right -> go left
        return "left"
    if L and not R:                         # obstacle on the left -> go right
        return "right"
    return "straight"                       # both sides close but the centre is free


# ---------------- distance backends ----------------
def zone_distances(dist_arr):
    """dist_arr: 2-D array of distances in metres -> [left, centre, right] nearest distance."""
    H, W = dist_arr.shape
    r0, r1 = int(H * ROW_BAND[0]), int(H * ROW_BAND[1])
    edges = [0, int(W * LANE[0]), int(W * LANE[1]), W]
    out = []
    for i in range(3):
        zone = dist_arr[r0:r1:4, edges[i]:edges[i + 1]:4]
        zone = zone[np.isfinite(zone)]
        out.append(float(np.percentile(zone, NEAR_PERCENTILE)) if zone.size else FAR_M)
    return out


def array_shape(model, n):
    """Find (height, width) of a flat depth output with n values."""
    for wn, hn in (("output_width", "output_height"), ("input_width", "input_height")):
        if hasattr(model, wn) and hasattr(model, hn):
            h, w = getattr(model, hn)(), getattr(model, wn)()
            if h * w == n:
                return h, w
    raise RuntimeError("cannot work out the depth map size (%d values)" % n)


class DepthBackend:
    name = "depth"

    def __init__(self):
        import maix.tensor as mt
        self.mt = mt
        self.model = nn.YOLO26Depth(model=DEPTH_MODEL_PATH, dual_buff=False)
        self.w = self.model.input_width()
        self.h = self.model.input_height()
        self.fmt = self.model.input_format()
        self.debug = ""
        self.last_dist = None

    def measure(self, img):
        depth = self.model.get_depth(img, image.Fit.FIT_CONTAIN, CAL_A, CAL_B)
        if depth is None:
            return None
        try:
            arr = self.mt.tensor_to_numpy_float32(depth).reshape(
                self.model.output_height(), self.model.output_width())
            zd = zone_distances(arr)
            self.last_dist = arr.copy()
        finally:
            del depth
        return zd


class DepthAnythingBackend:
    """EXPERIMENTAL: relative depth turned into metres with one number (DA_K)."""
    name = "depthanything"

    def __init__(self):
        import maix.tensor as mt
        self.mt = mt
        self.model = nn.DepthAnything(model=DA_MODEL_PATH, dual_buff=False)
        self.w = self.model.input_width()
        self.h = self.model.input_height()
        self.fmt = self.model.input_format()
        self.debug = ""
        self.last_dist = None

    def measure(self, img):
        depth = self.model.get_depth(img, image.Fit.FIT_CONTAIN)
        if depth is None:
            return None
        try:
            flat = self.mt.tensor_to_numpy_float32(depth)
            h, w = array_shape(self.model, flat.size)
            raw = flat.reshape(h, w).astype(np.float32)
            if DA_HIGHER_IS_NEARER:
                dist = DA_K / np.maximum(raw, 1e-3)
            else:
                dist = DA_K * raw
            zd = zone_distances(dist)
            self.last_dist = dist
            r0, r1 = int(h * ROW_BAND[0]), int(h * ROW_BAND[1])
            c0, c1 = int(w * LANE[0]), int(w * LANE[1])
            self.debug = "raw(centre) %.2f" % float(np.median(raw[r0:r1:4, c0:c1:4]))
        finally:
            del depth
        return zd


class YoloBackend:
    name = "yolo"

    def __init__(self):
        self.det = nn.YOLOv5(model=YOLO_MODEL_PATH)
        self.w = self.det.input_width()
        self.h = self.det.input_height()
        self.fmt = self.det.input_format()
        self.debug = ""
        self.last_dist = None

    def measure(self, img):
        W = img.width()
        edges = [0, W * LANE[0], W * LANE[1], W]
        zd = [FAR_M, FAR_M, FAR_M]
        for o in self.det.detect(img, conf_th=0.5, iou_th=0.45):
            label = self.det.labels[o.class_id]
            dist = REAL_WIDTH_M.get(label, DEFAULT_WIDTH_M) * FOCAL_PX / max(o.w, 1)
            for i in range(3):
                if o.x + o.w > edges[i] and o.x < edges[i + 1]:
                    zd[i] = min(zd[i], dist)
            close = dist < WARN_DIST_M
            color = image.COLOR_RED if close else image.COLOR_GREEN
            img.draw_rect(o.x, o.y, o.w, o.h, color=color)
            img.draw_string(o.x, max(o.y - 12, 0), "%s %.1fm" % (label, dist), color=color)
        return zd


def make_backend():
    try:
        print("models on device:", os.listdir("/root/models"))
    except Exception:
        pass
    if BACKEND in ("auto", "depth"):
        try:
            b = DepthBackend()
            print("USING: YOLO26-depth (metric distance; sees walls, doors, stairs, any obstacle)")
            return b
        except Exception as e:
            print("YOLO26-depth not available:", e)
            if BACKEND == "depth":
                raise
    if BACKEND in ("auto", "depthanything"):
        try:
            b = DepthAnythingBackend()
            print("USING: Depth-Anything-V2 (EXPERIMENTAL; relative depth, calibrate DA_K)")
            return b
        except Exception as e:
            print("Depth-Anything not available:", e)
            if BACKEND == "depthanything":
                raise
    b = YoloBackend()
    print("USING: YOLO fallback (only knows COCO objects, NOT walls/doors/stairs)")
    return b


# ---------------- what / who is in front ----------------
def box_distance(dist_map, x, y, w, h, img_w, img_h):
    """Distance (m) to the box region, read from the depth map. None if unknown."""
    if dist_map is None:
        return None
    dh, dw = dist_map.shape
    x0, x1 = max(int(x * dw / img_w), 0), min(int((x + w) * dw / img_w) + 1, dw)
    y0, y1 = max(int(y * dh / img_h), 0), min(int((y + h) * dh / img_h) + 1, dh)
    reg = dist_map[y0:y1, x0:x1]
    reg = reg[np.isfinite(reg)]
    return float(np.percentile(reg, 25)) if reg.size else None


class ObjectNamer:
    """Names everyday objects (chair, couch, tv, person...). Items: (x, y, w, h, label)."""

    def __init__(self):
        self.det = getattr(nn, OBJ_MODEL_CLASS)(model=OBJ_MODEL_PATH, dual_buff=False)
        self.items = []

    def update(self, img):
        self.items = []
        for o in self.det.detect(img, conf_th=OBJ_CONF, iou_th=0.45):
            self.items.append((o.x, o.y, o.w, o.h, self.det.labels[o.class_id]))


class FaceID:
    """Recognises known faces. Tap the screen to learn the unknown face in view."""

    def __init__(self):
        self.rec = nn.FaceRecognizer(detect_model=FACE_DETECT_MODEL,
                                     feature_model=FACE_FEATURE_MODEL, dual_buff=False)
        if os.path.exists(FACES_FILE):
            self.rec.load_faces(FACES_FILE)
        print("known people:", self.rec.labels[1:])
        self.items = []
        self.learn_next = False
        self.pending_name = LEARN_NAME
        self.was_pressed = False
        self.msg, self.msg_t = "", 0.0
        if self.pending_name:
            self.msg, self.msg_t = "look at the camera to learn: " + self.pending_name, time.time()
        try:
            self.ts = touchscreen.TouchScreen()
        except Exception as e:
            print("touchscreen not available, cannot learn faces:", e)
            self.ts = None

    def poll_touch(self):
        if self.ts is None:
            return
        _x, _y, pressed = self.ts.read()
        if pressed and not self.was_pressed:
            self.learn_next = True
            self.msg, self.msg_t = "show ONE face to learn it", time.time()
        self.was_pressed = bool(pressed)

    def update(self, img):
        faces = self.rec.recognize(img, 0.5, 0.45, FACE_MATCH_TH, True)
        self.items = []
        unknown = []
        for f in faces:
            self.items.append((f.x, f.y, f.w, f.h, self.rec.labels[f.class_id]))
            if f.class_id == 0:
                unknown.append(f)
        if (self.learn_next or self.pending_name) and unknown:
            f = max(unknown, key=lambda q: q.w * q.h)
            idx = len(self.rec.labels) - 1        # labels[0] is "unknown"
            name = self.pending_name or (
                FACE_NAMES[idx] if idx < len(FACE_NAMES) else "person_%d" % (idx + 1))
            self.rec.add_face(f, name)
            self.rec.save_faces(FACES_FILE)
            self.learn_next = False
            self.pending_name = None
            self.msg, self.msg_t = "learned: " + name, time.time()
            print("learned face:", name)


def describe(items, dist_map, W, H):
    """Plain-text list like 'chair 1.8m (left); dining table 2.4m (centre)'."""
    parts = []
    for (x, y, w, h, label) in items:
        cx = x + w / 2
        side = "left" if cx < W * LANE[0] else ("right" if cx > W * LANE[1] else "centre")
        d = box_distance(dist_map, x, y, w, h, W, H)
        parts.append("%s %s (%s)" % (label, "?" if d is None else "%.1fm" % d, side))
    return "; ".join(parts)


def make_helpers(backend):
    namer = faceid = None
    if OBJECTS_ENABLED and backend.name != "yolo":
        try:
            namer = ObjectNamer()
            print("object names: ON (%s)" % OBJ_MODEL_PATH)
        except Exception as e:
            print("object names OFF:", e)
    if FACES_ENABLED:
        try:
            faceid = FaceID()
            print("face recognition: ON (tap the screen to learn a face)")
        except Exception as e:
            print("face recognition OFF:", e)
    return namer, faceid


# ---------------- main ----------------
backend = make_backend()
namer, faceid = make_helpers(backend)
cam = camera.Camera(CAM_W, CAM_H)
disp = display.Display()
frame_no = 0

zd = None                 # smoothed zone distances
candidate, count, active = None, 0, None
last_nearest, last_t, closing = None, time.time(), 0.0
last_print = 0.0
last_cue_sent = 0.0
scene_text = "SCN"

while not app.need_exit():
    img = cam.read()
    frame_no += 1
    raw = backend.measure(img)
    if raw is None:
        disp.show(img)
        continue

    # smooth the numbers so they don't jump around
    zd = raw if zd is None else [SMOOTH * r + (1 - SMOOTH) * s for r, s in zip(raw, zd)]

    # how fast is the nearest thing getting closer? (m/s, positive = approaching)
    now = time.time()
    nearest = min(zd)
    if last_nearest is not None and now > last_t:
        inst = (last_nearest - nearest) / (now - last_t)
        closing = 0.7 * closing + 0.3 * inst
    last_nearest, last_t = nearest, now

    # pick the cue, and only switch after it has been stable for a few frames
    cue = decide(zd)
    if cue == candidate:
        count += 1
    else:
        candidate, count = cue, 1
    if count >= CONFIRM_FRAMES and candidate != active:
        active = candidate
        print("cue ->", active)
        send_cue(active)                      # a change goes out immediately
        last_cue_sent = now
    elif now - last_cue_sent >= CUE_SEND_EVERY_S:
        send_cue(active)                      # otherwise repeat it: this is the ESP32's heartbeat
        last_cue_sent = now

    # draw the zones and numbers on the screen
    W, H = img.width(), img.height()
    edges = [0, int(W * LANE[0]), int(W * LANE[1]), W]
    y0, y1 = int(H * ROW_BAND[0]), int(H * ROW_BAND[1])
    for i, name in enumerate(("L", "C", "R")):
        near = zd[i] < WARN_DIST_M
        color = image.COLOR_RED if near else image.COLOR_GREEN
        img.draw_rect(edges[i], y0, edges[i + 1] - edges[i], y1 - y0, color=color, thickness=2)
        txt = "%s %.1fm" % (name, zd[i]) if zd[i] < FAR_M else name + " --"
        img.draw_string(edges[i] + 4, y0 + 4, txt, color=color)
    # names of objects and faces, each with its distance from the depth map
    if frame_no % RUN_AI_EVERY == 0:
        if namer is not None:
            namer.update(img)
        if faceid is not None:
            faceid.update(img)
        if namer is not None or faceid is not None:
            scene_text = scene_body(build_scene(
                namer.items if namer is not None else [],
                faceid.items if faceid is not None else [],
                backend.last_dist, W, H, box_distance))
            send_line(scene_text)
    if faceid is not None:
        faceid.poll_touch()
    for helper, rgb in ((namer, (255, 200, 0)), (faceid, (0, 140, 255))):
        if helper is None:
            continue
        col = image.Color.from_rgb(*rgb)
        for (x, y, w, h, label) in helper.items:
            d = box_distance(backend.last_dist, x, y, w, h, W, H)
            img.draw_rect(x, y, w, h, color=col, thickness=2)
            img.draw_string(x, max(y - 14, 0), label if d is None else "%s %.1fm" % (label, d), color=col)
    if faceid is not None and faceid.msg and now - faceid.msg_t < 4:
        img.draw_string(4, 44, faceid.msg, color=image.COLOR_WHITE)

    img.draw_string(4, 4, "cue: %s" % active, color=image.COLOR_WHITE)
    if backend.debug:
        img.draw_string(4, H - 16, backend.debug, color=image.COLOR_WHITE)
    if closing > 0.3:
        img.draw_string(4, 24, "closing %.1f m/s" % closing, color=image.COLOR_WHITE)
    disp.show(img)

    if now - last_print > 1.0:
        last_print = now
        print("L %.1f  C %.1f  R %.1f m | cue %s | closing %.1f m/s | %s" % (
            zd[0], zd[1], zd[2], active, closing, backend.debug))
        print("   sent to ESP32:", scene_text)
        for title, helper in (("objects", namer), ("faces", faceid)):
            if helper is not None and helper.items:
                print("   %s: %s" % (title, describe(helper.items, backend.last_dist, W, H)))
