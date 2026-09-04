import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

# ---------- SETUP MEDIAPIPE TASKS API (HandLandmarker) ----------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

THUMB_TIP = 4
INDEX_TIP = 8

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand_skeleton(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 120), 1)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 255, 120), -1)
    return pts


# ---------- LIVE FILTERS ----------
def make_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def make_xray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    inv = cv2.equalizeHist(inv)
    edges = cv2.Canny(gray, 60, 150)
    ghost = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    ghost[:, :, 0] = np.clip(ghost[:, :, 0].astype(int) + 40, 0, 255)
    ghost[edges > 0] = (255, 255, 255)
    return ghost


def make_invert(frame):
    return cv2.bitwise_not(frame)


def make_edge_glow(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 130)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    colored = cv2.applyColorMap(edges, cv2.COLORMAP_SPRING)
    return cv2.bitwise_and(colored, colored, mask=edges)


def make_pixelate(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def make_glitch(frame):
    """Channel RGB digeser + potongan garis horizontal digeser acak -> datamosh."""
    h, w = frame.shape[:2]
    b, g, r = cv2.split(frame)
    shift = 10
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    out = cv2.merge([b, g, r])
    num_slices = 6
    slice_h = max(4, h // 30)
    for _ in range(num_slices):
        y = np.random.randint(0, h - slice_h) if h > slice_h else 0
        dx = np.random.randint(-30, 30)
        out[y:y + slice_h] = np.roll(out[y:y + slice_h], dx, axis=1)
    return out


def make_cartoon(frame):
    """Warna diratain (bilateral filter) + garis tepi hitam tegas -> komik."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


def make_neon(frame):
    """Garis neon cyan-magenta dengan glow/bloom di sekitarnya."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    colored = cv2.applyColorMap(edges, cv2.COLORMAP_COOL)
    glow = cv2.GaussianBlur(colored, (0, 0), 6)
    return cv2.add(colored, glow)


def make_sketch(frame):
    """Efek sketsa pensil berwarna, pakai modul photo bawaan OpenCV.

    Diproses di resolusi setengah lalu di-upscale lagi -> lebih ringan,
    dikombinasikan dengan crop bounding-box di render_window jadi jauh
    lebih cepat daripada sebelumnya (full frame + sigma_s besar).
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
    _, color_sketch = cv2.pencilSketch(small, sigma_s=40, sigma_r=0.07, shade_factor=0.05)
    return cv2.resize(color_sketch, (w, h), interpolation=cv2.INTER_LINEAR)


def make_duotone(frame):
    """Gradasi 2 warna (ungu gelap -> oranye terang) ala poster/film noir."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow = np.array([120, 30, 80], dtype=np.float32)     # BGR ungu tua
    highlight = np.array([40, 190, 255], dtype=np.float32)  # BGR oranye terang
    out = shadow[None, None, :] * (1 - gray[..., None]) + highlight[None, None, :] * gray[..., None]
    return out.astype(np.uint8)


def make_night_vision(frame):
    """Hijau night-vision + noise butiran + vignette gelap di pinggir."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    out = np.zeros_like(frame)
    noise = np.random.randint(0, 35, gray.shape, dtype=np.uint8)
    out[:, :, 1] = cv2.add(gray, noise)

    h, w = frame.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = np.clip(1 - (dist / max_dist) * 0.6, 0, 1)
    for c in range(3):
        out[:, :, c] = (out[:, :, c] * vignette).astype(np.uint8)
    return out


def make_kaleidoscope(frame):
    """Kuadran kiri-atas dicerminkan ke 3 sisi lain -> efek kaleidoskop simetris.

    Catatan: karena sekarang dipanggil dengan crop (bukan full frame),
    kuadran diambil dari crop tersebut -> efeknya jadi konsisten dengan
    isi jendela portal, bukan lagi dari pojok kiri-atas frame kamera.
    """
    h, w = frame.shape[:2]
    half_w, half_h = max(1, w // 2), max(1, h // 2)
    quadrant = cv2.resize(frame[0:half_h, 0:half_w], (half_w, half_h))
    top = np.hstack([quadrant, cv2.flip(quadrant, 1)])
    bottom = np.hstack([cv2.flip(quadrant, 0), cv2.flip(quadrant, -1)])
    result = np.vstack([top, bottom])
    # pastikan ukuran balik sama persis dengan crop asli (jaga2 pembulatan ganjil)
    return cv2.resize(result, (w, h))


LIVE_FILTERS = {
    "THERMAL": make_thermal,
    "XRAY": make_xray,
    "INVERT": make_invert,
    "EDGE": make_edge_glow,
    "PIXELATE": make_pixelate,
    "GLITCH": make_glitch,
    "CARTOON": make_cartoon,
    "NEON": make_neon,
    "SKETCH": make_sketch,
    "DUOTONE": make_duotone,
    "NIGHTVISION": make_night_vision,
    "KALEIDOSCOPE": make_kaleidoscope,
}

MODES = list(LIVE_FILTERS.keys())


# ---------- RENDER ISI KOTAK (window filter di posisi yang sama, no warp) ----------
def render_window(frame, filter_fn, quad_pts):
    h, w = frame.shape[:2]
    pts = np.int32(quad_pts)

    # bounding box dari quad, di-clamp ke batas frame
    bx, by, bw, bh = cv2.boundingRect(pts)
    x1 = max(bx, 0)
    y1 = max(by, 0)
    x2 = min(bx + bw, w)
    y2 = min(by + bh, h)
    bw, bh = x2 - x1, y2 - y1

    if bw <= 0 or bh <= 0:
        return frame

    # crop kecil -> filter jalan di sini, bukan di full frame
    crop = frame[y1:y2, x1:x2]
    filtered_crop = filter_fn(crop)

    if filtered_crop.shape[:2] != (bh, bw):
        filtered_crop = cv2.resize(filtered_crop, (bw, bh))

    # mask quad
    mask = np.zeros((bh, bw), dtype=np.uint8)
    local_pts = pts - [x1, y1]
    cv2.fillConvexPoly(mask, local_pts, 255)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(crop, crop, mask=mask_inv)
    fg = cv2.bitwise_and(filtered_crop, filtered_crop, mask=mask)
    result_crop = cv2.add(bg, fg)
    cv2.polylines(result_crop, [local_pts], True, (255, 255, 255), 2, cv2.LINE_AA)

    result = frame.copy()
    result[y1:y2, x1:x2] = result_crop
    return result


def get_quad_from_hands(hands_pts):
    if len(hands_pts) != 2:
        return None
    hands_sorted = sorted(hands_pts, key=lambda pts: pts[INDEX_TIP][0])
    left_hand, right_hand = hands_sorted[0], hands_sorted[1]

    def top_bottom(hand):
        a, b = hand[THUMB_TIP], hand[INDEX_TIP]
        return (a, b) if a[1] < b[1] else (b, a)

    left_top, left_bottom = top_bottom(left_hand)
    right_top, right_bottom = top_bottom(right_hand)
    return (left_top, right_top, right_bottom, left_bottom)


def pinch_distance(hand_pts):
    x1, y1 = hand_pts[THUMB_TIP]
    x2, y2 = hand_pts[INDEX_TIP]
    return math.hypot(x2 - x1, y2 - y1)


# ---------- GESTURE: pinch -> buka cepat -> ganti filter ----------
CLOSE_PINCH = 45
SPREAD_PINCH = 130
GESTURE_WINDOW = 1.5
COOLDOWN = 0.8
GRACE_PERIOD = 0.3


class GestureSwitcher:
    def __init__(self, modes):
        self.modes = modes
        self.mode_idx = 0
        self.state = "idle"
        self.primed_time = 0
        self.cooldown_until = 0
        self.last_dist = None
        self.last_seen = 0

    @property
    def mode(self):
        return self.modes[self.mode_idx]

    def update(self, pinch_dist, now):
        if pinch_dist is not None:
            self.last_dist = pinch_dist
            self.last_seen = now
        elif self.last_dist is not None and (now - self.last_seen) < GRACE_PERIOD:
            pinch_dist = self.last_dist
        else:
            pinch_dist = None

        status_text = ""

        if self.state == "cooldown":
            if now > self.cooldown_until:
                self.state = "idle"
            return status_text

        if pinch_dist is None:
            return status_text

        if self.state == "idle":
            if pinch_dist < CLOSE_PINCH:
                self.state = "primed"
                self.primed_time = now
            status_text = "Pinch (nempelin jempol+telunjuk) buat ganti filter"

        elif self.state == "primed":
            elapsed = now - self.primed_time
            if elapsed > GESTURE_WINDOW:
                self.state = "idle"
            elif pinch_dist > SPREAD_PINCH:
                self.mode_idx = (self.mode_idx + 1) % len(self.modes)
                self.state = "cooldown"
                self.cooldown_until = now + COOLDOWN
                status_text = f"Filter ganti -> {self.mode}"
            else:
                status_text = "Sekarang BUKA pinch-nya cepat!"

        return status_text


# ---------- MAIN LOOP ----------
def main():
    print("Mode yang aktif:", MODES)
    print("(pencet angka 1-9 buat lompat langsung ke mode 1-9)")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    switcher = GestureSwitcher(MODES)
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((now - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_pts = []
        pinch_values = []
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                pts = draw_hand_skeleton(frame, landmarks, w, h)
                hands_pts.append(pts)
                pd = pinch_distance(pts)
                pinch_values.append(pd)

                tx, ty = pts[THUMB_TIP]
                ix, iy = pts[INDEX_TIP]
                color = (0, 255, 0) if pd < CLOSE_PINCH else \
                        (0, 100, 255) if pd > SPREAD_PINCH else (200, 200, 200)
                cv2.line(frame, (tx, ty), (ix, iy), color, 2)
                cv2.putText(frame, f"{int(pd)}px", ((tx + ix) // 2, (ty + iy) // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        quad = get_quad_from_hands(hands_pts)
        min_pinch = min(pinch_values) if pinch_values else None
        status_text = switcher.update(min_pinch, now)

        if quad:
            frame = render_window(frame, LIVE_FILTERS[switcher.mode], quad)
        else:
            cv2.putText(frame, "Angkat 2 tangan (jempol+telunjuk) buat bikin frame",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.putText(frame, f"Mode: {switcher.mode}  ({switcher.mode_idx + 1}/{len(MODES)})",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if status_text:
            cv2.putText(frame, status_text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        cv2.imshow("Hand Portal FX - tekan q untuk keluar", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(MODES):
                switcher.mode_idx = idx

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()