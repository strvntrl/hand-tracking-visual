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
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)
landmarker = HandLandmarker.create_from_options(options)

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
DETECT_SCALE = 0.5

def get_hand_points(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def make_neon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    colored = cv2.applyColorMap(edges, cv2.COLORMAP_COOL)
    glow = cv2.GaussianBlur(colored, (0, 0), 6)
    return cv2.add(colored, glow)

def make_sketch(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
    _, color_sketch = cv2.pencilSketch(small, sigma_s=40, sigma_r=0.07, shade_factor=0.05)
    return cv2.resize(color_sketch, (w, h), interpolation=cv2.INTER_LINEAR)

def make_duotone(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow = np.array([120, 30, 80], dtype=np.float32)     # BGR ungu tua
    highlight = np.array([40, 190, 255], dtype=np.float32)  # BGR oranye terang
    out = shadow[None, None, :] * (1 - gray[..., None]) + highlight[None, None, :] * gray[..., None]
    return out.astype(np.uint8)

def make_night_vision(frame):
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

_HALFTONE_CACHE = {}
_POINT_NOISE_CACHE = {}
_CACHE_CAP = 40

def _get_halftone_mask(h, w, cell=6):
    key = (h, w, cell)
    if key in _HALFTONE_CACHE:
        return _HALFTONE_CACHE[key]
    if len(_HALFTONE_CACHE) >= _CACHE_CAP:
        _HALFTONE_CACHE.clear()
    mask = np.zeros((h, w), dtype=np.float32)
    radius = cell // 2 - 1
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            cv2.circle(mask, (x + cell // 2, y + cell // 2), max(1, radius), 1.0, -1)
    _HALFTONE_CACHE[key] = mask
    return mask

def _get_point_noise(h, w):
    key = (h, w)
    if key in _POINT_NOISE_CACHE:
        return _POINT_NOISE_CACHE[key]
    if len(_POINT_NOISE_CACHE) >= _CACHE_CAP:
        _POINT_NOISE_CACHE.clear()
    noise = np.random.rand(h, w).astype(np.float32)
    _POINT_NOISE_CACHE[key] = noise
    return noise

def make_spiderverse(frame):
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7
    )
    edges = cv2.erode(edges, np.ones((2, 2), np.uint8))
    toon = cv2.bitwise_and(vivid, vivid, mask=edges)
    toon[edges == 0] = (0, 0, 0)

    dot_mask = _get_halftone_mask(h, w, cell=6)
    darkness = 1.0 - (gray.astype(np.float32) / 255.0)
    halftone_strength = np.clip(darkness * 1.4, 0, 1) * dot_mask
    halftone_overlay = (halftone_strength[..., None] * np.array([20, 20, 20])).astype(np.uint8)
    toon = cv2.subtract(toon, halftone_overlay)

    b, g, r = cv2.split(toon)
    shift = max(2, w // 220)
    r_ghost = np.roll(r, shift, axis=1)
    b_ghost = np.roll(b, -shift, axis=1)
    ghosted = cv2.merge([b_ghost, g, r_ghost])
    result = cv2.addWeighted(toon, 0.7, ghosted, 0.3, 0)

    return result

def make_emboss(frame):
    kernel = np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2],
    ], dtype=np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    embossed = cv2.filter2D(gray, -1, kernel)
    embossed = np.clip(embossed + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

def make_hologram(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tinted = cv2.merge([gray, gray, (gray * 0.15).astype(np.uint8)])

    h, w = tinted.shape[:2]
    scan = tinted.copy()
    scan[::3, :, :] = (scan[::3, :, :] * 0.45).astype(np.uint8)

    glow = cv2.GaussianBlur(scan, (0, 0), 4)
    result = cv2.addWeighted(scan, 0.75, glow, 0.5, 0)

    noise = np.random.randint(0, 15, (h, w), dtype=np.uint8)
    result[:, :, 0] = cv2.add(result[:, :, 0], noise)
    result[:, :, 1] = cv2.add(result[:, :, 1], noise)
    return result

def make_drawn(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    inv_blur = 255 - blur
    inv_blur[inv_blur == 0] = 1 
    sketch = cv2.divide(gray, inv_blur, scale=256)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    color_wash = cv2.bilateralFilter(frame, 9, 60, 60)
    result = cv2.addWeighted(color_wash, 0.35, sketch_bgr, 0.65, 0)
    return result

def make_dotscan(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    noise = _get_point_noise(h, w)
    dot_mask = (noise < np.clip(gray, 0, 1) ** 1.2).astype(np.uint8) * 255
    dot_mask = cv2.dilate(dot_mask, np.ones((2, 2), np.uint8))

    warm_tint = np.array([60, 140, 255], dtype=np.float32) / 255.0  # BGR oranye hangat
    colored = (dot_mask.astype(np.float32) / 255.0)[..., None] * warm_tint[None, None, :] * 255
    return colored.astype(np.uint8)

def make_riso(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow = np.array([80, 40, 10], dtype=np.float32)      
    highlight = np.array([245, 235, 225], dtype=np.float32) 
    duo = shadow[None, None, :] * (1 - gray[..., None]) + highlight[None, None, :] * gray[..., None]
    duo = duo.astype(np.uint8)

    dot_mask = _get_halftone_mask(h, w, cell=5)
    darkness = 1.0 - gray
    strength = np.clip(darkness * 1.3, 0, 1) * dot_mask
    overlay = (strength[..., None] * np.array([40, 25, 10])).astype(np.uint8)
    result = cv2.subtract(duo, overlay)

    noise = np.random.randint(0, 12, (h, w), dtype=np.uint8)
    for c in range(3):
        result[:, :, c] = cv2.subtract(result[:, :, c], noise // 2)
    return result

def make_popart(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    dot_mask = _get_halftone_mask(h, w, cell=5)
    darkness = 1.0 - gray.astype(np.float32) / 255.0
    strength = np.clip(darkness * 1.6, 0, 1) * dot_mask

    white = np.array([255, 255, 255], dtype=np.float32)
    red = np.array([0, 0, 255], dtype=np.float32)  # BGR merah
    result = white[None, None, :] * (1 - strength[..., None]) + red[None, None, :] * strength[..., None]
    return result.astype(np.uint8)

def make_acid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    levels = 5
    step = 256 // levels
    quant = (gray // step) * step
    colored = cv2.applyColorMap(quant, cv2.COLORMAP_SUMMER)

    b, g, r = cv2.split(colored)
    shift = max(2, gray.shape[1] // 150)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    out = cv2.merge([b, g, r])

    noise = np.random.randint(0, 20, gray.shape, dtype=np.uint8)
    out[:, :, 1] = cv2.add(out[:, :, 1], noise)
    return out

def make_liquidchrome(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    metallic = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    wave = 4 * np.sin(2 * np.pi * (map_y / 28.0))
    map_x_wave = (map_x + wave).astype(np.float32)
    map_y_f = map_y.astype(np.float32)
    rippled = cv2.remap(metallic, map_x_wave, map_y_f,
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    glow = cv2.GaussianBlur(rippled, (0, 0), 15)
    result = cv2.addWeighted(rippled, 0.8, glow, 0.35, 0)
    return result

def make_filmgrain(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    gray = np.clip(gray * 1.3 - 20, 0, 255)

    noise = np.random.normal(0, 25, (h, w)).astype(np.float32)
    grainy = np.clip(gray + noise, 0, 255)

    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = np.clip(1 - (dist / max_dist) * 0.5, 0, 1)
    grainy = (grainy * vignette).astype(np.uint8)

    return cv2.cvtColor(grainy, cv2.COLOR_GRAY2BGR)

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
    "SPIDERVERSE": make_spiderverse,
    "EMBOSS": make_emboss,
    "HOLOGRAM": make_hologram,
    "DRAWN": make_drawn,
    "DOTSCAN": make_dotscan,
    "RISO": make_riso,
    "POPART": make_popart,
    "ACID": make_acid,
    "LIQUIDCHROME": make_liquidchrome,
    "FILMGRAIN": make_filmgrain,
}

MODES = list(LIVE_FILTERS.keys())

DISPLAY_MODES = ["SINGLE", "TRIPLE"]

TRIPLE_FINGER_PAIRS = [
    (THUMB_TIP, INDEX_TIP),   # kotak 1: jempol - telunjuk
    (INDEX_TIP, MIDDLE_TIP),  # kotak 2: telunjuk - jari tengah
    (MIDDLE_TIP, RING_TIP),   # kotak 3: jari tengah - jari manis
]

def render_window(frame, filter_fn, quad_pts):
    h, w = frame.shape[:2]
    pts = np.int32(quad_pts)

    bx, by, bw, bh = cv2.boundingRect(pts)
    x1 = max(bx, 0)
    y1 = max(by, 0)
    x2 = min(bx + bw, w)
    y2 = min(by + bh, h)
    bw, bh = x2 - x1, y2 - y1

    if bw <= 0 or bh <= 0:
        return frame

    crop = frame[y1:y2, x1:x2]
    filtered_crop = filter_fn(crop)

    if filtered_crop.shape[:2] != (bh, bw):
        filtered_crop = cv2.resize(filtered_crop, (bw, bh))

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

def get_quad_from_hands(hands_pts, tip_a=THUMB_TIP, tip_b=INDEX_TIP):
    if len(hands_pts) != 2:
        return None

    def hand_x(hand):
        return (hand[tip_a][0] + hand[tip_b][0]) / 2

    hands_sorted = sorted(hands_pts, key=hand_x)
    left_hand, right_hand = hands_sorted[0], hands_sorted[1]

    def top_bottom(hand):
        a, b = hand[tip_a], hand[tip_b]
        return (a, b) if a[1] < b[1] else (b, a)

    left_top, left_bottom = top_bottom(left_hand)
    right_top, right_bottom = top_bottom(right_hand)
    return (left_top, right_top, right_bottom, left_bottom)

def pinch_distance(hand_pts):
    x1, y1 = hand_pts[THUMB_TIP]
    x2, y2 = hand_pts[INDEX_TIP]
    return math.hypot(x2 - x1, y2 - y1)

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
    print("Mode filter yang aktif:", MODES)
    print("(pencet angka 1-9 buat lompat langsung ke mode filter 1-9)")
    print("Mode tampilan:", DISPLAY_MODES, "-> tekan 'm' buat ganti SINGLE <-> TRIPLE")
    print("Tekan 'f' buat toggle fullscreen, window juga bisa di-resize manual (drag ujungnya)")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    WIN_NAME = "Press Q to stop - Press M to toggle display mode"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL) 
    is_fullscreen = False

    switcher = GestureSwitcher(MODES)
    display_mode_idx = 0  # 0 = SINGLE, 1 = TRIPLE
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        small_frame = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE,
                                  interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((now - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_pts = []
        pinch_values = []
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                pts = get_hand_points(landmarks, w, h)
                hands_pts.append(pts)
                pd = pinch_distance(pts)
                pinch_values.append(pd)

        min_pinch = min(pinch_values) if pinch_values else None
        switcher.update(min_pinch, now)

        current_display_mode = DISPLAY_MODES[display_mode_idx]

        if current_display_mode == "SINGLE":
            quad = get_quad_from_hands(hands_pts, THUMB_TIP, INDEX_TIP)
            if quad:
                frame = render_window(frame, LIVE_FILTERS[switcher.mode], quad)
        else:  # TRIPLE
            for i, (tip_a, tip_b) in enumerate(TRIPLE_FINGER_PAIRS):
                quad_i = get_quad_from_hands(hands_pts, tip_a, tip_b)
                if quad_i:
                    filter_name = MODES[(switcher.mode_idx + i) % len(MODES)]
                    frame = render_window(frame, LIVE_FILTERS[filter_name], quad_i)

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
            if win_w > 0 and win_h > 0 and (win_w, win_h) != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, (win_w, win_h))
        except cv2.error:
            pass  

        cv2.imshow(WIN_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN_NAME, w, h)
        elif key == ord("m"):
            display_mode_idx = (display_mode_idx + 1) % len(DISPLAY_MODES)
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(MODES):
                switcher.mode_idx = idx

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()