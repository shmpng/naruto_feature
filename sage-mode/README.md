# 👁️ Naruto Sage Mode Filter

Real-time eye filter using MediaPipe Face Landmarker.  
Combines an **orange eyeshadow** (Sage Mode) with a **yellow iris sticker** — both toggled on/off by closing both eyes for 3 seconds.

---

## 📁 Project Structure

```
naruto_filter/
├── main.py               # Entry point
├── app.py                # SageEyeFilterApp — camera loop & toggle logic
├── iris_filter.py        # Yellow iris sticker (clips to eyelid shape)
├── eyeshadow_filter.py   # Orange Sage Mode eyeshadow + lash lines
├── landmarks.py          # All MediaPipe landmark indices & socket shapes
├── utils.py              # Shared helpers (EAR, image utils, model download)
├── requirements.txt
│
├── eye_overlay.jpg       # Yellow iris sticker  (white background)
├── realorange.jpg        # Orange eyeshadow texture  (black background)
└── face_landmarker.task  # Auto-downloaded on first run
```

---

## 🚀 Setup

```bash
pip install -r requirements.txt
python main.py
```

The MediaPipe model (`face_landmarker.task`) downloads automatically on first run (~30 MB).

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| Close both eyes for **3 s** | Toggle filters **ON / OFF** |
| `S` | Reset filters off |
| `D` | Debug mode — show iris sticker without activating |
| `Q` | Quit |

---

## 🖼️ Assets

| File | Description |
|------|-------------|
| `eye_overlay.jpg` | Yellow circle iris sticker on a **white** background |
| `realorange.jpg` | Orange eyeshadow texture on a **black** background |

White background is stripped automatically from the iris sticker.  
Black background is stripped automatically from the eyeshadow texture.

---

## ⚙️ How It Works

1. **EAR (Eye Aspect Ratio)** is computed each frame for both eyes.
2. When both EARs drop below the closed threshold for **3 continuous seconds**, the filters toggle.
3. A teal countdown ring in the top-left shows progress while eyes are closed.
4. When active, the **eyeshadow** is painted first (into the socket above the lid), then the **iris sticker** is blended on top — both clipped to the real eyelid contour so they open and close naturally with the eye.
