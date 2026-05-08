"""
Run this ONCE to generate eye_overlay.png and eyeshadow.png
from the base64-embedded images below.

Just run:   python save_images.py
"""

import base64, os

# ── Image 1: yellow minus circle → eye_overlay.png ──────────
EYE_OVERLAY_B64 = """\
YOUR_BASE64_HERE
"""

# ── Image 2: orange eyeshadow → eyeshadow.png ───────────────
EYESHADOW_B64 = """\
YOUR_BASE64_HERE
"""

def save(b64, filename):
    if os.path.exists(filename):
        print(f"✅ {filename} already exists — skipping")
        return
    try:
        data = base64.b64decode(b64.strip())
        with open(filename, "wb") as f:
            f.write(data)
        print(f"✅ Saved {filename} ({len(data)//1024} KB)")
    except Exception as e:
        print(f"❌ Failed to save {filename}: {e}")

save(EYE_OVERLAY_B64, "eye_overlay.png")
save(EYESHADOW_B64,   "eyeshadow.png")
print("\nDone. Now run:  python eye_filter.py")
