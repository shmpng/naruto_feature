"""
main.py
Entry point — just run:  python main.py
"""

import traceback

try:
    import mediapipe as mp
    print(f"✅ MediaPipe v{mp.__version__}")
except ImportError:
    print("❌  MediaPipe not found.  Run:  pip install -r requirements.txt")
    raise SystemExit(1)

from app import SageEyeFilterApp


if __name__ == "__main__":
    try:
        SageEyeFilterApp().run()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\n❌ {e}")
        traceback.print_exc()
