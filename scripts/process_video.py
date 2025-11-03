
#!/usr/bin/env python3
"""Process a single video with the BeeMonitor pipeline.

For now this script dispatches into your original implementation under
beemonitor.legacy.BeeMonitor. We'll refactor the pipeline modules next.
"""
import argparse, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--config", default="beemonitor/config/pipeline.default.yaml", help="YAML config")
    args = parser.parse_args()

    try:
        # Try to call a main-like entrypoint if present
        from beemonitor.legacy import BeeMonitor as legacy
        if hasattr(legacy, "main"):
            return legacy.main(args)  # type: ignore
        else:
            print("⚠️ beemonitor.legacy.BeeMonitor has no `main(args)` entrypoint yet.")
            print("   Next step: we will wire your functions into modular interfaces.")
            return 0
    except Exception as e:
        print(f"Error importing legacy pipeline: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
