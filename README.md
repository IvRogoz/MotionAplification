# MotionAplification
Motion magnification tools for offline videos and live camera streams.

Reference article:
https://warped3.substack.com/p/motion-magnification

## Features
- Eulerian magnification for video files.
- Live camera magnification with real-time preview.
- Optional recording of processed live output.
- Camera device dropdown with refresh.
- Camera mode probing (resolution/FPS) and one-click apply.
- Capture/save resolution controls.
- Desktop UI (Tkinter), no web server required.

## Requirements
- Python 3.10+ recommended.
- Windows/macOS/Linux with OpenCV-compatible camera/video codecs.

## Installation
```bash
pip install -r requirements.txt
```

## Run Desktop UI
```bash
python ui_app.py
```

## Live Camera Workflow
1. Open the `Live Camera` tab.
2. Choose a camera from `Camera Device`.
3. Click `Probe Modes` to detect common supported modes.
4. Pick a mode and click `Use Selected Mode` (optional but recommended).
5. Enable recording and choose `Save As` path if you want output saved.
6. Click `Start Live Camera`.
7. Press `q` in the OpenCV window to stop.

## Video File Workflow
1. Open the `Video Processing` tab.
2. Select input video.
3. Select output video path.
4. Set magnification/filter parameters.
5. Click `Process Video`.

## Main Scripts
- `ui_app.py`: desktop UI.
- `ampEul.py`: file-based Eulerian magnification.
- `ampLive.py`: live magnification pipeline.
- `ampPhase.py`, `ampPhaseSingleChannel.py`: phase-based variants.

## Notes
- `chunk_size` must be greater than `overlap`.
- `high_cutoff` must be greater than `low_cutoff`.
- If a requested capture mode is unsupported by camera driver, OpenCV may silently fall back to a nearby/default mode.

