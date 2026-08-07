# Local QC Inspector

A local, offline inspection app for one automotive assembly. It verifies ten visual-presence locations in a photo:

- `Screw 1`–`Screw 8`: the silver screws around the upper metal plate, numbered clockwise from the upper-left.
- `Clip 1` and `Clip 2`: the blue clips at the bottom, numbered left-to-right.

The app has four screens: **Inspect a part**, **Training & calibration**, **Layout**, and **Operating guide**. It does not send images to a cloud service. Every profile, training photo, and model is stored in the `qc_data` folder alongside the application.

## Before you begin

You need a Windows PC, a USB/web camera (or image files), and Python 3.10 or newer. Python is free. If setup says Python is not installed, install the current 64-bit Python from [python.org/downloads](https://www.python.org/downloads/) and make sure **Add Python to PATH** is checked during installation.

For a stable factory setup, use a rigid camera mount, fixed fixture, fixed orientation, fixed focus/exposure, and diffuse lighting. This visual inspection cannot establish screw torque, correct screw type, electrical function, or anything hidden from the camera.

## First-time setup

1. Put this whole `LocalQCInspector` folder somewhere you can keep it. Do not move or delete the `qc_data` folder after training; that is where the local model lives.
2. **With Wi-Fi connected, double-click `setup_once_with_wifi.bat`.** Wait for the word `SUCCESS`. This downloads Python packages into this folder; it is the only step that needs the internet.
3. Disconnect Wi-Fi as a deliberate test. Double-click **`launch.bat`**. This launcher never runs an installer, updater, package download, or cloud inference.
4. Your browser should open automatically at `http://127.0.0.1:8501`. If it does not, copy that address from the command window into a browser on the same PC.
5. On **Create your first inspection profile**, enter a name such as `Seat blower - Station 1`.
6. Click **Known-good part images**. In the file selector, select the supplied good-part photos (for this task they are in `C:\Users\sjala\Downloads\photos`). Select as many as possible. All selected images must have all eight screws and both blue clips installed.
7. Click **Create profile and calibrate**. Training may take a minute when many photos are selected.
8. Open the **Layout** tab. Confirm that every orange box is centered on a screw head or blue clip. The default layout was made for the supplied assembly. If any box is off, change its Center X, Center Y, Width, or Height value, then click **Save layout and recalibrate**.
9. Return to **Inspect a part**, upload one of the known-good images, and click **Run 10-point inspection**. It should report PASS after sufficient good images are present. The overlay makes the ID and decision at every location visible.

## How to use it

1. Place the part in the station fixture, with the metallic plate at the top of the camera view.
2. In **Inspect a part**, choose **Connected camera** to take a photo from a browser-connected camera, or choose **Upload image** to inspect an image file.
3. Click **Run 10-point inspection**.
4. Read the prominent result:

   - **PASS** — all ten required parts match the trained good condition.
   - **FAIL** — the named item(s), such as `Screw 6` or `Clip 1`, appear missing. Do not release the part until an operator verifies it.
   - **REVIEW** — the app could not safely make a confident decision. Check the image/part, camera alignment, and training set. Treat REVIEW as a hold, not a pass.

5. The annotated image can be downloaded from the inspection page for a quality record.

## Training for a different camera, environment, or defect

Create a separate profile per part variant, camera station, or fixture. Do not mix different viewpoints in one profile unless they are deliberately representative and the overlay remains correctly aligned.

### Add good images

1. Open **Training & calibration**.
2. Under **Add fully assembled (good) parts**, choose images where all ten locations are present.
3. Click **Add good images and retrain**.

Aim for at least 20–30 good images per station. Include permitted variation in part position, brightness, and finish. Add new good images whenever the optics, fixture, or lighting changes.

### Add a missing-screw or missing-clip condition

1. Prepare real photos with a known condition. For example, take 3–5 photos in which only `Screw 1` is missing.
2. Under **Add examples with a known missing item**, choose those photos.
3. Select `Screw 1` in **Items missing in every selected image**.
4. Click **Add labelled defect images and retrain**.
5. Repeat for `Screw 2` through `Screw 8`, `Clip 1`, and `Clip 2` as required.

Only group images together when the same locations are missing in every image. If one photo has Screw 1 missing and another has Screw 4 missing, add them as two separate batches. The Training Coverage table shows where labelled missing examples are still needed.

The app can flag a strong difference from good images even before a defect image is available, but labelled real defect images make the missing-item decision substantially more reliable.

## The layout coordinate fields

The Layout page uses values between 0 and 1. `Center X=0.500` means half way across the image; `Center Y=0.500` means half way down. Width and height are the size of the orange region. After any change, always click **Save layout and recalibrate** and inspect a few known-good and known-bad examples.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| Browser does not open | Keep `launch.bat` running and open `http://127.0.0.1:8501` manually. |
| Camera is unavailable | Use a supported USB camera, select **Connected camera**, and allow browser camera permission. If the camera is used by another application, close that application. |
| A good part shows REVIEW | Add more good images from the real station; confirm the fixture/camera orientation and orange boxes in Layout. |
| A screw/clip ID is wrong | In Layout, reposition that ID's orange rectangle over its actual center, save, and retrain. |
| A missing part is not correctly named | Add 3–5 real labelled photos for that exact missing item in Training & calibration. |
| The app fails after an update | Stop the command window, replace only application code if needed, but preserve `qc_data`; then launch again. |

## Technical notes

- The application registers an input photo to a saved reference using local OpenCV feature matching, then checks each fixed inspection region.
- It trains a compact local similarity model per ID from good and labelled missing examples. There is no external AI account, API key, or network inference.
- `setup_once_with_wifi.bat` is the one-time package installation. `launch.bat` intentionally never accesses a package server and starts the local application at `127.0.0.1`; inspection and retraining then work offline.
- To stop the application, close the browser and the command window, or press `Ctrl+C` in the command window.

## Files

- `setup_once_with_wifi.bat` — run once while connected to install local dependencies.
- `launch.bat` — the offline-only launcher; use for normal operation after setup.
- `app.py` — Streamlit graphical interface.
- `qc_engine.py` — local image registration, training, and inspection engine.
- `requirements.txt` — exact Python packages required.
- `qc_data/` — created at runtime; contains profiles and should be backed up as quality data.
