## CarParkProject - Explanation and Step-by-Step Usage

This project contains tools to pick parking spaces on a static image and then run an automated parking-space occupancy detector on a video. The main scripts are:

- `ParkingSpacePicker.py` - interactive tool to define parking slot top-left corners on `carParkImg.png`.
- `main.py` - runs the parking-space detection using a video feed (`carPark.mp4`) and positions stored in `CarParkPos`.
- `m.py` - an alternate/experimental detector that reads `polygons` and uses trackbars to tune adaptive threshold parameters.
- `carPark.mp4` - sample video used by `main.py` and `m.py`.
- `carParkImg.png` - static image used by `ParkingSpacePicker.py`.

This file explains how each script works and gives a minute-by-minute set of steps to run them on Windows (PowerShell). It also lists common issues and fixes.

---

## Requirements

- Python 3.8+ (3.10 recommended)
- OpenCV (cv2)
- cvzone
- numpy

Create a virtual environment and install dependencies. Example (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

The included `requirements.txt` lists the minimal packages used by the scripts.

---

## Quick file descriptions and notes

- `ParkingSpacePicker.py`:
  - Opens `carParkImg.png` and allows you to click left mouse to register a new parking position (top-left corner), and right mouse to remove a saved position if the click is within an existing rectangle.
  - It saves the list of positions to a pickle file named `CarParkPos` in the working directory.
  - The code uses `width, height = 107, 48` as the rectangle size for parking spaces. Adjust if your image or spaces differ.

- `main.py`:
  - Loads `CarParkPos` (pickle) and opens `carPark.mp4` for frame-by-frame analysis.
  - For each frame it converts to gray, blurs, applies adaptive thresholding, median blur, dilation, and then checks each saved parking rectangle for non-zero pixels.
  - If number of non-zero pixels (count) in a rectangle is below a hard-coded threshold (900), the spot is considered free.
  - It draws rectangles and counts on the video and shows total free spots via `cvzone.putTextRect`.

- `m.py`:
  - Similar to `main.py` but uses a different pickle file name (`polygons`) and has interactive trackbars to tune adaptive threshold parameters live.
  - Rectangle size and thresholds are slightly different here (width=103, height=43).

---

## Minute-by-minute usage (detailed)

These steps assume your PowerShell current directory is `d:\downloads\CarParkProject` and that the files `carPark.mp4` and `carParkImg.png` exist in that directory.

Minute 0 - 2: Prepare environment

1. Open PowerShell and change to the project folder:

   ```powershell
   cd d:\downloads\CarParkProject
   ```

2. (Optional but recommended) Create and activate a virtual environment, then install requirements:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   If you can't activate scripts due to ExecutionPolicy, run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force` and re-run activation.

Minute 2 - 6: Define parking spaces with `ParkingSpacePicker.py`

1. Run the picker to create or edit parking positions:

   ```powershell
   python .\ParkingSpacePicker.py
   ```

2. A window titled "Image" will open showing `carParkImg.png`.
   - Left-click where you want the top-left corner of each parking rectangle to be. Each click appends a `(x, y)` coordinate to `CarParkPos` (a pickle file).
   - Right-click inside an existing rectangle to remove it.
   - Every change is saved instantly to the `CarParkPos` pickle file.

3. Close the window (Alt+F4 or the window close control) once you've added all spaces.

Minute 6 - 8: Inspect/verify saved positions (optional)

1. You can check the saved positions by running a short Python snippet:

   ```powershell
   python - <<'PY'
   import pickle
   with open('CarParkPos','rb') as f:
       pos = pickle.load(f)
   print(pos)
   PY
   ```

Minute 8 - 12: Run the detector in `main.py`

1. Run the main detector:

   ```powershell
   python .\main.py
   ```

2. A window named "Image" will show the video frames with parking rectangles drawn. Green rectangles indicate free spots; red ones indicate occupied.

3. The top-left shows a summary like "Free: X/Y". This updates per frame.

4. To stop the program, close the window or press Ctrl+C in the PowerShell terminal to interrupt.

Minute 12 - 16: Tune detection with `m.py` (optional)

1. Run the alternate script to tune threshold parameters interactively:

   ```powershell
   python .\m.py
   ```

2. This opens a window named "Vals" with three trackbars: `Val1`, `Val2`, `Val3`.
   - `Val1` controls the blockSize for adaptive threshold (must be odd). Typical values: 11-51.
   - `Val2` controls C (constant subtracted). Typical values: 1-30.
   - `Val3` controls the median blur kernel size (must be odd).

3. Tweak the trackbars until detection masks (if you un-comment `imshow` lines) look clean.

4. Note: `m.py` loads positions from `polygons` (not `CarParkPos`). If you want to reuse `CarParkPos`, copy or rename the file:

   ```powershell
   copy .\CarParkPos .\polygons
   ```

---

## Common issues and troubleshooting

- Issue: "FileNotFoundError" when loading pickle files
  - `main.py` expects `CarParkPos` to exist. Create it by running `ParkingSpacePicker.py` first.
  - `m.py` expects `polygons`. Either run a separate picker that saves `polygons` or copy `CarParkPos` to `polygons`.

- Issue: OpenCV windows appear blank or script errors when reading video
  - Ensure `carPark.mp4` is present and readable.
  - Check OpenCV build support for the video codec used in the file. If issues persist, try converting the video to MP4 with a common codec (H.264) using ffmpeg.

- Issue: ExecutionPolicy prevents running activation script
  - Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force` in an elevated or user PowerShell.

- Note about thresholds and rectangle size
  - The scripts have slightly different rectangle sizes and thresholds (`main.py` uses 107x48 and threshold count 900; `m.py` uses 103x43). If you are using the same positions across scripts, align widths/heights in the code.

---

## Suggested small improvements (manual edits)

- Make `ParkingSpacePicker.py` and `main.py` use the same pickle filename and the same rectangle width/height variables to avoid confusion.
- Parameterize thresholds and rectangle sizes via command-line arguments or a small config file.

---

## requirements.txt

See `requirements.txt` in the project root. Use `pip install -r requirements.txt` as shown earlier.

---

If you'd like, I can:
- Update the scripts so they share the same filename and rectangle sizes, or
- Add a small CLI to `main.py` to accept the pickle filename and rectangle size.

Tell me which one you prefer and I'll implement it.

---

## Algorithm & Implementation Details

This section explains how the detection code works internally (both the original rectangle-based pipeline and the updated polygon-based `m.py`). It describes data shapes, the preprocessing steps, the masking/counting approach, thresholding heuristics, common edge cases, and suggestions for improving robustness.

### High-level pipeline

- Input: video frames read by OpenCV (BGR images). Shapes: (H, W, 3).
- Convert to grayscale -> blur -> adaptive threshold -> median blur -> dilation to produce a binary mask image (H, W) where foreground (cars) appears white (255) when using THRESH_BINARY_INV.
- For each parking region (rectangle or polygon): create a binary mask of the same size (H, W) with the region filled.
- Apply the mask to the thresholded image and count non-zero pixels inside the region.
- Compare the count against a threshold to decide occupied vs free. Draw overlays and output a per-frame summary.

### Preprocessing details (what each step does)

- Grayscale (cv2.cvtColor): reduces color channels to a single luminance channel. Output shape: (H, W).
- GaussianBlur (cv2.GaussianBlur): reduces sensor noise; kernel (3,3) small smoothing.
- Adaptive Thresholding (cv2.adaptiveThreshold with ADAPTIVE_THRESH_GAUSSIAN_C): computes threshold per local block to handle uneven lighting. Block size and C (subtracted constant) are exposed in `m.py` trackbars as Val1 and Val2.
- MedianBlur: removes small speckle noise while preserving edges; kernel size exposed as Val3.
- Dilation: expands white regions to fill small gaps, which helps count larger connected components instead of tiny speckles.

Data after preprocessing: `imgThres` is a single-channel uint8 image where target foreground pixels are 255 and background 0.

### Polygon masking and counting (how `m.py` works now)

- `polygons` is a list of polygons. Each polygon is a list of 4 (x, y) tuples: [[(x1,y1), (x2,y2), (x3,y3), (x4,y4)], ...]
- For each polygon we build a mask array `mask = np.zeros((H, W), dtype=np.uint8)` and call `cv2.fillPoly(mask, [pts], 255)` where `pts` is an int32 numpy array of polygon points.
- We compute `imgCrop = cv2.bitwise_and(imgThres, imgThres, mask=mask)` which keeps only thresholded pixels inside the polygon.
- We compute `count = cv2.countNonZero(imgCrop)`. This is the scalar used to detect presence of a vehicle.

### Thresholding heuristic

- The original rectangle approach used a hard-coded threshold (900). That was tuned to a particular rectangle size (approx 103x43).
- For polygons of different sizes, `m.py` now computes the polygon area (in pixels) with `cv2.contourArea(pts)` and derives `adaptive_thresh = max(300, int(area * 0.18))` by default. This scales the threshold roughly proportionally to polygon area.
- You can adjust the multiplier (0.18) or the minimum (300) to be more/less sensitive. I can expose this multiplier as a trackbar if you'd like live tuning.

Rationale: larger parking areas naturally have more background pixels; scaling avoids falsely classifying large empty spots as occupied just because area is large. The minimum prevents tiny polygons from having thresholds that are too small.

### Data shapes and types

- Input frame: uint8 BGR image (H, W, 3)
- After grayscale/blur: uint8 (H, W)
- After adaptiveThreshold: uint8 (H, W) with values 0 or 255
- Polygon mask: uint8 (H, W) with values 0 or 255
- Masked crop: uint8 (H, W) with values 0 or 255; counts computed with `cv2.countNonZero` returning an integer scalar

### Edge cases and limitations

- Lighting changes and shadows: adaptive threshold helps, but strong shadows or reflections may produce false positives. Consider using color information (S channel), background subtraction, or temporal smoothing.
- Perspective foreshortening: polygons mitigate this by matching the actual parking shape, but if the camera angle is extreme the top and bottom of the space may show very different pixel densities. Perspective-corrected ROI (homography) can help.
- Partial occlusion: a car partially in the spot may not produce enough non-zero pixels to cross the threshold. Using morphological features (connected component size) or temporal filtering (require occupancy across N frames) helps.
- Noise and small artifacts: trackbar-tuned median/dilate steps reduce false speckles, but you may still need to fine-tune for different cameras.

### Performance and complexity

- Per-frame work is linear in the number of ROI pixels across all polygons. For P polygons and average area A (in pixels), mask creation and bitwise_and are O(P * A). This is fast for typical parking lot sizes (tens of polygons, thousands of pixels each) on modern machines.
- Using compiled OpenCV operations keeps per-frame time small; avoid Python loops over pixel data.

### Tuning tips

- Use `m.py`'s trackbars:
   - Increase Val1 (blockSize) to make thresholding consider larger neighborhoods; this smooths illumination differences but loses detail. Keep it odd.
   - Increase Val2 (C) to make the threshold stricter (subtracting more), which can make small dark items appear as background.
   - Increase Val3 (median kernel) to remove more speckle noise, but don't over-smooth.
- Check the `count` printed per polygon (displayed on the video) to calibrate the adaptive multiplier (0.18) or the hard-coded min threshold. A good practice: capture several frames under different lighting and compute the distribution of counts for empty vs occupied spots.

### Suggested improvements (next steps)

- Perspective correction: compute a homography to a bird's-eye view and run detection on normalized rectangular ROIs — this simplifies thresholds and makes counts comparable across spots.
- Temporal filtering: require occupancy to be consistent across N consecutive frames before toggling state.
- Background modeling: using MOG2 or KNN background subtractor to get robust moving-object masks.
- Machine learning approach: train a small classifier (SVM or lightweight CNN) on ROI patches to detect cars; more robust but requires annotated data.

---

If you want, I can now:
- Update `EXPLANATION.md` with example numbers from actual frames (I can sample a few frames if you allow me to run the scripts),
- Add a trackbar to `m.py` to tune the area-to-threshold multiplier live, or
- Update `main.py` to use polygons as well so it's consistent across tools.

Tell me which of those you'd like next and I'll implement it.
