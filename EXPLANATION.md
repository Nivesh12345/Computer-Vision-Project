# Parking Space Detection System

## Overview

This system uses **computer vision** and **image processing** techniques to detect parking space occupancy from video frames. The approach is based on **pixel intensity analysis** within predefined regions of interest (ROIs).

---

## Complete Pipeline

```txt
┌─────────────────────────────────────────────────────────────────┐
│                    PARKING DETECTION PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

Step 1: SETUP PHASE (One-time)
┌──────────────────────┐
│  carParkImg.png      │  Reference image
│  (Static frame)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ PolygonSpacePicker   │  Manual polygon definition
│ - Click 4 corners    │  User defines parking spaces
│ - Save coordinates   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  polygons (pickle)   │  Saved parking space coordinates
│  [[(x1,y1),(x2,y2),  │  List of 4-point polygons
│    (x3,y3),(x4,y4)]] │
└──────────────────────┘


Step 2: DETECTION PHASE (Real-time)
┌──────────────────────┐
│   carPark.mp4        │  Input video stream
│   (Video frames)     │  BGR color, 1920x1080 (example)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                      │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
    [COLOR CONVERSION]
           │
           ▼
    [GAUSSIAN BLUR]
           │
           ▼
    [ADAPTIVE THRESHOLD]
           │
           ▼
    [MEDIAN BLUR]
           │
           ▼
    [MORPHOLOGICAL DILATION]
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                   DETECTION & ANALYSIS                        │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
    [POLYGON MASKING]
           │
           ▼
    [PIXEL COUNTING]
           │
           ▼
    [THRESHOLD COMPARISON]
           │
           ▼
    [TEMPORAL FILTERING]
           │
           ▼
┌──────────────────────┐
│  Occupancy Result    │  Free/Occupied status
│  + Confidence Score  │  Per parking space
└──────────────────────┘
```

---

## Phase 1: Polygon Definition (Setup)

### 1.1 Manual ROI Selection

**Tool**: `PolygonSpacePicker.py`

**Process**:
```
User clicks 4 points on static image:

    P1 ●─────────● P2
       │         │
       │  Space  │
       │         │
    P4 ●─────────● P3

Coordinates stored as: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
```

**Data Structure**:
```python
polygons = [
    [(100, 200), (200, 200), (200, 250), (100, 250)],  # Space 1
    [(210, 200), (310, 200), (310, 250), (210, 250)],  # Space 2
    # ... more spaces
]
```

**Storage**: Serialized using Python's `pickle` module
- File: `polygons`
- Format: Binary pickle file
- Contains: List of polygon coordinate lists

---

## Phase 2: Image Preprocessing

### 2.1 Color Space Conversion (BGR → Grayscale)

**Function**: `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`

**Purpose**: Reduce 3-channel color to 1-channel intensity

```
INPUT (BGR):                    OUTPUT (Grayscale):
┌─────────────┐                ┌─────────────┐
│ B: 120      │                │             │
│ G: 130  →   │   Transform    │  I: 127     │
│ R: 140      │   ─────────→   │             │
└─────────────┘                └─────────────┘

Formula: Gray = 0.299×R + 0.587×G + 0.114×B
```

**Why**: 
- Simpler computation (1 channel vs 3)
- Parking occupancy doesn't depend on color
- Reduces noise from color variations

**Data Shape**: `(H, W, 3)` → `(H, W)`

---

### 2.2 Gaussian Blur (Noise Reduction)

**Function**: `cv2.GaussianBlur(gray, (3, 3), 1)`

**Parameters**:
- Kernel size: `(3, 3)` - 3×3 pixel window
- Sigma: `1` - Standard deviation of Gaussian distribution

**Purpose**: Smooth image to reduce sensor noise

```
Gaussian Kernel (3×3):
┌─────────────────┐
│ 0.075  0.124  0.075 │
│ 0.124  0.204  0.124 │
│ 0.075  0.124  0.075 │
└─────────────────┘

Each pixel = weighted average of neighbors
```

**Effect**:
```
BEFORE:                    AFTER:
┌───────────┐             ┌───────────┐
│ 100 255 98│             │ 120 140 115│
│ 102  95 103│    →       │ 115 118 120│
│  98 100 255│             │ 110 125 130│
└───────────┘             └───────────┘
(Noisy)                   (Smoothed)
```

**Why**:
- Removes random pixel variations
- Prevents false detections from noise
- Improves threshold stability

---

### 2.3 Adaptive Thresholding (Binarization)

**Function**: `cv2.adaptiveThreshold(blur, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 16)`

**Parameters**:
- `maxValue`: `255` - White pixel value
- `adaptiveMethod`: `ADAPTIVE_THRESH_GAUSSIAN_C` - Gaussian-weighted mean
- `thresholdType`: `THRESH_BINARY_INV` - Inverted binary
- `blockSize`: `25` - Neighborhood size (25×25 pixels)
- `C`: `16` - Constant subtracted from mean

**How It Works**:
```
For each pixel at (x, y):
1. Calculate mean of 25×25 neighborhood using Gaussian weights
2. Threshold = Mean - C (16)
3. If pixel_value < threshold: output = 255 (white)
   Else: output = 0 (black)
```

**Why Adaptive (not global)**:
```
Global Threshold Problem:
┌──────────────────────────┐
│ Bright area │ Dark area  │
│  (sunlight) │  (shadow)  │
│             │            │
│ Same threshold fails!    │
└──────────────────────────┘

Adaptive Solution:
┌──────────────────────────┐
│ Threshold=150│Threshold=80│
│  (bright)    │  (dark)    │
│ Local adapt! │            │
└──────────────────────────┘
```

**THRESH_BINARY_INV Effect**:
```
Normal Binary:              Inverted Binary:
Dark car → 0 (black)       Dark car → 255 (white)
Bright ground → 255        Bright ground → 0

We want cars as WHITE (foreground)
So we use INVERTED
```

**Visual Example**:
```
INPUT (Grayscale):          OUTPUT (Binary):
┌─────────────┐            ┌─────────────┐
│ 180 185 190 │            │ 0   0   0   │ ← Bright ground
│ 175  80  85 │     →      │ 0  255 255  │ ← Dark car
│ 170  75  80 │            │ 0  255 255  │
└─────────────┘            └─────────────┘
```

---

### 2.4 Median Blur (Speckle Removal)

**Function**: `cv2.medianBlur(thresh, 5)`

**Parameter**: Kernel size = `5` (5×5 window)

**How It Works**:
```
For each pixel:
1. Take 5×5 neighborhood (25 pixels)
2. Sort pixel values
3. Replace center with MEDIAN value
```

**Example**:
```
5×5 Window values (sorted):
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
                                    ↑
                                 Median = 0

Removes isolated white pixels (salt noise)
```

**Effect**:
```
BEFORE:                    AFTER:
┌─────────────┐           ┌─────────────┐
│ 0   0  255  │           │ 0   0   0   │
│ 0  255  0   │    →      │ 0   0   0   │
│ 255  0   0  │           │ 0   0   0   │
└─────────────┘           └─────────────┘
(Noisy)                   (Clean)
```

**Why Median (not Mean)**:
- Preserves edges better
- Removes outliers effectively
- Non-linear filter (better for binary images)

---

### 2.5 Morphological Dilation (Gap Filling)

**Function**: `cv2.dilate(median, kernel, iterations=1)`

**Kernel**: `np.ones((3, 3), np.uint8)` - 3×3 square structuring element

**How It Works**:
```
Structuring Element:
┌───────┐
│ 1 1 1 │
│ 1 1 1 │
│ 1 1 1 │
└───────┘

If ANY pixel in 3×3 neighborhood is white (255),
make center pixel white
```

**Effect**:
```
BEFORE:                    AFTER:
┌─────────────┐           ┌─────────────┐
│ 255 0  255  │           │ 255 255 255 │
│  0  0   0   │    →      │ 255 255 255 │
│ 255 0  255  │           │ 255 255 255 │
└─────────────┘           └─────────────┘
(Gaps)                    (Filled)
```

**Purpose**:
- Connect broken car regions
- Fill small holes
- Make car blobs more solid

**Visual Comparison**:
```
Original Car Detection:    After Dilation:
┌──────────────┐          ┌──────────────┐
│ ████  ████   │          │ ██████████   │
│ ████  ████   │    →     │ ██████████   │
│ ████  ████   │          │ ██████████   │
└──────────────┘          └──────────────┘
(Fragmented)              (Solid blob)
```

---

## Phase 3: Polygon Masking & Detection

### 3.1 Create Binary Mask for Each Polygon

**Function**: `cv2.fillPoly(mask, [pts], 255)`

**Process**:
```python
# Create blank mask (all zeros)
mask = np.zeros((height, width), dtype=np.uint8)

# Fill polygon region with white (255)
cv2.fillPoly(mask, [polygon_points], 255)
```

**Visual**:
```
Original Frame:            Polygon Mask:
┌────────────────┐        ┌────────────────┐
│                │        │   0  0  0  0   │
│   ┌──────┐    │        │   0┌──────┐0   │
│   │ Car  │    │   →    │   0│ 255  │0   │
│   └──────┘    │        │   0└──────┘0   │
│                │        │   0  0  0  0   │
└────────────────┘        └────────────────┘
```

**Why**:
- Isolates specific parking space
- Ignores pixels outside polygon
- Allows per-space analysis

---

### 3.2 Apply Mask (Bitwise AND)

**Function**: `cv2.bitwise_and(processed_frame, processed_frame, mask=mask)`

**Operation**:
```
Processed Frame:           Mask:                Result:
┌────────────┐            ┌────────────┐       ┌────────────┐
│ 255 255  0 │            │  0   0   0 │       │  0   0   0 │
│ 255 255  0 │     AND    │ 255 255  0 │   =   │ 255 255  0 │
│  0   0   0 │            │ 255 255  0 │       │  0   0   0 │
└────────────┘            └────────────┘       └────────────┘

Bitwise AND: 
255 AND 255 = 255
255 AND 0   = 0
0   AND 255 = 0
0   AND 0   = 0
```

**Result**: Only pixels inside polygon are preserved

---

### 3.3 Count Non-Zero Pixels

**Function**: `cv2.countNonZero(cropped)`

**What It Does**:
```
Masked Region:
┌────────────┐
│  0   0   0 │
│ 255 255  0 │  → Count = 8 (number of 255 pixels)
│ 255 255  0 │
│ 255 255  0 │
└────────────┘
```

**Interpretation**:
- **High count** = Many white pixels = Car present (dark object)
- **Low count** = Few white pixels = Empty space (bright ground)

---

### 3.4 Adaptive Threshold Calculation

**Formula**:
```python
area = cv2.contourArea(polygon_points)
threshold = max(300, int(area * 0.18))
```

**Why Adaptive**:
```
Small Parking Space:        Large Parking Space:
Area = 2000 pixels          Area = 5000 pixels
Threshold = 360             Threshold = 900

Larger spaces naturally have more pixels,
so threshold scales proportionally
```

**Threshold Multiplier (0.18)**:
- Empirically determined
- Means: ~18% of space area should be "car pixels"
- Adjustable based on camera angle, lighting

**Minimum Threshold (300)**:
- Prevents tiny thresholds for small spaces
- Ensures minimum sensitivity

---

### 3.5 Occupancy Decision

**Logic**:
```python
if pixel_count >= threshold:
    status = "OCCUPIED"
    confidence = min(1.0, pixel_count / threshold)
else:
    status = "FREE"
    confidence = min(1.0, pixel_count / threshold)
```

**Confidence Score**:
```
pixel_count = 450, threshold = 300
confidence = 450 / 300 = 1.5 → capped at 1.0

pixel_count = 150, threshold = 300
confidence = 150 / 300 = 0.5

Higher confidence = stronger detection
```

**Decision Boundary**:
```
Pixel Count vs Threshold:

  1200 │                    ████ OCCUPIED
       │                 ███
  900  │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ← Threshold
       │              ░░░
  600  │           ░░░
       │        ░░░
  300  │     ░░░  FREE
       │  ░░░
    0  └─────────────────────────────
       Frame 1  2  3  4  5  6  7  8
```

---

## Phase 4: Temporal Filtering

### 4.1 Frame History Buffer

**Data Structure**:
```python
from collections import deque

# For each parking space, maintain last 5 states
space_history = [
    deque([True, True, False, True, True], maxlen=5),   # Space 1
    deque([False, False, False, False, False], maxlen=5), # Space 2
    # ... one deque per space
]
```

**Buffer Behavior**:
```
Initial: []
Frame 1: [True]
Frame 2: [True, False]
Frame 3: [True, False, True]
Frame 4: [True, False, True, True]
Frame 5: [True, False, True, True, False]
Frame 6: [False, True, True, False, True]  ← Oldest dropped
         └─ New value added
```

---

### 4.2 Majority Vote Algorithm

**Function**:
```python
def apply_temporal_filter(space_index, is_occupied):
    history = space_history[space_index]
    history.append(is_occupied)
    
    if len(history) < 3:
        return is_occupied  # Not enough data
    
    occupied_count = sum(history)  # Count True values
    return occupied_count > len(history) / 2
```

**Example**:
```
History: [True, False, True, True, False]
         └─ 3 True, 2 False

occupied_count = 3
threshold = 5 / 2 = 2.5
3 > 2.5 → Result = OCCUPIED
```

**Why This Works**:
```
Without Temporal Filtering:
Frame: 1    2    3    4    5    6    7    8
State: OCC FREE OCC FREE OCC FREE OCC FREE
       ↑ Flickering! Unstable!

With Temporal Filtering (5-frame window):
Frame: 1    2    3    4    5    6    7    8
Raw:   OCC FREE OCC FREE OCC FREE OCC FREE
Filt:  OCC  OCC  OCC  OCC  OCC  OCC  OCC  OCC
       ↑ Stable! Majority vote smooths noise
```

**Benefits**:
- Reduces false positives from shadows
- Smooths detection over time
- Requires consistent state change

---

## Complete Data Flow Example

### Input Frame Processing

```
STEP 1: Original Frame (BGR)
┌─────────────────────────────────┐
│        Parking Lot Image        │
│  ┌────┐  ┌────┐  ┌────┐        │
│  │Car │  │    │  │Car │        │
│  └────┘  └────┘  └────┘        │
│   Space1  Space2  Space3        │
└─────────────────────────────────┘
Shape: (1080, 1920, 3)
Dtype: uint8, Range: [0-255]

↓ cv2.cvtColor(BGR2GRAY)

STEP 2: Grayscale
┌─────────────────────────────────┐
│  ░░░░░░  ▓▓▓▓▓▓  ░░░░░░        │
│  ░░░░░░  ▓▓▓▓▓▓  ░░░░░░        │
└─────────────────────────────────┘
Shape: (1080, 1920)
Dtype: uint8, Range: [0-255]

↓ cv2.GaussianBlur((3,3), 1)

STEP 3: Blurred
┌─────────────────────────────────┐
│  ░░░░░░  ▓▓▓▓▓▓  ░░░░░░        │
│  ░░░░░░  ▓▓▓▓▓▓  ░░░░░░        │
└─────────────────────────────────┘
(Smoother transitions)

↓ cv2.adaptiveThreshold(...)

STEP 4: Binary (Inverted)
┌─────────────────────────────────┐
│  ██████  ░░░░░░  ██████         │
│  ██████  ░░░░░░  ██████         │
└─────────────────────────────────┘
Cars=White(255), Ground=Black(0)

↓ cv2.medianBlur(5)

STEP 5: Denoised
┌─────────────────────────────────┐
│  ██████  ░░░░░░  ██████         │
│  ██████  ░░░░░░  ██████         │
└─────────────────────────────────┘
(Speckles removed)

↓ cv2.dilate(...)

STEP 6: Dilated
┌─────────────────────────────────┐
│  ██████  ░░░░░░  ██████         │
│  ██████  ░░░░░░  ██████         │
└─────────────────────────────────┘
(Gaps filled)
```

### Per-Space Detection

```
SPACE 1 (Has Car):
┌──────────────────────────────┐
│ Polygon Mask:                │
│ ┌──────┐                     │
│ │ 255  │  (White region)     │
│ └──────┘                     │
│                              │
│ Masked Result:               │
│ ┌──────┐                     │
│ │ ████ │  (Car pixels)       │
│ └──────┘                     │
│                              │
│ Pixel Count: 850             │
│ Threshold: 360               │
│ 850 >= 360 → OCCUPIED ✓      │
│ Confidence: 1.0              │
└──────────────────────────────┘

SPACE 2 (Empty):
┌──────────────────────────────┐
│ Polygon Mask:                │
│ ┌──────┐                     │
│ │ 255  │  (White region)     │
│ └──────┘                     │
│                              │
│ Masked Result:               │
│ ┌──────┐                     │
│ │      │  (No car pixels)    │
│ └──────┘                     │
│                              │
│ Pixel Count: 120             │
│ Threshold: 360               │
│ 120 < 360 → FREE ✓           │
│ Confidence: 0.33             │
└──────────────────────────────┘
```

---

## Mathematical Formulas

### 1. Grayscale Conversion
```
I(x,y) = 0.299×R(x,y) + 0.587×G(x,y) + 0.114×B(x,y)

Where:
- I = Intensity (grayscale value)
- R, G, B = Red, Green, Blue channels
- Weights based on human perception
```

### 2. Gaussian Blur
```
G(x,y) = (1/16) × [1  2  1]   [I(x-1,y-1)  I(x,y-1)  I(x+1,y-1)]
                  [2  4  2] ⊗ [I(x-1,y)    I(x,y)    I(x+1,y)  ]
                  [1  2  1]   [I(x-1,y+1)  I(x,y+1)  I(x+1,y+1)]

Where ⊗ denotes convolution
```

### 3. Adaptive Threshold
```
T(x,y) = μ(x,y) - C

Where:
- T(x,y) = Local threshold at pixel (x,y)
- μ(x,y) = Gaussian-weighted mean of blockSize×blockSize neighborhood
- C = Constant (16 in our case)

Output:
dst(x,y) = { 255  if src(x,y) < T(x,y)  (THRESH_BINARY_INV)
           { 0    otherwise
```

### 4. Polygon Area (Shoelace Formula)
```
Area = (1/2) × |Σ(x_i × y_(i+1) - x_(i+1) × y_i)|

For polygon with vertices: (x1,y1), (x2,y2), (x3,y3), (x4,y4)
```

### 5. Adaptive Threshold Calculation
```
threshold = max(min_threshold, area × multiplier)
threshold = max(300, area × 0.18)
```

### 6. Confidence Score
```
confidence = min(1.0, pixel_count / threshold)

Range: [0.0, 1.0]
```

### 7. Temporal Filter (Majority Vote)
```
filtered_state = (Σ history_i) > (N / 2)

Where:
- N = history buffer size (5 frames)
- history_i ∈ {0, 1} (False=0, True=1)
```

---

## Performance Characteristics

### Computational Complexity

**Per Frame**:
```
1. Color Conversion:     O(H × W)
2. Gaussian Blur:        O(H × W × k²)  where k=3
3. Adaptive Threshold:   O(H × W × b²)  where b=25
4. Median Blur:          O(H × W × m²)  where m=5
5. Dilation:             O(H × W × d²)  where d=3

Per Polygon (P polygons):
6. Mask Creation:        O(A)  where A=polygon area
7. Bitwise AND:          O(A)
8. Pixel Count:          O(A)

Total: O(H × W × b²) + O(P × A)
```

**Typical Values**:
- Frame: 1920×1080 = 2,073,600 pixels
- Polygons: 20 spaces
- Average polygon area: 5,000 pixels

**Estimated Time** (CPU):
- Preprocessing: ~10ms
- Detection (20 spaces): ~5ms
- **Total: ~15ms → 66 FPS**

---

## Key Parameters & Tuning

| Parameter | Value | Effect | Tuning Guide |
|-----------|-------|--------|--------------|
| **Gaussian Kernel** | (3,3) | Noise reduction | Larger = more blur, slower |
| **Adaptive Block** | 25 | Local threshold window | Larger = smoother lighting adaptation |
| **Adaptive C** | 16 | Threshold offset | Higher = stricter (less white pixels) |
| **Median Kernel** | 5 | Speckle removal | Larger = more smoothing |
| **Dilation Iterations** | 1 | Gap filling | More = larger blobs |
| **Threshold Multiplier** | 0.18 | Occupancy sensitivity | Lower = more sensitive |
| **Min Threshold** | 300 | Minimum pixel count | Higher = less sensitive |
| **History Size** | 5 | Temporal smoothing | Larger = more stable, slower response |

---

## Advantages of This Approach

1. **Fast**: 60+ FPS on CPU (no GPU needed)
2. **Simple**: No machine learning, no training data
3. **Robust**: Adaptive thresholding handles lighting changes
4. **Flexible**: Polygon shapes fit any parking space layout
5. **Stable**: Temporal filtering reduces false positives
6. **Lightweight**: ~100MB dependencies (just OpenCV)

---


---


**Why Our Approach Wins**:
- Fixed parking spaces (don't need object detection)
- Good lighting in typical parking lots
- Speed and simplicity matter
- No training data needed

---

## Summary

This system uses **classical computer vision** techniques:

1. **Preprocessing**: Convert to grayscale, blur, threshold, denoise, dilate
2. **Masking**: Isolate each parking space with polygon masks
3. **Counting**: Count white pixels (car indicators) in each space
4. **Thresholding**: Compare count to adaptive threshold
5. **Filtering**: Use temporal majority vote for stability

**Result**: Fast, simple, effective parking occupancy detection without machine learning complexity.
