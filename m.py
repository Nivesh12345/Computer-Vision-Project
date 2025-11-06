import cv2
import pickle
import cvzone
import numpy as np

cap = cv2.VideoCapture('carPark.mp4')
# width, height = 103, 43  # no longer used for polygon masks
with open('polygons', 'rb') as f:
    posList = pickle.load(f)  # list of polygons: [[(x,y),...], ...]


def empty(a):
    pass


cv2.namedWindow("Vals")
cv2.resizeWindow("Vals", 640, 240)
cv2.createTrackbar("Val1", "Vals", 25, 50, empty)
cv2.createTrackbar("Val2", "Vals", 16, 50, empty)
cv2.createTrackbar("Val3", "Vals", 5, 50, empty)


def checkSpaces():
    spaces = 0
    h_img, w_img = imgThres.shape[:2]
    for poly in posList:
        # Create mask for polygon (same size as thresholded image)
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        pts = np.array(poly, np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # Apply mask to thresholded image and count non-zero
        imgCrop = cv2.bitwise_and(imgThres, imgThres, mask=mask)
        count = cv2.countNonZero(imgCrop)

        # Heuristic: scale threshold by polygon area (optional)
        area = cv2.contourArea(pts)
        # Base threshold 900 used previously corresponds to roughly 103*43*0.2 ~ 885 (empirical)
        # We'll derive threshold as min(900, area*0.2*1.2) to adapt to different polygon sizes
        adaptive_thresh = max(300, int(area * 0.18))

        if count < adaptive_thresh:
            color = (0, 200, 0)
            thic = 3
            spaces += 1
        else:
            color = (0, 0, 200)
            thic = 2

        # Draw polygon and count
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thic)
        # put count near polygon centroid
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        cv2.putText(img, str(count), (cx - 10, cy + 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    cvzone.putTextRect(img, f'Free: {spaces}/{len(posList)}', (50, 60), thickness=3, offset=20,
                       colorR=(0, 200, 0))


while True:

    # Get image frame
    success, img = cap.read()
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # img = cv2.imread('img.png')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    # ret, imgThres = cv2.threshold(imgBlur, 150, 255, cv2.THRESH_BINARY)

    val1 = cv2.getTrackbarPos("Val1", "Vals")
    val2 = cv2.getTrackbarPos("Val2", "Vals")
    val3 = cv2.getTrackbarPos("Val3", "Vals")
    if val1 % 2 == 0: val1 += 1
    if val3 % 2 == 0: val3 += 1
    imgThres = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, val1, val2)
    imgThres = cv2.medianBlur(imgThres, val3)
    kernel = np.ones((3, 3), np.uint8)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)

    checkSpaces()
    # Display Output

    cv2.imshow("Image", img)
    # cv2.imshow("ImageGray", imgThres)
    # cv2.imshow("ImageBlur", imgBlur)
    key = cv2.waitKey(1)
    if key == ord('r'):
        pass