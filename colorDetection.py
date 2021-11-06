import cv2
import numpy as np

##################
widthImg = 640
heightImg = 580
################
cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 110)


def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue max", "TrackBars", 20, 179, empty)
cv2.createTrackbar("Sat min", "TrackBars", 13, 255, empty)
cv2.createTrackbar("Sat max", "TrackBars", 245, 255, empty)
cv2.createTrackbar("Val min", "TrackBars", 164, 255, empty)
cv2.createTrackbar("Val max", "TrackBars", 255, 255, empty)


def preProcessing(img):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    Kernal = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, Kernal, iterations=2)
    imgThres = cv2.erode(imgDial, Kernal, iterations=1)

    return imgThres


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 5000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            if objCor >= 5:
                ObjectType = "Orange color Detected"
            else:
                ObjectType = "none"
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, ObjectType, (x + (w // 2) - 100, y + (h // 2) - 100), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                        (0, 255, 255), 3)


while True:

    success, img = cap.read()
    imgr = cv2.resize(img, (widthImg, heightImg))
    imgContour = imgr.copy()
    hsv = cv2.cvtColor(imgr, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val max", "TrackBars")

    # hsv hue set value

    lower_red = np.array([h_min, s_min, v_min])
    upper_red = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    res = cv2.bitwise_and(imgr, imgr, mask=mask)

    imgThres = preProcessing(res)
    getContours(imgThres)

    cv2.imshow("Vedio1", imgr)
    cv2.imshow("res", res)
    cv2.imshow("Image Thresh", imgThres)
    cv2.imshow("Image Contour", imgContour)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break