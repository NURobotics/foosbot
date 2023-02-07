import cv2
import utilis
import numpy as np


def prepImage(img, canny_thresh=[100,100]):
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, canny_thresh[0], canny_thresh[1])
    kernel = np.ones((5,5))
    imgDilated = cv2.dilate(imgCanny, kernel, iterations=3)
    imgEroded = cv2.erode(imgDilated, kernel, iterations=2)

    return imgEroded

def resize_img(img, scale_percent=50):
    # scale percent - percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def startVideo():
    cap = cv2.VideoCapture(0)
    cap.set(10, 160)
    cap.set(3, 1920)
    cap.set(4, 1080)

    #load aruco parameters
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)

    while True:
        success, frame = cap.read()
        img = resize_img(frame, scale_percent=30)
        # img = cv2.imread("ventra.jpg")
        # img = resize_img(img, scale_percent=20)

        # detect ArUco markers in the input frame
        (all_corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        
        if len(all_corners) > 0:
            #only usinf one
            markerCorner = all_corners[0]
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            # corners = np.int0(markerCorner.reshape((4, 2)))
            corners = np.int0(markerCorner)
            # (topLeft, topRight, bottomRight, bottomLeft) = corners
            cv2.polylines(img, corners, True, (0, 255, 100), 2)

            #aruco perimeter
            aruco_perimeter = cv2.arcLength(markerCorner, True)

            #pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter/38

            contours, hierarchy = cv2.findContours(prepImage(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_thresh = 800

            # print(contours)

            for cnt in contours:
                #draw outline
                # cv2.polylines(img, [cnt], True, (255, 0, 100), 2)

                #threshold for area size
                area = cv2.contourArea(cnt)
                if area > area_thresh:
                    #bounding box
                    rect = cv2.minAreaRect(cnt)
                    (x, y), (w, h), angle = rect
                    (x, y) = (int(x), int(y))

                    #get width and hight of the objects by applying the ratio to cm
                    object_w = w / pixel_cm_ratio
                    object_h = h / pixel_cm_ratio

                    #center dot
                    cv2.circle(img, (x, y), 3, (0, 100, 255), -1)
                    #rectangle
                    box = np.int0(cv2.boxPoints(rect))
                    cv2.polylines(img, [box], True, (255, 0, 100), 2)
                    #text
                    cv2.putText(img, f"{round(object_w, 1)}x{round(object_h, 1)} cm", (x + 5, y - 15), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)


        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

startVideo()