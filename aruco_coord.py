import cv2
import utilis
import numpy as np

MEASURED_PERIMETER = 38

def resize_img(img, scale_percent=50):
    # scale percent - percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def reorder(points):
    new_points = np.zeros_like(points)
    points = points.reshape((4, 2))
    add = points.sum(1)

    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def get_ball_center(img):
    roi = cv2.selectROI("Select ball", img)
    # [Top_Left_X, Top_Left_Y, Width, Height]
    cX = roi[0] + int(roi[2] / 2)
    cY = roi[1] + int(roi[3] / 2)
    cv2.destroyWindow("Select ball")
    return [cX, cY]

def main():
    #load aruco parameters
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)

    img = cv2.imread("negball.jpg")
    img = resize_img(img, scale_percent=20)

    # detect ArUco markers in the input frame
    (all_corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    
    # verify *at least* one ArUco marker was detected
    if len(all_corners) == 0:
        return
    
    markerCorner = all_corners[0]
    # extract the marker corners (which are always returned in
    # top-left, top-right, bottom-right, and bottom-left order)
    # corners = np.int0(markerCorner.reshape((4, 2)))
    corners = np.int0(markerCorner)
    (topLeft, topRight, bottomRight, bottomLeft) = reorder(corners[0])
    print(topRight)
    cv2.polylines(img, corners, True, (89, 215, 100), 5)
    cv2.circle(img, topRight, 4, (0, 0, 255), -1)
    cv2.putText(img, "(0, 0)",
                (topRight[0], topRight[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 55, 255), 2)

    ball_center = get_ball_center(img) #[322, 125] #
    cv2.circle(img, ball_center, 4, (100, 190, 255), -1)
    print(ball_center)

    y_dist_px = topRight[1] - ball_center[1]
    x_dist_px = ball_center[0] - topRight[0]
    intersection_point = [topRight[0], ball_center[1]]
    print(intersection_point)
    print(x_dist_px, y_dist_px)

    #aruco perimeter
    aruco_perimeter = cv2.arcLength(markerCorner, True)

    #pixel to cm ratio
    pixel_cm_ratio = aruco_perimeter/MEASURED_PERIMETER
    print(pixel_cm_ratio)

    cm_x = abs(x_dist_px) / pixel_cm_ratio
    cm_y = abs(y_dist_px) / pixel_cm_ratio
    print(cm_x, cm_y)

    cv2.line(img, topRight, [topRight[0], ball_center[1]], (13, 89, 0), 4)
    cv2.line(img, intersection_point, ball_center, (13, 89, 0), 4)

    cv2.putText(img, str(abs(x_dist_px)),
                (intersection_point[0] + int(x_dist_px/2) - 5, intersection_point[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 5, 12), 2)
    cv2.putText(img, str(abs(y_dist_px)),
                (intersection_point[0] - 50, intersection_point[1] + int(y_dist_px/2) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 5, 12), 2)

    cv2.putText(img, str(round(cm_x, 1))+" cm",
                (intersection_point[0] + int(x_dist_px/2) - 5, intersection_point[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 5, 12), 2)
    cv2.putText(img, str(round(cm_y, 1))+" cm",
                (intersection_point[0] + 20, intersection_point[1] + int(y_dist_px/2) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 5, 12), 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)

main()