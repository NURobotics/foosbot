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

def get_roi_topleft(img):
    roi = cv2.selectROI("Select ball", img)
    # [Top_Left_X, Top_Left_Y, Width, Height]
    cv2.destroyWindow("Select ball")
    return [roi[0], roi[1]]

def main():
    #load aruco parameters
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)

    img = cv2.imread("ruler_rotated.jpg")
    img = resize_img(img, scale_percent=50)

    # detect ArUco markers in the input frame
    (all_corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    
    # verify *at least* one ArUco marker was detected
    if len(all_corners) == 0:
        return
    
    markerCorner = all_corners[0]

    corners = np.int0(markerCorner)
    # (topLeft, topRight, bottomRight, bottomLeft) = reorder(corners[0])
    print("corners:", corners[0])
    
    dst = np.asarray([[-4,0], [0,0], [0,-4], [-4,-4]])
    print("dst:", dst)
    transform_matrix = cv2.getPerspectiveTransform(np.float32(corners[0]), np.float32(dst))

    maxWidth = img.shape[1] * 2
    maxHeight = img.shape[0] *2

    print("M:", transform_matrix)
    
    dst = cv2.warpPerspective(corners, transform_matrix, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)


    # start_point = get_roi_topleft(img)
    # cv2.circle(img, start_point, 4, (100, 190, 255), -1)
    # print("start:", start_point)
    # end_point = get_roi_topleft(img)
    # cv2.circle(img, end_point, 4, (100, 190, 255), -1)
    # print("end", end_point)

    # y_dist_px = end_point[1] - start_point[1]
    # x_dist_px = end_point[0] - start_point[0]
    # intersection_point = [start_point[0], end_point[1]]
    # print("corner", intersection_point)
    # print("dists", x_dist_px, y_dist_px)

    # #aruco perimeter
    # aruco_perimeter = cv2.arcLength(markerCorner, True)

    # #pixel to cm ratio
    # pixel_cm_ratio = aruco_perimeter/MEASURED_PERIMETER
    # print(pixel_cm_ratio)

    # cm_x = abs(x_dist_px) / pixel_cm_ratio
    # cm_y = abs(y_dist_px) / pixel_cm_ratio
    # print(cm_x, cm_y)

    # cv2.line(img, start_point, intersection_point, (13, 89, 0), 2) #horizontal
    # cv2.line(img, intersection_point, end_point, (13, 89, 0), 2) #vertical

    # cv2.putText(img, str(abs(x_dist_px)),
    #             (intersection_point[0] + int(x_dist_px/2) - 5, intersection_point[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 5, 12), 2)
    # cv2.putText(img, str(abs(y_dist_px)),
    #             (intersection_point[0] - 50, intersection_point[1] + int(y_dist_px/2) - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 5, 12), 2)

    # cv2.putText(img, str(round(cm_x, 1))+" cm",
    #             (intersection_point[0] + int(x_dist_px/2) - 5, intersection_point[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 5, 12), 2)
    # cv2.putText(img, str(round(cm_y, 1))+" cm",
    #             (intersection_point[0] + 20, intersection_point[1] + int(y_dist_px/2) - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 5, 12), 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)

main()