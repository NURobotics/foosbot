import cv2
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

    img = cv2.imread("basic_aruco.jpg")
    img = resize_img(img, scale_percent=20)

    # detect ArUco markers in the input frame
    (all_corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

    # verify *at least* one ArUco marker was detected
    if len(all_corners) == 0:
        print("No ArUco markers detected")
        return

    markerCorner = all_corners[0]

    corners = np.int0(markerCorner)
    # (topLeft, topRight, bottomRight, bottomLeft) = reorder(corners[0])
    print("corners:", corners[0])

    image_size = 4

    dst = np.asarray([[0,0], [image_size,0], [image_size, image_size], [0, image_size]])

    dst *= 100

    # Draw the coordinates of the ArUCo detection
    cv2.putText(img, "TL", tuple(corners[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(img, "TR", tuple(corners[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(img, "BR", tuple(corners[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(img, "BL", tuple(corners[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv2.imshow("image", img)

    transform_matrix = cv2.getPerspectiveTransform(np.float32(corners[0]), np.float32(dst))

    dimensions = (img.shape[1] * 2, img.shape[0] * 2)

    # use the perspective transform matrix to warp the image
    warped = cv2.warpPerspective(img, transform_matrix, dimensions)

    # Draw a red square on the warped image halfway down the image, 200x200 pixels

    cv2.rectangle(warped, (200, 200), (400, 400), (0, 0, 255), 5)

    # In this scale, each inch is 100 pixels

    # Draw a 1 inch box

    top_left = (0, 430)
    bottom_right = (top_left[0] + 100, top_left[1] + 100)


    cv2.rectangle(warped, top_left, bottom_right, (0, 255, 0), 5)

    cv2.imshow("warped", warped)
    cv2.waitKey(0)

main()