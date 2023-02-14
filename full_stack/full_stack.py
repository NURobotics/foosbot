import cv2
import numpy as np

MEASURED_PERIMETER = 38

def resize_img(img, scale_percent=50):
    # scale percent - percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def get_roi_topleft(img):
    roi = cv2.selectROI("Select ball", img)
    # [Top_Left_X, Top_Left_Y, Width, Height]
    cv2.destroyWindow("Select ball")
    return [roi[0], roi[1]]

def main():
    #load aruco parameters
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)

    img = cv2.imread("test_grid_image.png")
    # img = cv2.imread("basic_aruco.jpg")
    img = resize_img(img, scale_percent=30)

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

    top_left_real_world = [1, 1]

    image_size = 3.8

    # dst = np.asarray([[0,0], [image_size,0], [image_size, image_size], [0, image_size]])

    dst = np.asarray([
        np.add([0, 0], top_left_real_world),
        np.add([image_size, 0], top_left_real_world),
        np.add([image_size, image_size], top_left_real_world),
        np.add([0, image_size], top_left_real_world)
    ])

    print(dst)

    dst *= 100

    # Draw the coordinates of the ArUCo detection
    # cv2.putText(img, "TL", tuple(corners[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # cv2.putText(img, "TR", tuple(corners[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # cv2.putText(img, "BR", tuple(corners[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # cv2.putText(img, "BL", tuple(corners[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv2.imshow("image", img)

    transform_matrix = cv2.getPerspectiveTransform(np.float32(corners[0]), np.float32(dst))

    print(transform_matrix)

    dimensions = (1100, 1500)

    cv2.imshow("image", img)
    cv2.waitKey(0)

    # use the perspective transform matrix to warp the image
    warped = cv2.warpPerspective(img, transform_matrix, dimensions)

    # In this scale, each inch is 100 pixels

    # Draw a 1 inch grid across the image
    top_left = (0, 0)

    for i in range(0, 15):
        for j in range(0, 15):
            bottom_right = (top_left[0] + 100, top_left[1] + 100)
            cv2.rectangle(warped, top_left, bottom_right, (0, 255, 0), 2)
            top_left = (top_left[0], top_left[1] + 100)
        top_left = (top_left[0] + 100, 0)


    cv2.imshow("warped", warped)

    # Save the warped image
    cv2.imwrite("warped.png", warped)
    cv2.waitKey(0)

main()