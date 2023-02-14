from dataclasses import dataclass
import warnings
import cv2
import numpy as np

@dataclass
class InchCoordinate:
    x: float
    y: float

@dataclass
class PixelCoordinate:
    x: int
    y: int

def resize_img(img, scale_percent):
    # scale percent - percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

class ArucoTransformer:
    def __init__(self, image_size_in=3.8, scale_percent=50):
        self.transformaton_matrix = None
        self.px_per_in = 100
        self.image_size_in = image_size_in
        self.scale_percent = scale_percent

    """
    Returns the transformation matrix for the image, found by using the aruco markers

    :param image: the image to get the transformation matrix for; should be a numpy array
    :return: the transformation matrix for the image; will be a numpy array

    """
    def set_transformation_matrix(self, image):
        arucoParams = cv2.aruco.DetectorParameters_create()
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)

        image = resize_img(image, scale_percent=self.scale_percent)

        (all_corners, ids, rejected) = cv2.aruco.detectMarkers(image,
                                                               arucoDict,
                                                               parameters=arucoParams)

        if len(all_corners) == 0:
            print("No ArUco markers detected")
            return

        markerCorner = all_corners[0]

        corners = np.int0(markerCorner)

        print("corners:", corners[0])

        image_size = self.image_size_in

        top_left_real_world = [3, 3]

        dst = np.asarray([
            np.add([0, 0], top_left_real_world),
            np.add([image_size, 0], top_left_real_world),
            np.add([image_size, image_size], top_left_real_world),
            np.add([0, image_size], top_left_real_world)
        ])

        print(dst)

        dst *= self.px_per_in

        self.transformaton_matrix = cv2.getPerspectiveTransform(np.float32(corners[0]), np.float32(dst))

        return self.transformaton_matrix


    """
    Returns the transformation matrix for the image based on the current
    transformation matrix

    :param image: the image to get the transformation matrix for; should be a numpy array
    :param include_grid: if true, will return the image with a 1 in grid drawn on it
    :return: the transformation matrix for the image; will be a numpy array
    """
    def convert_image(self, image, include_grid=False):
        image = resize_img(image, scale_percent=self.scale_percent)

        if self.transformaton_matrix is None:
            # raise warning
            warnings.warn("Warning in convert_image: transformation matrix not set")
            return None

        dimensions = (1100, 1500)

        # cv2.imshow("image", image)
        # cv2.waitKey(0)

        warped_img = cv2.warpPerspective(image, self.transformaton_matrix, dimensions)

        # if include_grid:
        top_left = (0, 0)

        for i in range(0, 15):
            for j in range(0, 15):
                bottom_right = (top_left[0] + 100, top_left[1] + 100)
                cv2.rectangle(warped_img, top_left, bottom_right, (0, 255, 0), 2)
                top_left = (top_left[0], top_left[1] + 100)
            top_left = (top_left[0] + 100, 0)

        # image will be a numpy array
        # returns a numpy array with the image transformed based on the transformation matrix
        return warped_img

    """
    Returns the x and y coordinates of the aruco marker in inches

    :param x: the x coordinate of the aruco marker in pixels
    :param y: the y coordinate of the aruco marker in pixels
    :param bypass: if true, will just return the x and y coordinates in inches in a PixelCoordinate object
    :return: the x and y coordinates of the aruco marker in inches in a InchCoordinate object. If error, returns None
    """
    def convert_to_inches(self, x, y, bypass=False):
        # Scale the point by the scale percent
        x *= (self.scale_percent / 100)
        y *= (self.scale_percent / 100)

        # Check if x and y are numeric
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            # raise warning
            warnings.warn("Warning in convert_to_inches: x and y must be numeric")

            return None

        if bypass:
            return InchCoordinate(x, y)

        if self.transformaton_matrix is None:
            # raise warning
            warnings.warn("Warning in convert_to_inches: transformation matrix not set")

            return None

        coord = np.asarray([[x, y, 1]])

        coord = np.matmul(coord, self.transformaton_matrix)
        print(coord)
        coordinate = InchCoordinate(coord[0][0] / self.px_per_in, coord[0][1] / self.px_per_in)

        return coordinate

if __name__ == "__main__":
    image = cv2.imread("test_grid_image.png")

    transformer = ArucoTransformer()

    transformer.set_transformation_matrix(image.copy())

    transformed_image = transformer.convert_image(image, include_grid=True)

    print(transformer.convert_to_inches(162, 156, bypass=False))


    cv2.circle(image, (162, 156), 5, (0, 0, 255), -1)
    cv2.imshow("transformed", transformed_image)
    cv2.waitKey(0)

    # cv2.imwrite("transformed.jpg", transformed_image)