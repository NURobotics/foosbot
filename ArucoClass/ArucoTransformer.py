from dataclasses import dataclass
import warnings

@dataclass
class InchCoordinate:
    x: float
    y: float

@dataclass
class PixelCoordinate:
    x: int
    y: int

class ArucoTransformer:
    def __main__(self):
        self.transformaton_matrix = None
        self.px_per_in = 100

    """
    Returns the transformation matrix for the image, found by using the aruco markers

    :param image: the image to get the transformation matrix for; should be a numpy array
    :return: the transformation matrix for the image; will be a numpy array

    """
    def set_transformation_matrix(self, image):
        return None


    """
    Returns the transformation matrix for the image based on the current
    transformation matrix

    :param image: the image to get the transformation matrix for; should be a numpy array
    :return: the transformation matrix for the image; will be a numpy array
    """
    def convert_image(self, image):
        # image will be a numpy array
        # returns a numpy array with the image transformed based on the transformation matrix
        return image

    """
    Returns the x and y coordinates of the aruco marker in inches

    :param x: the x coordinate of the aruco marker in pixels
    :param y: the y coordinate of the aruco marker in pixels
    :param bypass: if true, will just return the x and y coordinates in inches in a PixelCoordinate object
    :return: the x and y coordinates of the aruco marker in inches in a InchCoordinate object. If error, returns None
    """
    def convert_to_inches(self, x, y, bypass=False):
        # Check if x and y are numeric
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            # raise warning
            warnings.warn("Warning in convert_to_inches: x and y must be numeric")

            return None

        if bypass:
            return InchCoordinate(x, y)

        return None

