import cv2 as cv
import numpy as np

BALL_COLOR = "ball_color.jpg"
TEST_IMG = "data/frame0.jpg"

def show_img(img_path):
  cv.imshow("img", img_path)
  cv.waitKey()

def get_ball_color() -> list[np.ndarray]:
  img = cv.imread(BALL_COLOR, cv.IMREAD_COLOR)
  return np.mean(img, axis=(0, 1))

def color_match(img_path: str):
  img = cv.imread(img_path, cv.IMREAD_COLOR)
  ball_color_rgb = get_ball_color()
  img_with_ball_color = np.tile(ball_color_rgb, (img.shape[0], img.shape[1], 1))
  img = np.abs(np.subtract(img, img_with_ball_color))
  # Maybe we need to convert to gray because the img is in bgr and the 
  # show_img read rgb, and gray is the same for both idk
  img = cv.cvtColor(img.astype("uint8"), cv.COLOR_BGR2GRAY)
  img = np.invert(img)
  black_white_img = np.zeros_like(img)
  black_white_img[img > 230] = 255
  return black_white_img

def find_center_of_circle(img: np.ndarray):
  circles_found = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, 20, param1=30, param2=5, minRadius=5, maxRadius=150)
  circles_found = np.uint16(np.around(circles_found))
  return circles_found[0, 0]

def draw_circles(img_path: str):
  img = cv.imread(img_path, cv.IMREAD_COLOR)
  circle = find_center_of_circle(color_match(img_path))
  show_img(cv.circle(img, (circle[0], circle[1]), circle[2], (255,87,51)))

if __name__ == "__main__":
  draw_circles(TEST_IMG)
