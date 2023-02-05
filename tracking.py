import cv2 as cv
import numpy as np

BALL_COLOR = "ball_color.jpg"
FPS = 30
TEST_IMG = "data/frame0.jpg"

class velocity_px:
  def __init__(self, v_x: float, v_y: float):
    self.x = v_x
    self.y = v_y
    self.v = sqrt(v_x*v_x + v_y*v_y)

  def __str__(self):
    return f"vx: {self.x}, vy: {self.y}"

class center_pos:
  def __init__(self, cx: int, cy: int):
    if (cx < 0) or (cy < 0):
      return None
    self.x = cx
    self.y = cy

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

def find_circle(img: np.ndarray):
  # HoughCircles can only receive grayscale images
  circles_found = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, 20, param1=30, param2=5, minRadius=5, maxRadius=150)
  if circles_found is None:
    return None
  # find_circle is returning numbers as np.float32
  return circles_found[0, 0]

def draw_circles(img_path: str):
  img = cv.imread(img_path, cv.IMREAD_COLOR)
  circle = find_circle(color_match(img_path))
  if circle is None:
    print("could not draw circles because there is no circle")
    return
  show_img(cv.circle(img, (circle[0], circle[1]), circle[2], (255,87,51)))

def compute_velocity(curr_center_pos, past_center_pos, delta_t: float) -> velocity_px:
  vx = (curr_center_pos.x - past_center_pos.x)/delta_t
  vy = (curr_center_pos.y - past_center_pos.y)/delta_t
  vel = velocity_px(vx, vy)
  return vel

def ball_vel_from_consec_frames(frame_path: str, past_frame_path: str):
  center = find_circle(color_match(frame_path))
  if center is None:
    return None
  center = center_pos(center[0], center[1])

  past_center = find_circle(color_match(past_frame_path))
  if past_center is None:
    return None
  past_center = center_pos(past_center[0], past_center[1])
  
  delta_t = 1 / FPS
  return compute_velocity(center, past_center, delta_t)

def vels_many_consec_frames(start: int=1, end: int=791):
  vels = []
  for i in range(start, end + 1):
    vels.append(ball_vel_from_consec_frames(f"data/frame{i}.jpg", f"data/frame{i-1}.jpg"))
  return vels

if __name__ == "__main__":
  vels = vels_many_consec_frames(1, 10)
  for vel in vels:
    print(vel)