import cv2 as cv
import numpy as np
from math import sqrt
import time

BALL_COLOR_IMG = "test/ball_color.jpg"
FPS = 30
TEST_IMG = "data/frame0.jpg"
VID_PATH = "test_with_aruca/test_with_aruca.mov"
ball_color_rgb = None
_past_frame_path = None
_past_frame_center = None
_frame_circle = {}

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

def write_frames(folder: str):
  vidcap = cv.VideoCapture(VID_PATH)
  success, img = vidcap.read()
  count = 0
  while success:
    print(count)
    cv.imwrite("{}/frame{}.jpg".format(folder, count), img)
    success, img = vidcap.read()
    count += 1

def get_ball_color() -> list[np.ndarray]:
  img = cv.imread(BALL_COLOR_IMG, cv.IMREAD_COLOR)
  return np.mean(img, axis=(0, 1))

def color_match(img_path: str):
  img = cv.imread(img_path, cv.IMREAD_COLOR)
  global ball_color_rgb
  if ball_color_rgb is None:
    ball_color_rgb = get_ball_color()
    print(ball_color_rgb)
  img_with_ball_color = np.tile(ball_color_rgb, (img.shape[0], img.shape[1], 1))
  img = np.abs(np.subtract(img, img_with_ball_color))
  img = cv.cvtColor(img.astype("uint8"), cv.COLOR_BGR2GRAY)
  black_white_img = np.zeros_like(img)
  black_white_img[img < 20] = 255
  black_white_img[img > 235] = 255
  return black_white_img

def crop_circles(img):  
  posns = np.where(img == 255)
  if len(posns[0]) != 0:
      posnx = int(np.mean(posns[1]))
      posny = int(np.mean(posns[0]))
      half_window_size = 150
      half_window_size_x = min(posnx, half_window_size)
      half_window_size_y = min(posny, half_window_size)
      return img[posny - half_window_size_y: posny + half_window_size_y, posnx - half_window_size_x:posnx+half_window_size_x], posnx - half_window_size_x, posny - half_window_size_y
  else:
    return img, 0, 0

def find_circle(img: np.ndarray, img_path: str):
  if len(_frame_circle.get(img_path, [])):
    return _frame_circle[img_path]
  img, x_offset, y_offset = crop_circles(img)
  # HoughCircles can only receive grayscale images
  circles_found = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, 20, param1=30, param2=5, minRadius=5, maxRadius=150)
  if circles_found is None:
    return None
  # find_circle is returning numbers as np.float32
  main_circle = circles_found[0, 0]
  main_circle[0] += x_offset
  main_circle[1] += y_offset
  _frame_circle[img_path] = main_circle
  return main_circle

def draw_circles(img_path: str):
  img = cv.imread(img_path, cv.IMREAD_COLOR)
  circle = find_circle(color_match(img_path), img_path)
  if circle is None:
    print("could not draw circles because there is no circle")
    return
  show_img(cv.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (20,255,57)))

def compute_velocity(curr_center_pos, past_center_pos, delta_t: float) -> velocity_px:
  vx = (curr_center_pos.x - past_center_pos.x)/delta_t
  vy = (curr_center_pos.y - past_center_pos.y)/delta_t
  vel = velocity_px(vx, vy)
  return vel

def ball_vel_from_consec_frames(frame_path: str, past_frame_path: str):
  global _past_frame_path
  global _past_frame_center
  if past_frame_path == _past_frame_path:
    past_center = _past_frame_center
  else:
    past_center = find_circle(color_match(past_frame_path), past_frame_path)
    if past_center is None:
      return velocity_px(0, 0)
    past_center = center_pos(past_center[0], past_center[1])
  center = find_circle(color_match(frame_path), frame_path)
  if center is None:
    return velocity_px(0, 0)
  center = center_pos(center[0], center[1])
  _past_frame_path = frame_path
  _past_frame_center = center
  
  delta_t = 1 / FPS
  print("found vel")
  return compute_velocity(center, past_center, delta_t)

def vels_many_consec_frames(folder, end, start: int=1, format="list"):
  if format == "list":
    vels = []
    for i in range(start, end + 1):
      vels.append(ball_vel_from_consec_frames(f"{folder}/frame{i}.jpg", f"{folder}/frame{i-1}.jpg"))
    return vels
  elif format == "dict":
    vels = {}
    for i in range(start, end + 1):
      print(f"frame {i}")
      vels[i] = ball_vel_from_consec_frames(f"{folder}/frame{i}.jpg", f"{folder}/frame{i-1}.jpg")
    return vels
  else:
    raise ValueError('{format} wrong, use "list" or "dict"'.format(format=repr(format)))

def smooth_average(vels_dict):
  vels_list = list(vels_dict.values())
  vels_x = [0.1*sum([vel.x for vel in vels_list[i-9:i+1]]) for i in range(9, len(vels_list))]
  vels_y = [0.1*sum([vel.y for vel in vels_list[i-9:i+1]]) for i in range(9, len(vels_list))]
  counter = 1
  l = list(vels_dict.keys())
  for i in l:
    if counter < 10:
      vels_dict.pop(i)
      counter += 1
    else:
      vels[i] = velocity_px(vels_x.pop(0), vels_y.pop(0))
  return vels_dict

def frames_with_vel_arrow(folder, vels_dict, end, start: int=0):
    frames = []
    for i in range(start, end + 1):
      print(i)
      img = cv.imread(f"{folder}/frame{i}.jpg", cv.IMREAD_COLOR)
      if vels_dict.get(i) and (vels_dict[i].x != 0 and vels_dict[i].y != 0):
        center = find_circle(color_match(f"{folder}/frame{i}.jpg"), f"{folder}/frame{i}.jpg")
        if center is None:
          frames.append(img)
          continue
        center = (int(center[0]), int(center[1]))
        arrow_tip = (center[0] + int(vels_dict[i].x/2), center[1] + int(vels_dict[i].y/2))
        img_arrow = cv.arrowedLine(img, center, arrow_tip, (0, 255, 0), 5)
        frames.append(img_arrow)
        continue
      frames.append(img)
    return frames

def video_from_frames(frame_array):
    height,width,layers=frame_array[1].shape
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    video=cv.VideoWriter('vels10avg30fpsopt1.mp4',fourcc,30,(width,height))
    for frame in frame_array:
      video.write(frame)
    video.release()

if __name__ == "__main__":
  start_get_vels = time.time()
  vels = vels_many_consec_frames("test_with_aruca", end=901, format="dict")
  print("got all vels")
  end_get_vels = time.time()
  start_smooths = time.time()
  vels = smooth_average(vels)
  end_smooths = time.time()
  start_vid = time.time()
  video_from_frames(frames_with_vel_arrow(folder="test_with_aruca", vels_dict=vels, end=901))
  end_vid = time.time()
  print(f"Time to get the velocity of all frames: {end_get_vels-start_get_vels}s; per frame: \
  {(end_get_vels-start_get_vels)*1000/900}ms")
  print(f"Time to smooth all velocities: {end_smooths-start_smooths}s; per frame: \
    {(end_smooths-start_smooths)*1000/900}ms")
  print(f"Time to draw the arrows and assemble the frames into a video: {end_vid-start_vid}s; \
    per frame: {(end_vid-start_vid)*1000/900}ms")
  # draw_circles("test/background-black.jpg")
  # draw_circles("test/background-gray.jpg")
  # draw_circles("test/background-metallic.jpg")
  # draw_circles("data/frame0.jpg")
  # draw_circles("stable_video/frame100.jpg")