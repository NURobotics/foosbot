
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

VID_PATH = "simple_sample.MOV"

def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)

def get_ball_color():
    img = cv2.imread('ball_color.jpg', cv2.IMREAD_COLOR)
    return np.mean(img, axis=(0, 1))

def get_ball_size():
    img = cv2.imread('frame0.jpg', 0)
    img = cv2.medianBlur(img,9)
    edges = cv2.Canny(img, 50, 100)
    cimg = cv2.imread('frame0.jpg', cv2.IMREAD_COLOR)
    circles = cv2.HoughCircles(edges,
                               cv2.HOUGH_GRADIENT,
                               1,
                               20,
                               param1=50,
                               param2=20,
                               minRadius=0,
                               maxRadius=120)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :1]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    return circles[0,0][2]

def color_match(img_path):
    start = time.time()
    ball_color = get_ball_color()
    ball_size_px = get_ball_size()
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    start = time.time()
    img = np.abs(np.subtract(img, np.tile(ball_color, (img.shape[0], img.shape[1], 1))))
    #print(img.shape)
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
    img = np.invert(img)
    fimg = np.zeros_like(img)
    fimg[img >= 230] = 255
    #show_image(cv2.imread(img_path, cv2.IMREAD_COLOR))
    #show_image(fimg)
    #plt.imsave("binarized.png", fimg)
    print("time to binarize: {}".format(time.time() - start))

    center = get_center_of_mass(fimg)
    ball_size_px *=1.6
    ball_size_px = int(ball_size_px)
    if center is not None:
        #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #print(center)
        #cv2.circle(img, center, ball_size_px, (0, 255, 0), 2)
        #show_image(img)
        print("time in total: {}".format(time.time() - start))
        #plt.imsave("center_of_mass.png", img)
        return center
    else:
        return None

def get_center_of_mass(img):
    posns = np.where(img == 255)
    if len(posns[0]) != 0:
        return (int(np.mean(posns[1])), int(np.mean(posns[0])))
    return None



    """scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    ball_size_px *= scale_percent / 100
    best = (0, 0)
    best_score = 10000000
    for i in range(img.shape[0]):
        startx = max(i - ball_size_px, 0)
        endx = min(i + ball_size_px, img.shape[0])
        starty = max(j - ball_size_px, 0)
        endy = min(j + ball_size_px, img.shape[1])
        score = 0
        for inneri in range(startx, endx):
            for innerj in range(starty, endy):
                score += color_distance(img[inneri, innerj], ball_color)
        for j in range(img.shape[1]):
            print(j)
            if score < best_score:
                best = (i, j)
                best_score = score

    cv2.circle(img, best, ball_size_px, (0, 255, 0), 2)
    show_image(img)"""

def get_score(img, pos, ball_size_px, ball_color):
    startx = max(pos[0] - ball_size_px, 0)
    endx = min(pos[0] + ball_size_px, img.shape[0])
    starty = max(pos[1] - ball_size_px, 0)
    endy = min(pos[1] + ball_size_px, img.shape[1])

    score = 0
    for i in range(startx, endx):
        for j in range(starty, endy):
            score += color_distance(img[i, j], ball_color)

    return score

def color_distance(col1, col2):
    return abs(col1[0] - col2[0]) + abs(col1[1] - col2[1]) + abs(col1[2] - col2[2])

def write_frames(vid_path):
    vidcap = cv2.VideoCapture(VID_PATH)
    success, img = vidcap.read()
    count = 0
    while success:
        # stuff
        cv2.imwrite("frame{}.jpg".format(count), img)
        success, img = vidcap.read()
        count += 1

def track_posns(base_path="frame", count=500):
    last_frame = None
    curr_frame = color_match(base_path + "{}.jpg".format(0))
    for i in range(1, count):
        print("Current center: x: {} y: {}".format(curr_frame[0], curr_frame[1]))
        last_frame = curr_frame
        curr_frame = color_match(base_path + "{}.jpg".format(i))
        x = curr_frame[0] - last_frame[0]
        y = curr_frame[1] - last_frame[1]
        print("Prediction for next center: x: {} y: {}".format(x + last_frame[0], y + last_frame[1]))


"""
Eventual Algorithm:

Place ball at certain points on the table to callibrate color.


"""

if __name__ == '__main__':
    ###track_posns()
    for i in range(0, 50, 10):
        print(i)
        color_match('frame{}.jpg'.format(i))

    """for i in range(4):
        color_match('table_test{}.jpg'.format(i))"""

    color_match('table_test0.jpg')
