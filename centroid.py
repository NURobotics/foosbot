import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

VID_PATH = "simple_sample.MOV"

def show_image(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 700, 1000)              
    cv2.imshow("img", img)                             
    cv2.waitKey(0)   

def get_ball_color():
    img = cv2.imread('ball_color.jpg', cv2.IMREAD_COLOR)
    return np.mean(img, axis=(0, 1))

def get_ball_size(img_path):
    img = cv2.imread(img_path, 0)
    img = cv2.medianBlur(img,9)
    edges = cv2.Canny(img, 50, 100)
    cimg = cv2.imread(img_path, cv2.IMREAD_COLOR)
    circles = cv2.HoughCircles(edges,
                               cv2.HOUGH_GRADIENT,
                               1.5,
                               20,
                               param1=100,
                               param2=5,
                               minRadius=0,
                               maxRadius=75)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :1]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    return circles[0,0][2]

def color_match(img_path):
    ball_color = get_ball_color()
    ball_size_px = get_ball_size(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.abs(np.subtract(img, np.tile(ball_color, (img.shape[0], img.shape[1], 1))))
    #print(img.shape)
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
    img = np.invert(img)
    fimg = np.zeros_like(img)
    fimg[img >= 230] = 255
    show_image(fimg)
    show_image(cv2.imread(img_path, cv2.IMREAD_COLOR))
    plt.imsave("binarized.png", fimg)
    time1 = time.time()
    circles = crop_circles(fimg)
    time2 = time.time()
    print(time2-time1)

    ball_size_px *=1.6
    ball_size_px = int(ball_size_px)
    for i in circles[0, :1]:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cv2.circle(img,(i[0],i[1]),i[2],(255,87,51),3)
        cv2.circle(img,(i[0],i[1]),2,(255,87,51),4)
        show_image(img)
        #plt.imsave("center_of_mass.png", img)
        return circles
    else:
        return None

def find_center_of_circle(img):
    circles = cv2.HoughCircles(img,
                               cv2.HOUGH_GRADIENT,
                               2,
                               20,
                               param1=30,
                               param2=5,
                               minRadius=75,
                               maxRadius=100)
    circles = np.uint16(np.around(circles))
    return circles
    for i in circles[0, :1]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(250,150,150),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(155,150,150),3)
    cv2.waitKey(0)

def crop_circles(img):  
    posns = np.where(img == 255)
    if len(posns[0]) != 0:
        posnx = int(np.mean(posns[1]))
        posny = int(np.mean(posns[0]))
        img = img[posny - 100: posny + 100, posnx - 100:posnx+100]
    circles = find_center_of_circle(img)
    for i in circles[0, :1]:
        i[0] = i[0] + posnx - 100
        i[1] = i[1] + posny - 100
    return circles


def write_frames():
    vidcap = cv2.VideoCapture(VID_PATH)
    success, img = vidcap.read()
    count = 0
    while success:
        # stuff
        cv2.imwrite("data/frame{}.jpg".format(count), img)
        success, img = vidcap.read()
        count += 1

if __name__ == '__main__':
    # color_match("table_test2.jpg")
    color_match('data/frame0.jpg')
