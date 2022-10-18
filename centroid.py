import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


class Detector:

    def __init__(self, img_path='data/frame0.jpg', ball_color_path='ball_color.jpg', DEBUG=False):
        self.img, self.ball_size_px = None, None
        self.load_img(img_path)
        self.ball_color = self._retrieve_ball_color(ball_color_path)
        self.color_tile = np.tile(self.ball_color, (self.img.shape[0], self.img.shape[1], 1))
        self.DEBUG = DEBUG

    def load_img(self, path):
        self.img = cv2.imread(path, cv2.IMREAD_COLOR)
        self.ball_size_px = self._retrieve_ball_size(path)

    @staticmethod
    def _show_image(self, img):
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", 700, 1000)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    @staticmethod
    def _retrieve_ball_color(ball_color_path):
        img = cv2.imread(ball_color_path, cv2.IMREAD_COLOR)
        return np.mean(img, axis=(0, 1))

    @staticmethod
    def retrieve_ball_size(img_path):
        img = cv2.imread(img_path, 0)
        img = cv2.medianBlur(img,9)
        edges = cv2.Canny(img, 50, 100)
        cimg = cv2.imread(img_path, cv2.IMREAD_COLOR)
        circles = cv2.HoughCircles(edges,
                                   cv2.HOUGH_GRADIENT,
                                   1.5,
                                   20,
                                   param1=30,
                                   param2=5,
                                   minRadius=50,
                                   maxRadius=75)
        if circles is None:
            return None

        circles = np.uint16(np.around(circles))

        for i in circles[0, :1]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        return int(circles[0, 0][2])

    def color_match(self):
        img = np.abs(np.subtract(self.img, self.color_tile))

        if self.DEBUG:
            print(img.shape)

        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
        img = np.invert(img)
        fimg = np.zeros_like(img)
        fimg[img >= 230] = 255

        if self.DEBUG:
            self._show_image(fimg)
            self._show_image(self.img)
            plt.imsave("binarized.png", fimg)
            time1 = time.time()
        circles = self._crop_circles(fimg)

        if self.DEBUG:
            time2 = time.time()
            print(time2-time1)

        #self.ball_size_px *=1.6
        for i in circles[0, :1]:
            img = np.copy(self.img)
            cv2.circle(img,(i[0],i[1]),i[2],(255,87,51),3)
            cv2.circle(img,(i[0],i[1]),2,(255,87,51),4)

            if self.DEBUG:
                self._show_image(img)
                plt.imsave("center_of_mass.png", img)
            return circles
        else:
            return None

    @staticmethod
    def _find_center_of_circle(img):
        circles = cv2.HoughCircles(img,
                                   cv2.HOUGH_GRADIENT,
                                   2,
                                   20,
                                   param1=30,
                                   param2=5,
                                   minRadius=0,
                                   maxRadius=100)
        circles = np.uint16(np.around(circles))
        return circles

    @staticmethod
    def _crop_circles(img):
        posns = np.where(img == 255)
        if len(posns[0]) != 0:
            posnx = int(np.mean(posns[1]))
            posny = int(np.mean(posns[0]))
            img = img[posny - 100: posny + 100, posnx - 100:posnx+100]
        circles = Detector._find_center_of_circle(img)
        for i in circles[0, :1]:
            i[0] = i[0] + posnx - 100
            i[1] = i[1] + posny - 100
        return circles

    @staticmethod
    def _write_frames(vid_path, output_dir='data/'):
        vidcap = cv2.VideoCapture(vid_path)
        success, img = vidcap.read()
        count = 0
        while success:
            # stuff
            cv2.imwrite("{}frame{}.jpg".format(output_dir, count), img)
            success, img = vidcap.read()
            count += 1


if __name__ == '__main__':
    detector = Detector()
    detector.color_match()
