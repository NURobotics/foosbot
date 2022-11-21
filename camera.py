
import cv2
from centroid import *



if __name__ == '__main__':
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    ball_color_tile = np.tile(get_ball_color(), (frame.shape[0], frame.shape[1], 1))

    while rval:
        frame = color_match_frame(frame, ball_color_tile)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")