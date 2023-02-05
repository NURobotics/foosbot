"""
This code assumes that images used for calibration are of the same arUco marker board provided with code

"""

import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set this flag True for calibrating camera and False for validating results real time
calibrate_camera = False

# Set path to the images
calib_imgs_path = root.joinpath("greg")

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )

#Provide length of the marker's side
markerLength = 3.1  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.5   # Here, measurement unit is centimetre.

# create arUco board
board = aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)

'''uncomment following block to draw and show the board'''
# img = board.draw((864,1080))
# cv2.imshow("aruco", img)

arucoParams = aruco.DetectorParameters_create()

if calibrate_camera == True:
    img_list = []
    calib_fnms = calib_imgs_path.glob('*.jpg')
    print('Using ...', end='')
    for idx, fn in enumerate(calib_fnms):
        print(idx, '', end='')
        img = cv2.imread( str(root.joinpath(fn) ))
        img_list.append( img )
        h, w, c = img.shape
    print('Calibration images')

    counter, corners_list, id_list = [], [], []
    first = True
    for i, im in enumerate(tqdm(img_list)):
        cv2.imshow("img", im)
        cv2.waitKey(200)
        img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))
    print('Found {} unique markers'.format(np.unique(ids)))

    counter = np.array(counter)
    print ("Calibrating camera .... Please wait...")
    #mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)

else:
    camera = cv2.VideoCapture(1)
    ret, img = camera.read()

    with open('calibration.yaml') as f:
        loadeddict = yaml.safe_load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = img_gray.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    pose_r, pose_t = [], []
    while True:
        ret, img = camera.read()
        img_aruco = img
        im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        h,  w = im_gray.shape[:2]
        dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
        #cv2.imshow("original", img_gray)
        if corners == None:
            print ("pass")
        else:

            ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist, np.empty(1), np.empty(1)) # For a board
            #print ("Rotation ", rvec, "Translation", tvec, end="\r")
            print(f"Translation: x: {tvec}", end='\r') 
            #print(f"Translation: x: {tvec[1][0]}", end='\r') #, y: {round(tvec[1])}, z: {round(tvec[2])}
            if ret != 0:
                img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
                #img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 10)    # axis length 100 can be changed according to your requirement

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break;
            elif key == ord('c'):
                cv2.waitKey(0)
                
        cv2.imshow("World co-ordinate frame axes", img_aruco)
        #cv2.waitKey()

cv2.destroyAllWindows()
