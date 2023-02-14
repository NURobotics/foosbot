from tracking_live import *
import cv2 as cv

# define a video capture object
vid = cv.VideoCapture(0)
_frames = []
while(True):
  # Capture the video frame by frame
  ret, frame = vid.read()
  
  # Do tracking analysis
  frame = draw_circles(frame)
  if len(_frames) >= 2:
    _frames.pop(0)
    _frames.append(frame)
    cv.imshow('frame', _frames[-1])
  else:
    _frames.append(frame)
  # Display the resulting frame
    
  # the 'q' button is set as the
  # quitting button you may use any
  # desired button of your choice
  if cv.waitKey(1) & 0xFF == ord('q'):
    break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()