# Script demonstrates how to remove the background from a live video feed

import cv2
import numpy as np
import time

# Capture the video feed at a 640x480 resolution

# Create a VideoCapture object
cap = cv2.VideoCapture(1)

# Set the resolution
cap.set(3, 640)
cap.set(4, 480)


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")
    exit()

# Save the initial frame
ret, frame = cap.read()
if ret == False:
    print("Unable to read camera feed")
    exit()

initial_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

last_time = time.time()

min_frame_rate = np.inf

# Display the feed
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        print("Unable to read camera feed")
        exit()

    start_time = time.time()

    # remove the background by checking if the greyscale difference between the current frame and the initial frame is less than 50
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the difference between the initial frame and the current frame
    diff = cv2.absdiff(initial_frame, grey_frame)

    # Create a mask for the difference
    mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Create a masked image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the time it took to process the frame in seconds

    # frame_rate = 1 / (time.time() - start_time)
    # last_time = time.time()

    # print(f"Frame rate: {frame_rate:.2f} fps")

    # min_frame_rate = min(min_frame_rate, frame_rate)

    # Display the resulting frame
    cv2.imshow('frame', masked_image)

    # # Each second, save the current frame as the initial frame
    # if time.time() - last_time > 1:
    #     initial_frame = grey_frame
    #     last_time = time.time()

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # If the user presses the space bar, save the current frame as the initial frame
    if cv2.waitKey(25) & 0xFF == ord(' '):
        initial_frame = grey_frame

print("Min frame rate:", min_frame_rate)