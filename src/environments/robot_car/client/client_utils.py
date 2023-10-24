import cv2 as cv
import sys
from time import time, strftime, localtime


def get_timestamp_str():
    return strftime("%Y-%m-%d-%H-%M-%S", localtime(time()))


# Initialize webcams
# devnames: list of strings containing numbers ["0", "1"] or names (["/dev/video0", "/dev/video1"])
def init_cameras(devnames):
    cameras = []
    for i, devname in enumerate(devnames):
        print(f"Opening camera device {devname}")
        try:
            # Open device as int index
            c = cv.VideoCapture(int(devname))
        except ValueError:
            # Open device as string like "/dev/video0"
            c = cv.VideoCapture(devname)
        if not c.isOpened():
            print(f"Error: Could not open camera {devname}")
            for c in cameras:
                c.release()
            sys.exit(-1)

        # Test camera by taking a picture
        print(f"Testing camera {devname}")
        ret, pic = c.read()
        if not ret:
            print(f"Error: Could not read from camera {devname}")
            for c in cameras:
                c.release()
            sys.exit(-1)
        print(f"Got image of size {pic.shape}")

        # Add camera to list
        cameras.append(c)
        print(f"Initialized camera at index {i}: {devname}")
    return cameras
