import numpy as np
import cv2
import time


def filter(image):
    # use median across all pixels in the kernel
    image = cv2.medianBlur(image, 7)
    image = cv2.Canny(image, 100, 200)
    return image


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)

    while True:
        start = time.time()

        _, frame = cam.read()
        key = cv2.waitKey(1)

        frame = filter(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame_time = (time.time() - start) * 1000
        fps = 1000 / frame_time

        cv2.putText(frame, str(int(fps)) + ' fps', (5, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        cv2.imshow('', frame)
        if key == 27:  # esc
            break

    cv2.destroyAllWindows()
