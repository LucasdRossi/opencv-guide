from random import randint
from time import sleep
import numpy as np
import dlib
import cv2

if __name__ == '__main__':
    original_image = cv2.imread('../images/person.jpg')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '../files/shape_predictor_68_face_landmarks.dat')
    mouth_points = list(range(48, 61))
    alpha = 0.7


    while True:
        image = original_image.copy()
        mask = np.zeros(image.shape, np.uint8)
        faces = detector(image, 1)
        color = (randint(0, 256), randint(0, 256), randint(0, 256))
        key = cv2.waitKey(1)

        for face in faces:
            landmarks = predictor(image, face)
            mouth = []
            for i, points in enumerate(landmarks.parts()):
                if i in mouth_points:
                    points = (points.x, points.y)
                    mouth.append(points)

            cv2.fillPoly(
                mask, [np.asarray(mouth, dtype=np.int32)], (255, 255, 255))
            mask = cv2.bilateralFilter(mask, 7, 100, 100)

            color_mask = np.zeros(image.shape, np.uint8)
            color_mask[:, :] = color
            color_mask = cv2.addWeighted(
                color_mask, alpha, image, 1 - alpha, 0)

            image = np.where(mask, color_mask, image).astype(np.uint8)

        cv2.imshow('', image)

        if key == 27:  # 'esc'
            break
        
        # sleep(1)
    cv2.destroyAllWindows()
