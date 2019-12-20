import numpy as np
import cv2


def get_contours(image):
    image = cv2.Canny(image, 30, 270)
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    new_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            new_contours.append(contour)

    return np.asarray(new_contours)


def get_center(contourn):
    M = cv2.moments(contourn)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx-50, cy)


if __name__ == '__main__':
    shapes = cv2.imread('../images/geometricShapes.png')
    original_image = shapes.copy()
    shapes = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)

    contours = get_contours(shapes.copy())

    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        if len(approx) % 3 == 0:
            shape = "triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)

            if abs(w-h) <= 3:
                shape = "square"
            else:
                shape = "rectangle"
        elif len(approx) >= 11:
            shape = "circle"

        center = get_center(contour)

        cv2.putText(original_image, shape, center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.drawContours(original_image, [contour], -1, (0, 0, 0), 1)
        cv2.imshow('', original_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
