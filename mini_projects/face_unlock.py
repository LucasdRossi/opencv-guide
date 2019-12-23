'''
# to get the dataset out of your face
python3 face_unlock dataset
# to make recognition
python3 face_unlock recognize
'''
from sys import argv
import numpy as np
import cv2
import os


def dataset():
    cont = 0
    while True:
        _, img = cam.read()
        key = cv2.waitKey(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_face = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(detected_face) > 0:
            for x, y, w, h in detected_face:
                face = gray[y:y + h, x:x + w]
                cv2.resize(face, (200, 200), interpolation=cv2.INTER_LANCZOS4)
                file_name = dataset_path + str(cont) + ".jpg"
                cv2.imwrite(file_name, face)
                cont += 1
                print("{} photos".format(cont))

        # cv2.imshow('', face)
        if key == 27 or cont == 100:  # 'esc'
            break


def train():
    train_images, labels = [], []
    for i, image in enumerate(os.listdir(dataset_path)):
        if image[-4:] == ".jpg":
            face = cv2.imread(dataset_path+image, 0)
            train_images.append(np.asarray(face, dtype=np.uint8))
            labels.append(i)

    labels = np.asarray(labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(train_images, labels)
    print("model successfully loaded")
    return model


def recognize(model):
    while True:
        _, img = cam.read()
        key = cv2.waitKey(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_face = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(detected_face) > 0:
            for x, y, w, h in detected_face:
                cv2.rectangle(img, (x-10, y-10),
                              (x+w+10, y+h+10), (0, 0, 255), 2)
                face = gray[y:y + h, x:x + w]

                _, conf = model.predict(face)

                conf = int(100 * (1 - (conf)/400))

                text = "confidence: %.2f" % conf
                cv2.putText(img, text, (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if conf > 85:
                    cv2.putText(img, "unlocked", (500, 470),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "locked", (500, 470),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('', img)

        if key == 27:  # 'esc
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    dataset_path = "../files/face_dataset/"
    face_classifier = cv2.CascadeClassifier(
        '../files/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    arg = argv[1]
    if arg == "dataset":
        dataset()
    elif arg == "recognize":
        model = train()
        recognize(model)
    else:
        print("invalid argument")
