import cv2
import os
from argparse import ArgumentParser

def get_names(txt_file):
    names = []
    with open(txt_file, mode='r') as f:
        for line in f:
            names.append(line.strip())
    return names

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', type=str, default='LBPH',
                        help='face recognizer model name: LBPH, Eigen, Fisher')
    parser.add_argument('-md', '--model-directory', dest='model_dir', type=str, default='./models',
                        help='a directory where to load the trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    model = args.model
    model_dir = args.model_dir
    namesfile_path = os.path.join(model_dir, 'names.txt')
    model_path = os.path.join(model_dir, model+'.yml')

    names = get_names(namesfile_path)
    print("names: ",  str(names))

    face_detector = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    face_recognizer = None
    if model == 'Eigen':
        face_recognizer = cv2.face.createEigenFaceRecognizer()
    elif model == 'Fisher':
        face_recognizer = cv2.face.createFisherFaceRecognizer()
    else:
        face_recognizer = cv2.face.createLBPHFaceRecognizer()

    print('loading model in {0}'.format(model_path))
    face_recognizer.load(model_path)
    print('load model finished.')

    camera = cv2.VideoCapture(0)

    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,
                                                scaleFactor=1.3,
                                                minNeighbors=5,
                                                minSize=(200, 200),
                                                maxSize=(400, 400))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            roi = gray[y:y+h, x:x+w]
            if model != 'LBPH':
                roi = cv2.resize(roi, (360, 360))
            Id = -1
            Id = face_recognizer.predict(roi)
            print("id: {0}, name: {1}".format(Id, names[Id]))
            if Id < 0:
                cv2.putText(frame, 'Friend', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            else:
                cv2.putText(frame, names[Id], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)


        cv2.imshow('camera', frame)
        if cv2.waitKey(10)&0xff == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
