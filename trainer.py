import cv2
import os
import numpy as np
from argparse import ArgumentParser

def read_images(data_dir, resize=False):
    images = []
    labels = []
    names = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for i, subdirname in enumerate(dirnames):
            names.append(subdirname)
            subject_path = os.path.join(dirpath, subdirname)
            for filename in os.listdir(subject_path):
                if filename.split('.')[-1] != 'pgm':
                    continue
                file_path = os.path.join(subject_path, filename)
                print('reading image: {0}'.format(file_path))
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if resize:
                    img = cv2.resize(img, (360, 360))
                images.append(img)
                labels.append(i)
    # images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.int32)
    return images, labels, names

def save_names(names, out_dir):
    namesfile_path = os.path.join(out_dir, 'names.txt')
    with open(namesfile_path, mode='w') as f:
        for name in names:
            f.write(name + '\n')

def main():
    parser = ArgumentParser()
    parser.add_argument('-dd', '--data-directory', dest='data_dir', type=str, default='./data',
                        help='location root of your data ')
    parser.add_argument('-o', '--out-directory', dest='out_dir', type=str, default='./models',
                        help='a directory where to save the trained model')
    parser.add_argument('-m', '--model', dest='model', type=str, default='LBPH',
                        help='face recognizer model name: LBPH, Eigen, Fisher')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    model = args.model
    resize = True

    if model == 'LBPH':
        face_recognizer = cv2.face.createLBPHFaceRecognizer()
        resize = False
    elif model == 'Eigen':
        face_recognizer = cv2.face.createEigenFaceRecognizer()
    elif model =='Fisher':
        face_recognizer = cv2.face.createFisherFaceRecognizer()
    else:
        raise Exception('model name error!')

    print('start reading images in {0}'.format(data_dir))
    images, labels, names = read_images(data_dir, resize=resize)

    save_names(names, out_dir)

    print('start training face recognizer')
    face_recognizer.train(images, labels)

    model_path = os.path.join(out_dir, model + '.yml')# LBPH.yml, Eigen.yml, Fisher.yml
    print('save trained model to {0}'.format(model_path))
    face_recognizer.save(model_path)

if __name__ == '__main__':
    main()
