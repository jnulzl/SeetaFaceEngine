import sys
sys.path.insert(0, "./build/bin/")
sys.path.insert(1, "./FaceAlignment/python/")
sys.path.insert(2, "./FaceDetection/python/")
sys.path.insert(3, "./FaceIdentification/python/")
# print(sys.path)

import argparse
import cv2
import numpy as np
import face_detection
import face_alignment
import face_identification


parser = argparse.ArgumentParser(description='face detect python api test')
# general
parser.add_argument('--detect_model_path',
                    type=str,
                    default='./FaceDetection/model/seeta_fd_frontal_v1.0.bin',
                    help='Seeta face detection model path')

parser.add_argument('--align_model_path',
                    type=str,
                    default='./FaceAlignment/model/seeta_fa_v1.1.bin',
                    help='Seeta face alignment model path')

parser.add_argument('--identi_model_path',
                    type=str,
                    default='./FaceIdentification/model/seeta_fr_v1.0.bin',
                    help='Seeta face identification model path')

parser.add_argument('--image_path',
                    type=str,
                    default='./FaceDetection/data/0_1_1.jpg',
                    help='Test image path')

parser.add_argument('--save_path',
                    type=str,
                    default='./test_res.jpg',
                    help='Save image path')

parser.add_argument('--min_face_size',
                    type=int,
                    default=40,
                    help='The minimum size of faces to detect')

parser.add_argument('--max_face_size',
                    type=int,
                    default=600,
                    help='The maximum size of faces to detect')

parser.add_argument('--image_pyramid_scale_factor',
                    type=float,
                    default=0.8,
                    help='The factor between adjacent scales of image pyramid')

parser.add_argument('--window_step',
                    type=int,
                    default=4,
                    help='The sliding window step in horizontal and vertical directions')

parser.add_argument('--score_thresh',
                    type=float,
                    default=2.0,
                    help='The score thresh of detected faces')

args = parser.parse_args()

detector = face_detection.SeetaFaceDetect(args.detect_model_path)
detector.setMinFaceSize(args.min_face_size)
detector.setScoreThresh(args.score_thresh)
detector.setImagePyramidScaleFactor(args.image_pyramid_scale_factor)
detector.setWindowStep(args.window_step, args.window_step)

faceAlig = face_alignment.SeetaFaceAlignment(args.align_model_path)
faceIdenti= face_identification.SeetaFaceIdentification(args.identi_model_path)

img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Open image " + args.image_path + " failed!")
    exit(0)
img_gray = img.copy()
if 2 != len(img_gray.shape):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


t0 = cv2.getTickCount()
faces = detector.detect(img_gray)
facelandmarks = faceAlig.pointDetectLandmarks(img_gray, faces)
crop_imgs = faceIdenti.cropFace(img, facelandmarks)
gallery_fea = faceIdenti.extractFeatureWithCrop(img, facelandmarks)
probe_fea = faceIdenti.extractFeatureWithCrop(img, facelandmarks)
t1 = cv2.getTickCount()

secs = (t1 - t0)/cv2.getTickFrequency()

print("Taking time: ", secs, " seconds")

print("Image size (hxw): ", img.shape)

print("Detected face num: ", len(faces))

for index in range(len(faces)):
    cv2.imwrite(args.save_path + "_crop_%d.jpg"%(index), crop_imgs[index])
    rect = faces[index]
    print("rect[%d]" % (index),": ", rect)
    x = int(rect[0])
    y = int(rect[1])
    width = int(rect[2])
    height = int(rect[3])
    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 125, 100), 2, cv2.FONT_HERSHEY_PLAIN)

    points = np.array(facelandmarks[index]).reshape(-1, 2).astype(np.int)
    print("facelandmarks[%d]" % index, ": ", points)
    for point in points:
        cv2.circle(img, (point[0],point[1]), 2, (0, 0, 255), cv2.FILLED)

cv2.imwrite(args.save_path, img)
print("Saved result to path: ", args.save_path)
print("Sim is: ", faceIdenti.calcSimilarity(gallery_fea, probe_fea))

