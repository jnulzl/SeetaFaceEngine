import argparse
import cv2
import numpy as np
import face_identification

parser = argparse.ArgumentParser(description='face identification python api test')
# general
parser.add_argument('--model_path',
                    type=str,
                    default='',
                    required=True,
                    help='Seeta face identification model path')

parser.add_argument('--image_path',
                    type=str,
                    default='',
                    required=True,
                    help='Test image path')

args = parser.parse_args()

faceIdentification = face_identification.SeetaFaceIdentification(args.model_path)

img = cv2.imread(args.image_path, 1)
if img is None:
    print("Open image " + args.image_path + " failed!")
    exit(0)

# [x, y, width, height, roll, pitch, yaw, score]
# box = [158.0, 216.0, 224.0, 224.0, 0.0, 0.0, 0.0, 24.380210876464844]
facelandmarks = [[218.02700805664062, 320.05078125, 301.3433532714844, 300.4508972167969, 271.5216979980469, 375.4873046875, 253.0943145751953, 409.58197021484375, 314.4736328125, 393.3882141113281]]

t0 = cv2.getTickCount()
gallery_fea = faceIdentification.extractFeatureWithCrop(img, facelandmarks)
probe_fea = faceIdentification.extractFeatureWithCrop(img, facelandmarks)
t1 = cv2.getTickCount()
secs = (t1 - t0)/cv2.getTickFrequency() / 2
print("Detections takes ", secs, " seconds")

print("Sim is : ", faceIdentification.calcSimilarity(gallery_fea, probe_fea))

