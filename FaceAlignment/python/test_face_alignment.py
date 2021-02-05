import argparse
import cv2
import numpy as np
import face_alignment

parser = argparse.ArgumentParser(description='face alignment python api test')
# general
parser.add_argument('--model_path',
                    type=str,
                    default='',
                    required=True,
                    help='Seeta face alignment model path')

parser.add_argument('--image_path',
                    type=str,
                    default='',
                    required=True,
                    help='Test image path')

parser.add_argument('--save_path',
                    type=str,
                    default='',
                    required=True,
                    help='Save image path')

args = parser.parse_args()

faceAlignment = face_alignment.SeetaFaceAlignment(args.model_path)

img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Open image " + args.image_path + " failed!")
    exit(0)
img_gray = img.copy()
if 2 != len(img_gray.shape):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

# [x, y, width, height, roll, pitch, yaw, score]
box = [158.0, 216.0, 224.0, 224.0, 0.0, 0.0, 0.0, 24.380210876464844]

t0 = cv2.getTickCount()
pointses = faceAlignment.pointDetectLandmarks(img_gray, [box])
t1 = cv2.getTickCount()

secs = (t1 - t0)/cv2.getTickFrequency()

print("Detections takes ", secs, " seconds")
print("Image size (hxw): ", img_gray.shape)
print("Aligned face num: ", len(pointses))
print("Points: ", pointses)
for points_tmp in pointses:
    points = np.array(points_tmp).reshape(-1, 2).astype(np.int)
    for point in points:
        cv2.circle(img, (point[0],point[1]), 2, (0, 0, 255), cv2.FILLED)

cv2.imwrite(args.save_path, img)
print("Saved result to path: ", args.save_path)

