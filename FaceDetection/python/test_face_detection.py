import argparse
import cv2
import face_detection

parser = argparse.ArgumentParser(description='face detect python api test')
# general
parser.add_argument('--model_path',
                    type=str,
                    default='',
                    required=True,
                    help='Seeta face detection model path')

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

detector = face_detection.SeetaFaceDetect(args.model_path)
print(help(detector))

detector.setMinFaceSize(40)
detector.setScoreThresh(2.0)
detector.setImagePyramidScaleFactor(0.8)
detector.setWindowStep(4, 4)


img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Open image " + args.image_path + " failed!")
    exit(0)
img_gray = img.copy()
if 2 != len(img_gray.shape):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

t0 = cv2.getTickCount()
faces = detector.detect(img_gray)
t1 = cv2.getTickCount()

secs = (t1 - t0)/cv2.getTickFrequency()

print("Detections takes ", secs, " seconds")

print("Image size (hxw): ", img_gray.shape)
print("Detected face num: ", len(faces))

for rect in faces:
    print(rect)
    x = int(rect[0])
    y = int(rect[1])
    width = int(rect[2])
    height = int(rect[3])

    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 125, 100), 2, cv2.FONT_HERSHEY_PLAIN)
    cv2.imwrite(args.save_path, img)

print("Saved result to path: ", args.save_path)

