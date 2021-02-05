import seeta_facedet

class SeetaFaceDetect:

    def __init__(self, model_path):
        """
            Init detector

        :param
            model_path: str.
                The model path
        """
        self.__detector = seeta_facedet.PyFaceDetection(model_path)

    def detect(self, img):
        """
        Detect faces on input image.

        :param
            img: ndarray.
                The input image should be gray-scale

        :return: list[[x, y, width, height, roll, pitch, yaw, score]]. The face detection results
        """

        return self.__detector.Detect(img)

    def setMinFaceSize(self, min_face_fize=40):
        """
        Set the minimum size of faces to detect.

        :param
            min_face_fize: int.
                The minimum size is constrained as no smaller than 20

        :return: None
        """
        self.__detector.SetMinFaceSize(min_face_fize)

    def setMaxFaceSize(self, max_face_size=600):
        """
        Set the maximum size of faces to detect.

        :param
            max_face_size: int
                The maximum face size actually used is computed as the minimum among: user
                specified size, image width, image height.

        :return: None
        """

        self.__detector.SetMaxFaceSize(max_face_size)

    def setImagePyramidScaleFactor(self, image_pyramid_scale_factor=0.8):
        """
        Set the factor between adjacent scales of image pyramid.

        :param
            image_pyramid_scale_factor: float32
                The value of the factor lies in (0, 1). For example, when it is set as 0.5,
                an input image of size w x h will be resized to 0.5w x 0.5h, 0.25w x 0.25h,
                0.125w x 0.125h, etc.

        :return: None
        """

        self.__detector.SetImagePyramidScaleFactor(image_pyramid_scale_factor)

    def setWindowStep(self, step_x=4, step_y=4):
        """
        Set the sliding window step in horizontal and vertical directions.

        :param
            step_x: int
            step_y: int
                The steps should take positive values, and invalid ones will be ignored.
                Usually a step of 4 is a reasonable choice

        :return: None
        """

        self.__detector.SetWindowStep(step_x, step_y)

    def setScoreThresh(self, score_thresh):
        """
        Set the score thresh of detected faces.

        :param
            score_thresh: float32
                Detections with scores smaller than the threshold will not be returned.
                Typical threshold values include 0.95, 2.8, 4.5. One can adjust the
                threshold based on his or her own test set.

        :return: None
        """

        self.__detector.SetScoreThresh(score_thresh)
