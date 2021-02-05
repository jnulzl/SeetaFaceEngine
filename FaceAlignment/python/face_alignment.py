import seeta_fa

class SeetaFaceAlignment:

    def __init__(self, model_path, num_point=5, dim_point=2):
        """
            Init detector

        :param
            model_path: str.
                The model path
            num_point: int.
                The number of points. Default value : 5
            dim_point: int.
                The dim of each point. Default value : 2
        """
        self.__point_detector = seeta_fa.PyFaceAlignment(model_path, num_point, dim_point)

    def pointDetectLandmarks(self, img, bboxs):
        """
        Detect five facial landmarks, i.e., two eye centers, nose tip and two mouth corners.

        :param
            img: ndarray.
                The input image should be gray-scale
            bboxs: list[[x, y, width, height, roll, pitch, yaw, score]]
                The face detection results

        :return: list[[x,y,x,y,....]]. The face alignment results
        """

        return self.__point_detector.PointDetectLandmarks(img, bboxs)
