import viplnet


class SeetaFaceIdentification:

    def __init__(self, model_path = None):
        """
            Init detector

        :param
            model_path: str.
                The model path
        """
        self.__faceIdentification = viplnet.PyFaceIdentification(model_path)

    def LoadModel(self, model_path):
        """
        Load model of face feature extract

        :param
            model_path: str.
                The model of face feature extract

        :return:
        """

        return self.__faceIdentification.LoadModel(model_path)

    def feature_size(self):
        """
        Get feature's dimension.

        :return: None
        """
        return self.__faceIdentification.feature_size()

    def crop_width(self):
        """
        Get cropping face width.
        :return: None
        """

        return self.__faceIdentification.crop_width()

    def crop_height(self):
        """
        Get cropping face height.

        :return: None
        """

        return self.__faceIdentification.crop_height()

    def crop_channels(self):
        """
        Get cropping face channels.

        :return: None
        """

        return self.__faceIdentification.SetWindowStep()

    def cropFace(self, src_image, facelandmarks):
        """
          Crop face with 3-channels image and 5 located landmark points.

        :param
            src_image: ndarray
                The input image should be color-scale

            facelandmarks: list[[x,y,x,y,....]]
                The face alignment results

        :return: list[ndarray]. Cropped image
        """

        return self.__faceIdentification.CropFace(src_image, facelandmarks)

    def extractFeature(self, crop_images):
        """
          Extract feature with a cropping faces.

        :param
            crop_images: list[ndarray]
                Cropped image

        :return: list[[x1,x2,...]]. Face features
        """

        return self.__faceIdentification.ExtractFeature(crop_images)

    def extractFeatureWithCrop(self, src_image, facelandmarks):
        """
          Extract feature for face in a 3-channels image given 5 located landmark points.

        :param
            src_image: ndarray
                The input image should be color-scale

            facelandmarks: list[[x,y,x,y,....]]
                The face alignment results

        :return: list[[x1,x2,...]]. Face features
        """

        return self.__faceIdentification.ExtractFeatureWithCrop(src_image, facelandmarks)

    def calcSimilarity(self, fc1, fc2):
        """
         Calculate similarity of face features fc1 and fc2.
         dim = -1 default feature size

        :param
            fc1: list[[x1,x2,...]].
                Face features.

            fc1: list[[x1,x2,...]].
                Face features.

        :return: list[sim1, sim2, ...]
        """

        return self.__faceIdentification.CalcSimilarity(fc1, fc2)
