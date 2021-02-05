
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>

#include "face_detection.h"

namespace py = pybind11;

class PyFaceDetection {
public:
    PyFaceDetection(const std::string& model_path):detector_(model_path.c_str())
    {
        // x, y, width, height, roll, pitch, yaw, score
        face_info_.resize(8);
    }

    ~PyFaceDetection()
    {

    }

    std::vector<std::vector<float>> Detect(const py::array_t<uint8_t>& img)
    {
        py::buffer_info img_buf = img.request();
        if(2 != img_buf.ndim)
        {
            throw std::runtime_error("Input image must be gray image!");
        }

        seeta::ImageData img_data;
        img_data.data = reinterpret_cast<uint8_t*>(img_buf.ptr);
        img_data.width = img_buf.shape[1];
        img_data.height = img_buf.shape[0];
        img_data.num_channels = 1;

        auto faceInfos = detector_.Detect(img_data);

        std::vector<std::vector<float>> face_infos_res;
        for (int idx = 0; idx < faceInfos.size(); ++idx)
        {
            // x, y, width, height, roll, pitch, yaw, score
            face_info_[0] = faceInfos[idx].bbox.x;
            face_info_[1] = faceInfos[idx].bbox.y;
            face_info_[2] = faceInfos[idx].bbox.width;
            face_info_[3] = faceInfos[idx].bbox.height;
            face_info_[4] = faceInfos[idx].roll;
            face_info_[5] = faceInfos[idx].pitch;
            face_info_[6] = faceInfos[idx].yaw;
            face_info_[7] = faceInfos[idx].score;
            face_infos_res.push_back(face_info_);
        }

        return face_infos_res;
    }

    void SetMinFaceSize(int32_t size)
    {
        detector_.SetMinFaceSize(size);
    }

    void SetMaxFaceSize(int32_t size)
    {
        detector_.SetMaxFaceSize(size);
    }

    void SetImagePyramidScaleFactor(float factor)
    {
        detector_.SetImagePyramidScaleFactor(factor);
    }

    void SetWindowStep(int32_t step_x, int32_t step_y)
    {
        detector_.SetWindowStep(step_x, step_y);
    }

    void SetScoreThresh(float thresh)
    {
        detector_.SetScoreThresh(thresh);
    }

private:
    seeta::FaceDetection detector_;
    std::vector<float> face_info_;
};


PYBIND11_MODULE(MODEL_NAME, m) {

m.doc() = "Face detection python api";

//wrapper C++ PyFaceDetection to python PyFaceDetection
py::class_<PyFaceDetection>(m, "PyFaceDetection")
    .def(py::init<const std::string&>())
    .def("Detect", &PyFaceDetection::Detect)
    .def("SetMinFaceSize", &PyFaceDetection::SetMinFaceSize)
    .def("SetMaxFaceSize", &PyFaceDetection::SetMaxFaceSize)
    .def("SetImagePyramidScaleFactor", &PyFaceDetection::SetImagePyramidScaleFactor)
    .def("SetWindowStep", &PyFaceDetection::SetWindowStep)
    .def("SetScoreThresh", &PyFaceDetection::SetScoreThresh);

}

