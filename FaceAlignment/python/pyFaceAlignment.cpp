
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>

#include "face_alignment.h"

namespace py = pybind11;

class PyFaceAlignment {

public:
    PyFaceAlignment(const std::string& model_path, int num_point = 5, int dim_point = 2):faceAlignment_(model_path.c_str())
    {
        num_point_ = num_point;
        dim_point_ = dim_point;
        points_.resize(num_point_);
        points_float_.resize(num_point_ * dim_point_);
    }

    ~PyFaceAlignment()
    {

    }

    std::vector<std::vector<float>> PointDetectLandmarks(const py::array_t<uint8_t>& img
                                                                , const std::vector<std::vector<float>>& faceinfos)
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

        std::vector<std::vector<float>> pointses;
        for (int idx = 0; idx < faceinfos.size(); ++idx)
        {
            seeta::FaceInfo  faceInfo;
            // x, y, width, height, roll, pitch, yaw, score
            faceInfo.bbox.x = faceinfos[idx][0];
            faceInfo.bbox.y = faceinfos[idx][1];
            faceInfo.bbox.width = faceinfos[idx][2];
            faceInfo.bbox.height = faceinfos[idx][3];
            faceInfo.roll = faceinfos[idx][4];
            faceInfo.pitch = faceinfos[idx][5];
            faceInfo.yaw = faceinfos[idx][6];
            faceInfo.score = faceinfos[idx][7];
            faceAlignment_.PointDetectLandmarks(img_data, faceInfo, points_.data());

            for (int idy = 0; idy < num_point_; ++idy)
            {
                points_float_[dim_point_ * idy + 0] = points_[idy].x;
                points_float_[dim_point_ * idy + 1] = points_[idy].y;
            }
            pointses.push_back(points_float_);
        }

        return pointses;
    }

private:
    int num_point_;
    int dim_point_;
    seeta::FaceAlignment faceAlignment_;
    std::vector<seeta::FacialLandmark> points_;
    std::vector<float> points_float_;
};


PYBIND11_MODULE(MODEL_NAME, m) {

m.doc() = "Face alignment python api";

//wrapper C++ PyFaceAlignment to python PyFaceAlignment
py::class_<PyFaceAlignment>(m, "PyFaceAlignment")
    .def(py::init<const std::string&, int ,int >())
    .def("PointDetectLandmarks", &PyFaceAlignment::PointDetectLandmarks);
}

