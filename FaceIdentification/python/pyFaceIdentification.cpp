//
// Created by lizhaoliang-os on 2021/2/4.
//


#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>

#include "face_identification.h"

namespace py = pybind11;

class PyFaceIdentification {

public:
    PyFaceIdentification(const std::string& model_path = nullptr):faceIdentification_(model_path.c_str())
    {
        int num_point = 5;
        points_.resize(num_point);
    }

    ~PyFaceIdentification()
    {

    }

    uint32_t LoadModel(const std::string& model_path)
    {
        return faceIdentification_.LoadModel(model_path.c_str());
    }

    uint32_t feature_size()
    {
        return faceIdentification_.feature_size();
    }

    uint32_t crop_width()
    {
        return faceIdentification_.crop_width();
    }

    uint32_t crop_height()
    {
        return faceIdentification_.crop_height();
    }

    uint32_t crop_channels()
    {
        return faceIdentification_.crop_channels();
    }

    std::vector<py::array_t<uint8_t>>  CropFace(const py::array_t<uint8_t>& src_image,
                      const std::vector<std::vector<float>>& facelandmarks)
    {
        py::buffer_info img_buf = src_image.request();
        if(3 != img_buf.ndim || 3 != img_buf.shape[2])
        {
            throw std::runtime_error("Input image must have three channels!");
        }

        seeta::ImageData img_data;
        img_data.data = reinterpret_cast<uint8_t*>(img_buf.ptr);
        img_data.width = img_buf.shape[1];
        img_data.height = img_buf.shape[0];
        img_data.num_channels = 3;

        std::vector<py::array_t<uint8_t>> desImgs;

        int num_points = points_.size();
        for (int idx = 0; idx < facelandmarks.size(); ++idx)
        {
            int dim_point = facelandmarks[idx].size() / num_points;
            for (int idy = 0; idy < num_points; ++idy)
            {
                points_[idy].x = facelandmarks[idx][dim_point * idy + 0];
                points_[idy].y = facelandmarks[idx][dim_point * idy + 1];
            }


            seeta::ImageData des_img;

            std::vector<uint8_t> des_img_data;
            des_img_data.resize(faceIdentification_.crop_channels() *
                                    faceIdentification_.crop_height() *
                                    faceIdentification_.crop_width());
            des_img.data = des_img_data.data();
            des_img.width = faceIdentification_.crop_width();
            des_img.height = faceIdentification_.crop_height();
            des_img.num_channels = faceIdentification_.crop_channels();

            faceIdentification_.CropFace(img_data, points_.data(), des_img);

            py::array_t<uint8_t> des_tmp(des_img_data.size(), des_img_data.data());
            des_tmp.resize({des_img.height, des_img.width, des_img.num_channels});

            desImgs.push_back(des_tmp);
        }

        return desImgs;
    }

    std::vector<std::vector<float>>  ExtractFeature(const std::vector<py::array_t<uint8_t>>& crop_images)
    {
        std::vector<std::vector<float>> faceFeatures;

        for (int idx = 0; idx < crop_images.size(); ++idx)
        {
            py::buffer_info img_buf = crop_images[idx].request();
            if(3 != img_buf.ndim || 3 != img_buf.shape[2])
            {
                throw std::runtime_error("Input image must have three channels!");
            }

            seeta::ImageData img_data;
            img_data.data = reinterpret_cast<uint8_t*>(img_buf.ptr);
            img_data.width = faceIdentification_.crop_width();
            img_data.height = faceIdentification_.crop_height();
            img_data.num_channels = faceIdentification_.crop_channels();

            std::vector<float> face_features(faceIdentification_.feature_size(),-1.0f);
            faceIdentification_.ExtractFeature(img_data, face_features.data());
            faceFeatures.push_back(face_features);
        }

        return faceFeatures;
    }

    std::vector<std::vector<float>>  ExtractFeatureWithCrop(const py::array_t<uint8_t>& src_image,
                                               const std::vector<std::vector<float>>& facelandmarks)
    {
        py::buffer_info img_buf = src_image.request();
        if(3 != img_buf.ndim || 3 != img_buf.shape[2])
        {
            throw std::runtime_error("Input image must have three channels!");
        }

        seeta::ImageData img_data;
        img_data.data = reinterpret_cast<uint8_t*>(img_buf.ptr);
        img_data.width = img_buf.shape[1];
        img_data.height = img_buf.shape[0];
        img_data.num_channels = 3;

        std::vector<std::vector<float>> faceFeatures;

        int num_points = points_.size();
        for (int idx = 0; idx < facelandmarks.size(); ++idx)
        {
            int dim_point = facelandmarks[idx].size() / num_points;
            for (int idy = 0; idy < num_points; ++idy)
            {
                points_[idy].x = facelandmarks[idx][dim_point * idy + 0];
                points_[idy].y = facelandmarks[idx][dim_point * idy + 1];
            }

            std::vector<float> face_feature(faceIdentification_.feature_size(),-1.0f);
            faceIdentification_.ExtractFeatureWithCrop(img_data, points_.data(), face_feature.data());

            faceFeatures.push_back(face_feature);
        }

        return faceFeatures;
    }

    std::vector<float>  CalcSimilarity(const std::vector<std::vector<float>>& fc1,
                                       const std::vector<std::vector<float>>& fc2)
    {
        std::vector<float> similarities;

        if(fc1.size() != fc2.size())
        {
            throw std::runtime_error("The num of fc1 must be equal fc2's");
        }

        for (int idx = 0; idx < fc1.size(); ++idx)
        {
            similarities.push_back(faceIdentification_.CalcSimilarity(const_cast<float* const>(fc1[idx].data()),
                                                                      const_cast<float* const>(fc2[idx].data()),
                                                                      fc1[idx].size()));
        }

        return similarities;
    }

private:
    seeta::FaceIdentification faceIdentification_;
    std::vector<seeta::FacialLandmark> points_;
};


PYBIND11_MODULE(MODEL_NAME, m) {

m.doc() = "Face identification python api";

//wrapper C++ PyFaceIdentification to python PyFaceIdentification
py::class_<PyFaceIdentification>(m, "PyFaceIdentification")
    .def(py::init<const std::string&>())
    .def("LoadModel", &PyFaceIdentification::LoadModel)
    .def("feature_size", &PyFaceIdentification::feature_size)
    .def("crop_width", &PyFaceIdentification::crop_width)
    .def("crop_height", &PyFaceIdentification::crop_height)
    .def("crop_channels", &PyFaceIdentification::crop_channels)
    .def("CropFace", &PyFaceIdentification::CropFace)
    .def("ExtractFeature", &PyFaceIdentification::ExtractFeature)
    .def("ExtractFeatureWithCrop", &PyFaceIdentification::ExtractFeatureWithCrop)
    .def("CalcSimilarity", &PyFaceIdentification::CalcSimilarity);
}

