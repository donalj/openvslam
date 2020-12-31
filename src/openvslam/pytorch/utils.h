#ifndef OPENVSLAM_PYTORCH_UTILS_H
#define OPENVSLAM_PYTORCH_UTILS_H
#include <torch/script.h> // One-stop header.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace openvslam {
namespace pytorch {

void resize_input(cv::Mat& input, int height, int width);
void mat_to_tensor(const cv::Mat& input, torch::Tensor& output);

void tensor_to_mat(const torch::Tensor& input, cv::Mat& output);

void show_image(cv::Mat& img, std::string title);
} // namespace segment
} // namespace openvslam

#endif // OPENVSLAM_PYTORCH_UTILS_H