#ifndef OPENVSLAM_SEGMENT_SEGMENTER_H
#define OPENVSLAM_SEGMENT_SEGMENTER_H


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// #include <torch/torch.h>
#include <torch/script.h> // One-stop header.
// #include <torchvision/vision.h>
// #include <torchvision/PSROIAlign.h>
// #include <torchvision/PSROIPool.h>
// #include <torchvision/ROIAlign.h>
// #include <torchvision/ROIPool.h>
// #include <torchvision/empty_tensor_op.h>
// #include <torchvision/nms.h>
// #include <torchvision/DeformConv.h>

#include <iostream>
#include <memory>


namespace openvslam {
namespace segment {

class segmenter {
public:
    segmenter();
    segmenter(std::string model, bool use_cuda);
    //! Destructor
    virtual ~segmenter() = default;
    cv::Mat get_segmentation_mask(const cv::Mat& input_image);
    void show_image(cv::Mat& img, std::string title);
private:
    torch::jit::script::Module module;
    torch::DeviceType device = torch::DeviceType::CPU;
    void load_model(std::string model_path);
    void resize_input(cv::Mat& input, int height, int width);
    void mat_to_tensor(const cv::Mat& input, torch::Tensor& output);
    void tensor_to_mat(const torch::Tensor& input, cv::Mat& output);
    torch::jit::IValue forward_pass(std::vector<torch::jit::IValue> input) ;
};
} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_SEGMENT_SEGMENTER_H