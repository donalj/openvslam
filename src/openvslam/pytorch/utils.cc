#include "openvslam/pytorch/utils.h"
namespace openvslam {
namespace pytorch {

void resize_input(cv::Mat& input, int height, int width) {
    input.convertTo(input, CV_32F, 1.0 / 255);
    cv::resize(input, input, cv::Size(width, height), 0, 0, 1);
}

void mat_to_tensor(const cv::Mat& input, torch::Tensor& output) {
    output = torch::from_blob(input.data, {input.rows, input.cols, 1}, torch::kF32);
    output = output.permute({2, 0, 1});
}

void tensor_to_mat(const torch::Tensor& input, cv::Mat& output) {
    int width = input.sizes()[0];
    int height = input.sizes()[1];
    try {
        output = cv::Mat(height, width, CV_32F, input.data_ptr());
    }
    catch (const c10::Error& e) {
        std::cout << "an error has occured : " << e.msg() << std::endl;
        output = cv::Mat(height, width, CV_32F);
    }
    cv::subtract(cv::Mat::ones(height, width, CV_32F), output, output);
    output.convertTo(output, CV_8UC1, 255);

}


void show_image(cv::Mat& img, std::string title) {
    cv::namedWindow(title, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}
} // namespace segment
} // namespace openvslam
