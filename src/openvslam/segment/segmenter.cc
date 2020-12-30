#include "openvslam/segment/segmenter.h"

namespace openvslam {
namespace segment {

// static auto registry =
//         torch::RegisterOperators()
//                 .op("torchvision::nms", &nms)
//                 .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor",
//                     &roi_align)
//                 .op("torchvision::roi_pool", &roi_pool)
//                 .op("torchvision::_new_empty_tensor_op", &new_empty_tensor)
//                 .op("torchvision::ps_roi_align", &ps_roi_align)
//                 .op("torchvision::ps_roi_pool", &ps_roi_pool)
//                .op("torchvision::deform_conv2d", &deform_conv2d);
//             //    .op("torchvision::_cuda_version", &_cuda_version)

segmenter::segmenter(){
    load_model("./models/maskrcnn_resnet50_fpn.pt");
    device = torch::DeviceType::CUDA;
    module.to(device);
}

segmenter::segmenter(std::string model, bool use_cuda) {
    load_model(model);
    if (use_cuda) {
        device = torch::DeviceType::CUDA;
        module.to(device);
    }
}

void segmenter::load_model(std::string model) {
    try {
        torch::jit::getProfilingMode() = false;   
        torch::jit::getExecutorMode() = false;
        torch::jit::setGraphExecutorOptimize(false);   
        module = torch::jit::load(model);
        module.to(device);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
}

torch::jit::IValue segmenter::forward_pass(std::vector<torch::jit::IValue> input) {
    return module.forward(input);
}

void segmenter::resize_input(cv::Mat& input, int height, int width) {
    input.convertTo(input, CV_32F, 1.0 / 255);
    cv::resize(input, input, cv::Size(width, height), 0, 0, 1);
}

void segmenter::mat_to_tensor(const cv::Mat& input, torch::Tensor& output) {
    output = torch::from_blob(input.data, {input.rows, input.cols, 1}, torch::kF32);
    output = output.permute({2, 0, 1});
}

void segmenter::tensor_to_mat(const torch::Tensor& input, cv::Mat& output) {
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

cv::Mat segmenter::get_segmentation_mask(const cv::Mat& input_image) {
    torch::Tensor tensor_image;
    auto image_placeholder = input_image.clone();

    auto image_dims = image_placeholder.size();
    resize_input(image_placeholder, image_dims.height, image_dims.width);
    mat_to_tensor(image_placeholder, tensor_image);
    c10::List<torch::Tensor> images = c10::List<torch::Tensor>({tensor_image.to(device)});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(images);
    auto out_tensor = forward_pass(inputs);
    auto out1 = out_tensor.toTuple();
    auto dets = out1->elements().at(1).toList();
    auto det0 = dets.get(0).toGenericDict();
    torch::Tensor masks = det0.at("masks").toTensor().to(device);
    torch::Tensor labels = det0.at("labels").toTensor().to(device);
    torch::Tensor boxes = det0.at("boxes").toTensor().to(device);
    torch::Tensor all_masks = torch::zeros({1, image_dims.height, image_dims.width}).to(device);
    for (int i = 0; i < masks.sizes()[0] ; i++) {
        if (labels[i].item<int>() == 3) {
            auto mask = masks.index({i});
            all_masks = all_masks + mask;
        }
    }
    all_masks = all_masks.permute({2, 1, 0});
    cv::Mat output_image;
    tensor_to_mat(all_masks.to(torch::DeviceType::CPU), output_image);
    return output_image;
}

void segmenter::show_image(cv::Mat& img, std::string title) {
    cv::namedWindow(title, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}
} // namespace segment
} // namespace openvslam
