#include "openvslam/segment/mask_rcnn.h"
#include <torchvision/PSROIAlign.h>
#include <torchvision/PSROIPool.h>
#include <torchvision/ROIAlign.h>
#include <torchvision/ROIPool.h>
#include <torchvision/empty_tensor_op.h>
#include <torchvision/nms.h>
#include <torchvision/DeformConv.h>

namespace openvslam {
namespace segment {


mask_rcnn::mask_rcnn() {
    _model = pytorch::model("/openvslam/models/maskrcnn_resnet50_fpn.pt", true);
}
mask_rcnn::mask_rcnn(std::string model_path, bool use_cuda) {
    _model = pytorch::model(model_path, use_cuda);
}

cv::Mat mask_rcnn::get_segmentation_mask(const cv::Mat& input_image) {
    auto device = _model.get_device();
    torch::Tensor tensor_image;
    auto image_placeholder = input_image.clone();
    auto image_dims = image_placeholder.size();
    pytorch::resize_input(image_placeholder, image_dims.height, image_dims.width);
    pytorch::mat_to_tensor(image_placeholder, tensor_image);
    c10::List<torch::Tensor> images = c10::List<torch::Tensor>({tensor_image.to(device)});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(images);
    auto out_tensor = _model.forward_pass(inputs);
    auto out1 = out_tensor.toTuple();
    auto dets = out1->elements().at(1).toList();
    auto det0 = dets.get(0).toGenericDict();
    torch::Tensor masks = det0.at("masks").toTensor().to(device);
    torch::Tensor labels = det0.at("labels").toTensor().to(device);
    torch::Tensor boxes = det0.at("boxes").toTensor().to(device);
    torch::Tensor all_masks = torch::zeros({1, image_dims.height, image_dims.width}).to(device);
    for (int i = 0; i < masks.sizes()[0] ; i++) {
        // if (labels[i].item<int>() == 3) {
        if (labels[i].item<int>() >-1000) {
            auto mask = masks.index({i});
            all_masks = all_masks + mask;
        }
    }
    all_masks = all_masks.permute({2, 1, 0});
    cv::Mat output_image;
    pytorch::tensor_to_mat(all_masks.to(torch::DeviceType::CPU), output_image);
    return output_image;
}

} // namespace segment
} // namespace openvslam
