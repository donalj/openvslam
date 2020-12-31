#include "openvslam/pytorch/model.h"

namespace openvslam {
namespace pytorch {

model::model(){
    load_model("/openvslam/models/maskrcnn_resnet50_fpn.pt");
    device = torch::DeviceType::CUDA;
    module.to(device);
}

model::model(std::string model, bool use_cuda) {
    load_model(model);
    if (use_cuda) {
        device = torch::DeviceType::CUDA;
        module.to(device);
    }
}

void model::load_model(std::string model) {
    try {
        torch::jit::getProfilingMode() = false;   
        torch::jit::getExecutorMode() = false;
        torch::jit::setGraphExecutorOptimize(false);   
        module = torch::jit::load(model);
        module.to(device);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cerr << e.msg();
        std::cerr << "\n\n\n\n";
    }
}

torch::jit::IValue model::forward_pass(std::vector<torch::jit::IValue> input) {
    return module.forward(input);
}

torch::DeviceType model::get_device(){
    return device;
}
} // namespace segment
} // namespace openvslam
