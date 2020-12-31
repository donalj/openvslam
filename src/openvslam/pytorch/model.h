#ifndef OPENVSLAM_PYTORCH_MODEL_H
#define OPENVSLAM_PYTORCH_MODEL_H

#include <torch/script.h> // One-stop header.
// #include <torchvision/vision.h>

#include <iostream>
#include <memory>


namespace openvslam {
namespace pytorch {

class model {
public:
    model();
    model(std::string model, bool use_cuda);
    virtual ~model() = default;
    torch::DeviceType get_device();
    torch::jit::IValue forward_pass(std::vector<torch::jit::IValue> input) ;
private:
    torch::jit::script::Module module;
    torch::DeviceType device = torch::DeviceType::CPU;
    void load_model(std::string model_path);
};
} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_SEGMENT_model_H