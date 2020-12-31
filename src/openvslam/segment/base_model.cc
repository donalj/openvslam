#include "openvslam/segment/base_model.h"
#include <torchvision/PSROIAlign.h>
#include <torchvision/PSROIPool.h>
#include <torchvision/ROIAlign.h>
#include <torchvision/ROIPool.h>
#include <torchvision/empty_tensor_op.h>
#include <torchvision/nms.h>
#include <torchvision/DeformConv.h>


namespace openvslam {
namespace segment {


base_model::base_model(pytorch::model model) {
    _model = model;
}

} // namespace segment
} // namespace openvslam
