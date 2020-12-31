#ifndef OPENVSLAM_SEGMENT_MASK_RCNN_H
#define OPENVSLAM_SEGMENT_MASK_RCNN_H

#include "openvslam/pytorch/model.h"
#include "openvslam/pytorch/utils.h"
#include <iostream>
#include <memory>


namespace openvslam {
namespace segment {

class mask_rcnn {
public:
    mask_rcnn();
    mask_rcnn(std::string model, bool use_cuda);
    //! Destructor
    cv::Mat get_segmentation_mask(const cv::Mat& input_image);
private:
    pytorch::model _model;
};
} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_SEGMENT_MASK_RCNN_H