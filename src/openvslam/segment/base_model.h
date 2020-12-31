#ifndef OPENVSLAM_SEGMENT_BASE_MODEL_H
#define OPENVSLAM_SEGMENT_BASE_MODEL_H

#include "openvslam/pytorch/model.h"
#include "openvslam/pytorch/utils.h"


#include <iostream>
#include <memory>


namespace openvslam {
namespace segment {

class base_model {
public:
    base_model(pytorch::model model);
    //! Destructor
    virtual cv::Mat get_segmentation_mask(const cv::Mat& input_image) = 0;
private:
    pytorch::model _model;
};
} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_SEGMENT_BASE_MODEL_H