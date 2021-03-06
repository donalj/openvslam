#ifndef OPENVSLAM_SEGMENT_MASK_RCNN_H
#define OPENVSLAM_SEGMENT_MASK_RCNN_H

#include "openvslam/segment/base_model.h"
namespace openvslam {
namespace segment {

class mask_rcnn : public base_model{
public:
    mask_rcnn(pytorch::model model);
    //! Destructor
    cv::Mat get_segmentation_mask(const cv::Mat& input_image);
private:
    pytorch::model _model;
};
} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_SEGMENT_MASK_RCNN_H