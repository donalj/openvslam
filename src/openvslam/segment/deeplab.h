#ifndef OPENVSLAM_SEGMENT_DEEPLAB_H
#define OPENVSLAM_SEGMENT_DEEPLAB_H

#include "openvslam/segment/base_model.h"

namespace openvslam {
namespace segment {

class deeplab : public base_model{
public:
    deeplab(pytorch::model model);
    //! Destructor
    cv::Mat get_segmentation_mask(const cv::Mat& input_image);
private:
    pytorch::model _model;
};
} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_SEGMENT_DEEPLAB_H