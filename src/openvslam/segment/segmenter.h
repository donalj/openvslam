#ifndef OPENVSLAM_SEGMENT_SEGMENTER_H
#define OPENVSLAM_SEGMENT_SEGMENTER_H

#include "openvslam/segment/mask_rcnn.h"
#include "openvslam/segment/deeplab.h"
#include <iostream>
#include <memory>


namespace openvslam {
namespace segment {

class segmenter {
public:
    segmenter();
    segmenter(std::string model);
    //! Destructor
    cv::Mat segment(const cv::Mat& input_image);
private:
    std::unique_ptr<segment::base_model> segmentation_model;
};
} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_SEGMENT_SEGMENTER_H