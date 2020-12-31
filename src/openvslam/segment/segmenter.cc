#include "openvslam/segment/segmenter.h"


namespace openvslam {
namespace segment {

segmenter::segmenter(){
    pytorch::model mask_rcnn_pt = pytorch::model("/openvslam/models/maskrcnn_resnet50_fpn.pt", true);
    segmentation_model = std::make_unique<segment::mask_rcnn>(mask_rcnn_pt);
}
segmenter::segmenter(std::string model){
    if (model == "mask_rcnn"){
        pytorch::model mask_rcnn_pt = pytorch::model("/openvslam/models/maskrcnn_resnet50_fpn.pt", true);
        segmentation_model = std::make_unique<segment::mask_rcnn>(mask_rcnn_pt);
    }
    else if (model == "deeplab"){
        pytorch::model deeplab_pt = pytorch::model("/openvslam/models/deeplabv3_resnet50.pt", true);
        segmentation_model = std::make_unique<segment::deeplab>(deeplab_pt);

    }
    else {
        std::cout << model << " is not currently supported" << std::endl;
    }
}
//! Destructor
cv::Mat segmenter::segment(const cv::Mat& input_image){
    return segmentation_model->get_segmentation_mask(input_image);
    }
}// namespace segment
} // namespace openvslam

