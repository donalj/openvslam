#include "openvslam/type.h"
#include "openvslam/segment/segmenter.h"
#include <gtest/gtest.h>
#include <ctime>

using namespace openvslam;

TEST(load_model, load_existing_model) {
    std::clock_t start;
    double duration;
    auto img = cv::imread("/mnt/c/Users/donal-new/OneDriveQUB/PhD/repos/openvslam/image.png", 0);
    auto model_path = "/mnt/c/Users/donal-new/OneDriveQUB/PhD/repos/openvslam/src/openvslam/segment/models/maskrcnn_resnet50_fpn.pt";
    auto use_cuda = true;
    segment::segmenter seg(model_path, use_cuda);
    start = std::clock();
    auto output = seg.get_segmentation_mask(img);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<< duration <<'\n';
    auto final = img.clone();
    cv::subtract(img, output, final, cv::Mat(), CV_8U);
    seg.show_image(final, "output");

}


