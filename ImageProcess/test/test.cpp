#include "FocusStacking.h"
#include "Projecting.h"
// #include "parseSetting.h"

int main()
{
    // avent::FocusStacking focus_stacker;
    // cv::Mat img;
    // focus_stacker.fuse("../data/images/FocusStacking", img);
    // cv::imshow("Outcome", img);
    // cv::waitKey(0);
    cv::Mat img = cv::imread("../data/images/image.png");
    cv::Mat dst;
    avent::Projecting projecter("../data/config.yaml");
    projecter.locateHeap(img, dst);
    // projecter.locateTarget(img, dst);
    cv::imshow("Outcome", dst);
    cv::waitKey(0);

    return 0;
}