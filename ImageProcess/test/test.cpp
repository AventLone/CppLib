#include "FocusStacking.h"

int main()
{
    avent::FocusStacking focus_stacker;
    cv::Mat img;
    focus_stacker.fuse("../images", img);
    cv::imshow("Outcome", img);
    cv::waitKey(0);
    return 0;
}