#include "FocusStacking.h"
#include "parseSetting.h"

int main()
{
    avent::FocusStacking focus_stacker;
    cv::Mat img;
    cv::imshow("Outcome", img);
    cv::waitKey(0);
    return 0;
}