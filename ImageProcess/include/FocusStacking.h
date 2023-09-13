/**
 * @author Avent
 * @version 0.1
 * @date 2023-04-25
 * @copyright Copyright (Avent) 2023
 **/
#pragma once
#include <opencv2/opencv.hpp>

class FocusStacking
{
public:
    FocusStacking() = default;

    void run(const std::vector<cv::Mat>& imgs_, cv::Mat& dst);
    void run(const std::string& dir_path, cv::Mat& dst);

    static cv::Mat doLap(const cv::Mat& img);

private:
    /** 初始化SIFT特征检测器 **/
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();

    std::vector<cv::Mat> alignImgs(const std::vector<cv::Mat>& imgs);   // 图像对齐
    void focusStack(const std::vector<cv::Mat>& aligned_imgs, cv::Mat& output);
};
