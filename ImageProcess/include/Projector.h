/**********************************************************
 * @author AventLone
 * @version 0.2
 * @date 2023-04-25
 * @copyright Copyright (AventLone) 2023
 **********************************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_set>

class Projector
{
public:
    Projector(const std::string& setting_file, const cv::Point2f& translation, float depth);

    ~Projector() = default;

    std::vector<cv::Point2f> detect(const cv::Mat& src) const;

    std::unordered_set<uint8_t> locateHeap(const cv::Mat& src) const;

    /*** For Visualization ***/
    void detect(const cv::Mat& src, cv::Mat& dst) const;

    void locateHeap(const cv::Mat& src, cv::Mat& dst) const;

private:
    /*** Parameters ***/
    // const int threashold_1, threashold_2, threashold_3;
    // const uint16_t img_width = 1600, img_height = 1200;
    // const uint16_t circle_radius{350};
    // const cv::Point circle_center{img_width / 2 - 1, img_height / 2 - 1};
    // const cv::Point line1[2] = {cv::Point(img_width / 2 - 1, 0),
    //                             cv::Point(img_width / 2 - 1, img_height / 2 - circle_radius - 1)};
    // const cv::Point line2[2] = {cv::Point(0, img_height / 2 - 1),
    //                             cv::Point(img_width / 2 - circle_radius - 1, img_height / 2 - 1)};
    // const cv::Point line3[2] = {cv::Point(img_width / 2 - 1, img_height - 1),
    //                             cv::Point(img_width / 2 - 1, img_height / 2 + circle_radius - 1)};
    // const cv::Point line4[2] = {cv::Point(img_width - 1, img_height / 2 - 1),
    //                             cv::Point(img_width / 2 + circle_radius - 1, img_height / 2 - 1)};
    int mThreashold1, mThreashold2, mThreashold3;
    uint16_t mImgWidth, mImgHeight, mCircleRadius;
    cv::Point mCircleCenter;
    cv::Point mLine1[2], mLine2[2], mLine3[2], mLine4[2];

    cv::Mat mCameraMatrix;                               // Camera matrix
    cv::Mat mDistCoeffs;                                 // Distortion coefficients
    cv::Mat mNewCameraMatrix;                            // Camera matrix
    const float mDepth;                                  // Depth
    const cv::Point2f mTranslation;                      // Translation vector
    const cv::Point2f mErrorCompensation{5.0f, 16.0f};   // Error compensation
    double fx, fy, cx, cy;                               // Camera parameters
};