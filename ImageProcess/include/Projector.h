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
    Projector(const std::string& camera_info_path, const cv::Point2f& tvec, float depth);

    ~Projector() = default;

    std::vector<cv::Point2f> detect(const cv::Mat& src) const;

    std::unordered_set<uint8_t> locateHeap(const cv::Mat& src) const;

    /*** For Visualization ***/
    void detect(const cv::Mat& src, cv::Mat& dst) const;

    void locateHeap(const cv::Mat& src, cv::Mat& dst) const;

private:
    const int threashold_1 = 200, threashold_2 = 780, threashold_3 = 99999;
    const uint16_t img_width = 1600, img_height = 1200;
    const uint16_t circle_radius = 350;
    const cv::Point circle_center{img_width / 2 - 1, img_height / 2 - 1};
    const cv::Point line1[2] = {cv::Point(img_width / 2 - 1, 0),
                                cv::Point(img_width / 2 - 1, img_height / 2 - circle_radius - 1)};
    const cv::Point line2[2] = {cv::Point(0, img_height / 2 - 1),
                                cv::Point(img_width / 2 - circle_radius - 1, img_height / 2 - 1)};
    const cv::Point line3[2] = {cv::Point(img_width / 2 - 1, img_height - 1),
                                cv::Point(img_width / 2 - 1, img_height / 2 + circle_radius - 1)};
    const cv::Point line4[2] = {cv::Point(img_width - 1, img_height / 2 - 1),
                                cv::Point(img_width / 2 + circle_radius - 1, img_height / 2 - 1)};

    cv::Mat camera_matrix;                               // Camera matrix
    cv::Mat dist_coeffs;                                 // Distortion coefficients
    cv::Mat new_camera_matrix;                           // Camera matrix
    const float depth_;                                  // depth
    const cv::Point2f tvec_;                             // translation vector
    const cv::Point2f error_compensation{5.0f, 16.0f};   // Error compensation
    double fx, fy, cx, cy;                               // Camera parameters
};