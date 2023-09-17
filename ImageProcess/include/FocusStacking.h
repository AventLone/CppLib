/**********************************************************
 * @author AventLone
 * @version 0.2
 * @date 2023-04-25
 * @copyright Copyright (AventLone) 2023
 **********************************************************/
#pragma once
#include <opencv2/opencv.hpp>

namespace avent
{
class FocusStacking
{
public:
    explicit FocusStacking() = default;
    FocusStacking(const FocusStacking&) = delete;
    FocusStacking& operator=(const FocusStacking&) = delete;
    ~FocusStacking() = default;

    /********************************************************************************
     * @brief Extract the relatively sharpest part of each image in this group,
     * combine them into one.
     * @param imgs A std::vector that stores your group of images.
     * @param dst A cv::Mat to store the fusion outcome.
     ********************************************************************************/
    void fuse(const std::vector<cv::Mat>& imgs, cv::Mat& dst)
    {
        focusStack(alignImgs(imgs), dst);
    }
    // stitch,fuse, fusion

    /********************************************************************************
     * @brief Extract the relatively sharpest part of each image in this group,
     * combine them into one.Focus Stack all the images under the given folder
     * @param dir_path Path to a directory containing your group of images.
     * @param dst A cv::Mat to store the fusion outcome.
     ********************************************************************************/
    void fuse(const std::string& dir_path, cv::Mat& dst);

    // static cv::Mat doLap(const cv::Mat& img);

private:
    /*** SIFT Feature detector ***/
    cv::Ptr<cv::SIFT> mDetector{cv::SIFT::create()};
    cv::Ptr<cv::DescriptorMatcher> mMatcher{cv::BFMatcher::create()};

private:
    /********************************************************************************
     * @brief Align a group of images based on the object in images.
     * @param imgs A std::vector that stores your group of images.
     * @return A std::vector that stores a group of images that has been aligned.
     ********************************************************************************/
    std::vector<cv::Mat> alignImgs(const std::vector<cv::Mat>& imgs);
    /********************************************************************************
     * @brief Extract the relatively sharpest part of the image.
     * @param img A cv::Mat.
     * @return Value of your parameter.
     ********************************************************************************/
    cv::Mat doLap(const cv::Mat& img) const;
    /********************************************************************************
     * @brief Parse a setting file, yaml, to get value of the parameter.
     * @param setting_file Path to the setting file.
     * @param param Name of your parameter.
     * @return Value of your parameter.
     ********************************************************************************/
    void focusStack(const std::vector<cv::Mat>& aligned_imgs, cv::Mat& output);
};
}   // namespace avent
