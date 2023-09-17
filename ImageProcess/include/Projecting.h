/**********************************************************
 * @author AventLone
 * @version 0.2
 * @date 2023-04-25
 * @copyright Copyright (AventLone) 2023
 **********************************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_set>

namespace avent
{
class Projecting
{
    enum DisplayMode
    {
        SHOW_INDEX,   // Show the index of mosquitos.
        SHOW_AREA,    // Show the pixel area of mosquitos.
        SHOW_NOTHING
    };

public:
    explicit Projecting(const std::string& setting_file);
    Projecting(const Projecting&) = delete;
    Projecting& operator=(const Projecting&) = delete;
    ~Projecting() = default;

    /********************************************************************************
     * @brief Get coordinates of the targets on the image.
     * @param src A cv::Mat for input.
     * @return A std::vector containing a group of coordinates of the targets.
     ********************************************************************************/
    std::vector<cv::Point2f> locateTarget(const cv::Mat& src) const;
    /********************************************************************************
     * @brief Determine which part or parts of the plate has mosquito heap
     * @param src A cv::Mat for input.
     * @return A std::unordered_set<uint8_t> containing the indexes of the parts
     * existing mosquito heap on the plate.
     ********************************************************************************/
    std::unordered_set<uint8_t> locateHeap(const cv::Mat& src) const;

    /********************************************************************************
     * @brief Visualize coordinates of the targets on the image.
     * @param src A cv::Mat for input.
     * @param dst A cv::Mat for output.
     ********************************************************************************/
    void locateTarget(const cv::Mat& src, cv::Mat& dst, DisplayMode display_mode = SHOW_INDEX) const;
    /********************************************************************************
     * @brief Visualize a part or some parts of the plate has mosquito heap
     * @param src A cv::Mat for input.
     ********************************************************************************/
    void locateHeap(const cv::Mat& src, cv::Mat& dst) const;

private:
    /*** Intrinsics of the camera ***/
    uint16_t mImgWidth, mImgHeight;
    cv::Mat mCameraMatrix, mDistCoeffs, mNewCameraMatrix;
    double fx, fy, cx, cy;

    /*** Extrinsics of the camera ***/
    float mDepth;               // Depth
    cv::Point2f mTranslation;   // Translation vector

    /*** Parameters for locating heaps ***/
    uint16_t mCircleRadius;
    cv::Point mCircleCenter;
    cv::Point mLine1[2], mLine2[2], mLine3[2], mLine4[2];

    /*** Parameters for lowering errors ***/
    int mThreashold[3];
    cv::Point2f mErrorCompensation;   // Error compensation

    /*** Display mode ***/
    DisplayMode mDispalyMode;
};
}   // namespace avent