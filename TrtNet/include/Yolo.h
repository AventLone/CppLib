/**********************************************************
 * @author AventLone
 * @version 0.1
 * @date 2023-09-18
 * @copyright Copyright (AventLone) 2023
 **********************************************************/
#include "TrtNet.h"
namespace tensorRT
{
class Yolo : public Net
{
public:
    explicit Yolo(const std::string& setting_file, const std::string& net_file, const std::string& class_file);
    Yolo(const Yolo&) = delete;
    Yolo& operator=(const Yolo&) = delete;
    ~Yolo() = default;

private:
    const float mConfThreshold{0.5f};   // Confidence threshold
    const float mNmsThreshold{0.4f};    // Non-maximum suppression threshold

    std::vector<std::string> mClasses;   // Categories the model can recognize.

    /*** Parameters for post process ***/
    std::vector<int> mClassIndexes;
    std::vector<float> mConfidences;
    std::vector<cv::Rect> mBoxes;

private:
    void drawPred(int class_id, float confidence, const cv::Rect& box, cv::Mat& frame) const;

    /*** Two choices for personaalized postprocess ***/
    void (Yolo::*personalizedPostprocess)(const cv::Mat&, std::vector<cv::Mat>&, const std::vector<int>&) const;
    void getWholeWithMarks(const cv::Mat& src, std::vector<cv::Mat>& dst, const std::vector<int>& indices) const;
    void getSeparateTarget(const cv::Mat& src, std::vector<cv::Mat>& dst, const std::vector<int>& indices) const;

    void postprocess(const cv::Mat& src, std::vector<cv::Mat>& dst) override;
};
}   // namespace tensorRT
