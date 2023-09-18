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

    // std::vector<cv::Mat> getIndividuals(const cv::Mat& src);

private:
    const float mConfThreshold{0.5f};   // Confidence threshold
    const float mNmsThreshold{0.4f};    // Non-maximum suppression threshold

    std::vector<std::string> mClasses;

    void drawPred(int class_id, float confidence, const cv::Rect& box, cv::Mat& frame) const;

    void postprocess(cv::Mat&) override;
};
}   // namespace tensorRT
