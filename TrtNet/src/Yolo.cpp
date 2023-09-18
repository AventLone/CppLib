#include "Yolo.h"
#include "parseSetting.h"

namespace tensorRT
{
Yolo::Yolo(const std::string& setting_file, const std::string& net_file, const std::string& class_file)
    : Net(setting_file, net_file)
{
    std::ifstream ifs(class_file.c_str());
    std::string line;
    while (std::getline(ifs, line))
    {
        mClasses.push_back(line);
    }
    std::string mode = avent::parseSettings<std::string>(setting_file, "OutputMode");
    if (mode == "whole_mode")
    {
        personalizedPostprocess = &Yolo::getWholeWithMarks;
    }
    else if (mode == "separate_mode")
    {
        personalizedPostprocess = &Yolo::getSeparateTarget;
    }
    else
    {
        std::cerr << mode << " is a wrong format!" << std::endl;
        personalizedPostprocess = &Yolo::getWholeWithMarks;
    }
}

void Yolo::drawPred(int class_index, float confidence, const cv::Rect& box, cv::Mat& frame) const
{
    int top = box.y, bottom = box.y + box.height, left = box.x, right = box.x + box.width;

    /*** Draw a rectangle displaying the bounding box ***/
    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 1);

    /*** Get the label for the class name and its confidence. ***/
    std::string conf_label = cv::format("%.2f", confidence);
    std::string label;
    if (!mClasses.empty())
    {
        label = mClasses[class_index] + ":" + conf_label;
    }

    /*** Display the label at the top of the bounding box. ***/
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::rectangle(frame,
                  cv::Point(left, top - labelSize.height),
                  cv::Point(left + labelSize.width, top + baseLine),
                  cv::Scalar(255, 255, 255),
                  cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

void Yolo::getWholeWithMarks(const cv::Mat& src, std::vector<cv::Mat>& dst, const std::vector<int>& indices) const
{
    cv::Mat temp_src = src.clone();
    for (int index : indices)
    {
        drawPred(mClassIndexes[index], mConfidences[index], mBoxes[index], temp_src);
    }
    dst.emplace_back(temp_src);
}

void Yolo::getSeparateTarget(const cv::Mat& src, std::vector<cv::Mat>& dst, const std::vector<int>& indices) const
{
    for (int index : indices)
    {
        cv::Mat tmp = src(mBoxes[index]).clone();

        std::string conf_label = cv::format("%.2f", mConfidences[index]);
        std::string label;
        if (!mClasses.empty())
        {
            label = mClasses[index] + ":" + conf_label;
        }

        /*** Display the label at the top of the bounding box. ***/
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        // top = std::max(top, labelSize.height);
        cv::rectangle(
            tmp, cv::Point(0, 0), cv::Point(labelSize.width, 2 * baseLine + 6), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(tmp,
                    label,
                    cv::Point(0, 2 * baseLine + 3),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0),
                    1,
                    cv::LINE_AA);

        dst.emplace_back(tmp);
    }
}

void Yolo::postprocess(const cv::Mat& src, std::vector<cv::Mat>& dst)
{
    mClassIndexes.clear();
    mConfidences.clear();
    mBoxes.clear();

    float width_factor = static_cast<float>(src.cols) / static_cast<float>(mInputDims.d[2]);
    float height_factor = static_cast<float>(src.rows) / static_cast<float>(mInputDims.d[3]);

    /*** output组成：batch_id, x0, y0, x1, y1, cls_id, score ***/
    for (int i = 0; i < mOutput.rows; i++)
    {
        float confidence = mOutput.ptr<float>(i)[4];
        if (confidence >= mConfThreshold)
        {
            cv::Mat classes_scores = mOutput.row(i).colRange(5, mOutputDims.d[2]);
            // cv::Mat classes_scores = mOutput.row(i).colRange(5, 13);
            cv::Point class_id_point;
            double score;
            // 获取一组数据中最大值及其位置
            cv::minMaxLoc(classes_scores, 0, &score, 0, &class_id_point);
            /*** 置信度 0～1 之间 ***/
            if (score > 0.25)   //
            {
                float cx = mOutput.ptr<float>(i)[0];
                float cy = mOutput.ptr<float>(i)[1];
                float ow = mOutput.ptr<float>(i)[2];
                float oh = mOutput.ptr<float>(i)[3];

                cv::Rect box;
                box.x = static_cast<int>((cx - 0.5 * ow) * width_factor);
                box.y = static_cast<int>((cy - 0.5 * oh) * height_factor);
                box.width = static_cast<int>(ow * width_factor);
                box.height = static_cast<int>(oh * height_factor);

                mBoxes.push_back(box);
                mClassIndexes.push_back(class_id_point.x);
                mConfidences.push_back(score);
            }
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(mBoxes, mConfidences, mConfThreshold, mNmsThreshold, indices);
    (this->*personalizedPostprocess)(src, dst, indices);
}
}   // namespace tensorRT
