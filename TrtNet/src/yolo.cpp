#include "yolo.h"

namespace tensorRT
{
Yolo::Yolo(const NetParams& params, std::string class_file_path) : Net(params)
{
    std::ifstream ifs(class_file_path.c_str());
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
}

void Yolo::drawPred(int class_index, float confidence, const cv::Rect& box, cv::Mat& frame)
{
    int top = box.y, bottom = box.y + box.height, left = box.x, right = box.x + box.width;

    /*** Draw a rectangle displaying the bounding box ***/
    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 1);

    /*** Get the label for the class name and its confidence. ***/
    std::string conf_label = cv::format("%.2f", confidence);
    std::string label;
    if (!classes.empty())
    {
        label = classes[class_index] + ":" + conf_label;
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

void Yolo::postprocess(cv::Mat& img)
{
    std::vector<int> class_indexes;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point> points;
    cv::Point point;

    float width_factor = static_cast<float>(img.cols) / 640.0f;
    float height_factor = static_cast<float>(img.rows) / 640.0f;

    /*** output组成：batch_id, x0, y0, x1, y1, cls_id, score ***/
    for (int i = 0; i < mOutput.rows; i++)
    {
        float confidence = mOutput.ptr<float>(i)[4];
        if (confidence >= mConfThreshold)
        {
            cv::Mat classes_scores = mOutput.row(i).colRange(5, mParams.output_dims.d[2]);
            // cv::Mat classes_scores = mOutput.row(i).colRange(5, 13);
            cv::Point class_id_point;
            double score;
            // 获取一组数据中最大值及其位置
            cv::minMaxLoc(classes_scores, 0, &score, 0, &class_id_point);
            /*** 置信度 0～1 之间 ***/
            if (score > 0.25) //
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

                boxes.push_back(box);
                class_indexes.push_back(class_id_point.x);
                confidences.push_back(score);
            }
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, mConfThreshold, mNmsThreshold, indexes);
    for (int index : indexes)
    {
        drawPred(class_indexes[index], confidences[index], boxes[index], img);
    }
}
}   // namespace tensorRT
