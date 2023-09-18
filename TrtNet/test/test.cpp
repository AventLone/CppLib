#include "Yolo.h"
int main()
{
    auto net = std::make_unique<tensorRT::Yolo>(
        "../data/config.yaml", "../model/mosquito_yolov5_v3.trt", "../model/mosquito_classes.txt");
    // auto net = std::make_unique<tensorRT::Yolo>(net_params, "../model/coco.names");

    // cv::Mat img = cv::imread("/home/avent/Desktop/MosqiBot_Pictures/Task_14/Group_1/30.jpg");
    cv::Mat img = cv::imread("../data/images/4.jpg");

    cv::Mat img_detected;
    net->run(img, img_detected);

    // auto imgs = net->getIndividuals(img);
    // int i = 0;

    cv::imshow("22", img_detected);
    cv::waitKey(0);

    return 0;
}