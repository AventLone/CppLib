#include "parseSetting.h"

int main()
{
    // std::array<std::string, 3> params{"image_height", "CameraMatrix.fx", "camera_matrix"};
    // auto outcomes = avent::parseSettings("../data/config.yaml", params);

    // auto outcome_1 = std::get<int>(outcomes[0]);
    // auto outcome_2 = std::get<double>(outcomes[1]);
    // auto outcome_3 = std::get<cv::Mat>(outcomes[2]);
    // auto outcome = outcomes[0];

    // std::cout << outcome_1 << std::endl;
    // std::cout << outcome_2 << std::endl;
    // std::cout << outcome_3 << std::endl;
    std::string param = "Yolo.InputTensorName";
    // auto outcom = avent::parseSettings("../data/config.yaml", "Yolo.InputTensorName");
    auto outcom = avent::parseSettings<std::string>("../data/config.yaml", param);
    std::cout << outcom << std::endl;

    return 0;
}