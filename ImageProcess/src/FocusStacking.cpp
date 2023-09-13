#include "FocusStacking.h"
#include <filesystem>

void FocusStacking::run(const std::vector<cv::Mat>& imgs_, cv::Mat& dst)
{
    focusStack(alignImgs(imgs_), dst);
}

/*** Focus Stack all the images under the given folder ***/
void FocusStacking::run(const std::string& dir_path, cv::Mat& dst)
{
    std::filesystem::path p(dir_path);

    if (!std::filesystem::exists(p) || !std::filesystem::is_directory(p))
    {
        std::cerr << "Error: " << p << " is not a directory" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::string> img_paths;
    for (const auto& entry : std::filesystem::directory_iterator(p))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
        {
            img_paths.push_back(entry.path().string());
        }
    }
    if (img_paths.empty())
    {
        std::cerr << "Error: There is no image file under " << dir_path << " !" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<cv::Mat> imgs;
    imgs.reserve(img_paths.size());
    for (const auto& filename : img_paths)
    {
        imgs.emplace_back(cv::imread(filename));
    }
    if (imgs[0].empty())
    {
        std::cerr << "Error: " << img_paths[0] << " is empty" << std::endl;
        exit(EXIT_FAILURE);
    }

    focusStack(alignImgs(imgs), dst);
}

std::vector<cv::Mat> FocusStacking::alignImgs(const std::vector<cv::Mat>& imgs)
{
    std::vector<cv::Mat> aligned_imgs;
    aligned_imgs.push_back(imgs[0]);

    int w = imgs[0].cols, h = imgs[0].rows;

    cv::Mat refe_gray;
    cvtColor(imgs[0], refe_gray, cv::COLOR_BGR2GRAY);

    /** 初始化参照图像的关键点和描述子 **/
    std::vector<cv::KeyPoint> keypoints_refe;
    cv::Mat descriptors_refe;

    /** 检测关键点位置并计算描述子**/
    detector->detectAndCompute(refe_gray, cv::Mat(), keypoints_refe, descriptors_refe);


    for (uint16_t i = 1; i < imgs.size(); ++i)
    {
        int w_i = imgs[i].cols, h_i = imgs[i].rows;
        if (w_i != w || h_i != h)
        {
            std::cerr << "图片大小不一致！" << std::endl;
            exit(EXIT_FAILURE);
        }

        cv::Mat imgGray;
        cvtColor(imgs[i], imgGray, cv::COLOR_BGR2BGRA);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detectAndCompute(imgGray, cv::Mat(), keypoints, descriptors);

        std::vector<std::vector<cv::DMatch>> rawMatches;
        matcher->knnMatch(descriptors_refe, descriptors, rawMatches, 2);
        std::vector<cv::DMatch> goodMatches;
        for (auto& rawMatche : rawMatches)
        {
            if (rawMatche[0].distance < 0.69 * rawMatche[1].distance)
            {
                goodMatches.push_back(rawMatche[0]);
            }
        }

        /** 根据distance对goodMatches进行排序 **/
        std::sort(goodMatches.begin(),
                  goodMatches.end(),
                  [](cv::DMatch& a, cv::DMatch& b) { return a.distance < b.distance; });

        /** 取goodMatches的一部分 **/
        std::vector<cv::DMatch> matches(goodMatches.begin(),
                                        goodMatches.begin() + uint16_t(goodMatches.size() * 2 / 3));

        std::vector<cv::Point2f> pointsRefe;
        std::vector<cv::Point2f> points;
        for (const auto& match : matches)
        {
            points.push_back(keypoints[match.trainIdx].pt);
            pointsRefe.push_back(keypoints_refe[match.queryIdx].pt);
        }

        cv::Mat Homo = cv::findHomography(points, pointsRefe, cv::RANSAC, 2.0);
        cv::Mat outcome;
        cv::warpPerspective(imgs[i], outcome, Homo, imgs[i].size(), cv::INTER_LINEAR);
        aligned_imgs.push_back(outcome);
    }

    return aligned_imgs;
}

cv::Mat FocusStacking::doLap(const cv::Mat& img)
{
    int kernel_size = 3, blur_size = 3;
    cv::Mat img_blur;
    cv::GaussianBlur(img, img_blur, cv::Size(blur_size, blur_size), 0, 0);
    cv::Mat outcome;
    cv::Laplacian(img_blur, outcome, CV_32F, kernel_size);
    return outcome;
}

void FocusStacking::focusStack(const std::vector<cv::Mat>& aligned_imgs, cv::Mat& output)
{
    int w = aligned_imgs[0].cols, h = aligned_imgs[0].rows;
    std::vector<cv::Mat> laps;
    cv::Mat img_gray;
    for (const auto& img : aligned_imgs)
    {
        cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        laps.push_back(doLap(img_gray));
    }

    /** 对每张lap中的像素取绝对值 **/
    for (auto& lap : laps)
    {
        for (int i = 0; i < h; ++i)
        {
            auto* pixl = lap.ptr<float>(i);
            for (int j = 0; j < w; ++j)
            {
                pixl[j] = fabs(pixl[j]);   // 浮点型绝对值: fabs()
            }
        }
    }

    /*** 创建一个空白图像 ***/
    cv::Mat maxim = cv::Mat::zeros(cv::Size(laps[0].cols, laps[0].rows), laps[0].type());
    for (int i = 0; i < laps[0].rows; ++i)
    {
        for (int j = 0; j < laps[0].cols; ++j)
        {
            auto pixl = laps[0].at<float>(i, j);
            for (const auto& img : laps)
            {
                if (pixl < img.at<float>(i, j))
                {
                    pixl = img.at<float>(i, j);
                }
            }
            maxim.at<float>(i, j) = pixl;
        }
    }

    /** 根据maxim制作掩膜 **/
    std::vector<cv::Mat> masks;
    for (auto& lap : laps)
    {
        cv::Mat mask = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
        for (int i = 0; i < h; ++i)
        {
            const auto pixl_lap = lap.ptr<float>(i);
            const auto pixl_maxim = maxim.ptr<float>(i);
            auto pixl_mask = mask.ptr<uchar>(i);

            for (int j = 0; j < w; ++j)
            {
                if (pixl_lap[j] == pixl_maxim[j])
                {
                    pixl_mask[j] = 1;
                }
                else
                {
                    pixl_mask[j] = 0;
                }
            }
        }
        masks.emplace_back(mask);
    }

    /** 输出图片 **/
    for (uint16_t i = 0; i < aligned_imgs.size(); ++i)
    {
        cv::bitwise_not(aligned_imgs[i], output, masks[i]);
    }
    cv::bitwise_not(output, output);
}
