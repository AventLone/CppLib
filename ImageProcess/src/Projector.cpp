#include "Projector.h"


Projector::Projector(const std::string& setting_file, const cv::Point2f& translation, float depth)
    : mTranslation(translation), mDepth(depth)
{
    /** Read camera parameters from the yaml file **/
    
    cv::FileStorage fs(setting_file, cv::FileStorage::READ);
    fs["camera_matrix"] >> mCameraMatrix;
    fs["distortion_coefficients"] >> mDistCoeffs;
    if (mCameraMatrix.empty() || mDistCoeffs.empty())
    {
        throw std::runtime_error("\033[1;31mFailed to read camera parameters!\033[0m");
    }
    fs.release();
    mNewCameraMatrix =
        cv::getOptimalNewCameraMatrix(mCameraMatrix, mDistCoeffs, cv::Size(1600, 1200), 1, cv::Size(1600, 1200));

    fx = mNewCameraMatrix.at<double>(0, 0);
    fy = mNewCameraMatrix.at<double>(1, 1);
    cx = mNewCameraMatrix.at<double>(0, 2);
    cy = mNewCameraMatrix.at<double>(1, 2);
}

std::vector<cv::Point2f> Projector::detect(const cv::Mat& src) const
{
    cv::Mat undistorted_img, img_gray, img_blur, img_binary;
    cv::undistort(src, undistorted_img, mCameraMatrix, mDistCoeffs, mNewCameraMatrix);
    cv::cvtColor(undistorted_img, img_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(img_gray, img_blur, 5);
    cv::threshold(img_blur, img_binary, 100, 255, cv::THRESH_BINARY_INV);

    cv::Mat out, stats, centroids;
    int number = cv::connectedComponentsWithStats(img_binary, out, stats, centroids, 8, CV_16U);

    std::vector<cv::Point2f> points;   // Target mosquitos coordinates
    cv::Point2f point;
    // uint16_t index = 0;
    for (int i = 0; i < number; ++i)
    {
        /*** Remove small connected areas and oversized areas ***/
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 100 || stats.at<int>(i, cv::CC_STAT_AREA) > 600)
        {
            continue;
        }

        double u = centroids.at<double>(i, 0);   // Unit: pixel
        double v = centroids.at<double>(i, 1);

        /** Calculate the real-world coordinates from the pixel **/
        point.x = (u - cx) * mDepth / fx;
        point.y = (v - cy) * mDepth / fy;

        points.push_back(point + mTranslation + mErrorCompensation);
    }
    return points;
}

std::unordered_set<uint8_t> Projector::locateHeap(const cv::Mat& src) const
{
    cv::Mat undistorted_img, img_gray, img_blur, img_binary;
    cv::undistort(src, undistorted_img, mCameraMatrix, mDistCoeffs, mNewCameraMatrix);
    cv::cvtColor(undistorted_img, img_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(img_gray, img_blur, 5);
    cv::threshold(img_blur, img_binary, 100, 255, cv::THRESH_BINARY_INV);

    cv::Mat out, stats, centroids;
    int number = cv::connectedComponentsWithStats(img_binary, out, stats, centroids, 8, CV_16U);

    std::unordered_set<uint8_t> area_mark_set;
    for (int i = 0; i < number; ++i)
    {
        /*** Find the heaping area. ***/
        if (stats.at<int>(i, cv::CC_STAT_AREA) < threashold_2)
        {
            continue;
        }
        double u = centroids.ptr<double>(i)[0];   // Unit: pixel
        double v = centroids.ptr<double>(i)[1];
        int x = static_cast<int>(u) - circle_center.x;
        int y = static_cast<int>(v) - circle_center.y;
        if (x * x + y * y < circle_radius)
        {
            area_mark_set.insert(0);
        }
        else if (x > 0 && y < 0)
        {
            area_mark_set.insert(1);
        }
        else if (x < 0 && y < 0)
        {
            area_mark_set.insert(2);
        }
        else if (x < 0 && y > 0)
        {
            area_mark_set.insert(3);
        }
        else
        {
            area_mark_set.insert(4);
        }
    }
    return area_mark_set;
}

void Projector::detect(const cv::Mat& src, cv::Mat& dst) const
{
    cv::Mat undistorted_img, img_gray, img_blur, img_binary;
    cv::undistort(src, undistorted_img, mCameraMatrix, mDistCoeffs, mNewCameraMatrix);
    cv::cvtColor(undistorted_img, img_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(img_gray, img_blur, 5);
    cv::threshold(img_blur, img_binary, 100, 255, cv::THRESH_BINARY_INV);

    dst = undistorted_img.clone();

    cv::Mat out, stats, centroids;
    int number = cv::connectedComponentsWithStats(img_binary, out, stats, centroids, 8, CV_16U);

    // std::vector<cv::Point2f> points;   // Target mosquitos coordinates
    // cv::Point2f point;
    uint16_t index = 0;
    int area = 0;
    for (int i = 0; i < number; ++i)
    {
        area = stats.at<int>(i, cv::CC_STAT_AREA);
        /*** Remove small connected areas and oversized areas ***/
        if (area < threashold_1 || area > threashold_2)
        {
            continue;
        }

        // double u = centroids.at<double>(i, 0);   // Unit: pixel
        // double v = centroids.at<double>(i, 1);

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // cv::circle(dst, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::rectangle(dst, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2, 8, 0);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        // cv::putText(
        //     dst, std::to_string(++index), cv::Point(u, v), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(
            dst, std::to_string(area), cv::Point(x, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
}

void Projector::locateHeap(const cv::Mat& src, cv::Mat& dst) const
{
    cv::Mat undistorted_img, img_gray, img_blur, img_binary;
    cv::undistort(src, undistorted_img, mCameraMatrix, mDistCoeffs, mNewCameraMatrix);
    cv::cvtColor(undistorted_img, img_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(img_gray, img_blur, 5);
    cv::threshold(img_blur, img_binary, 100, 255, cv::THRESH_BINARY_INV);

    dst = undistorted_img.clone();
    cv::circle(dst, circle_center, circle_radius, cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::line(dst, line1[0], line1[1], cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::line(dst, line2[0], line2[1], cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::line(dst, line3[0], line3[1], cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::line(dst, line4[0], line4[1], cv::Scalar(255, 0, 0), 1, 8, 0);

    cv::Mat out, stats, centroids;
    int number = cv::connectedComponentsWithStats(img_binary, out, stats, centroids, 8, CV_16U);

    int area = 0;
    for (int i = 0; i < number; ++i)
    {
        /*** Find the heaping area. ***/
        area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < threashold_2 || area > threashold_3)
        {
            continue;
        }
        double u = centroids.ptr<double>(i)[0];   // Unit: pixel
        double v = centroids.ptr<double>(i)[1];

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        cv::rectangle(dst, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2, 8, 0);
        cv::circle(dst, cv::Point(u, v), 2, cv::Scalar(0, 0, 255), 2, 8, 0);
        cv::putText(
            dst, std::to_string(area), cv::Point(x, y - 6), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
}
