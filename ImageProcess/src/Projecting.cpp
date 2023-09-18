#include "Projecting.h"
#include "parseSetting.h"

namespace avent
{
Projecting::Projecting(const std::string& setting_file)
{
    std::array<std::string, 8> params{"Image.width",
                                      "Image.height",
                                      "CameraMatrix",
                                      "DistortionCoefficients",
                                      "BaseSizeAndDepth",
                                      "Threashold",
                                      "ErrorCompensation",
                                      "DisplayMode"};
    auto param_values = parseSettings(setting_file, params);

    mImgWidth = std::get<int>(param_values[0]);
    mImgHeight = std::get<int>(param_values[1]);
    mCameraMatrix = std::get<cv::Mat>(param_values[2]);
    mDistCoeffs = std::get<cv::Mat>(param_values[3]);

    cv::Vec3f base_size_depth = std::get<cv::Mat>(param_values[4]);
    cv::Mat threashold = std::get<cv::Mat>(param_values[5]);
    cv::Vec2f error_compensation = std::get<cv::Mat>(param_values[6]);

    mTranslation = cv::Point2f(base_size_depth[0] / 2.0f, base_size_depth[1] / 2.0f);
    mDepth = base_size_depth[2];
    mErrorCompensation = cv::Point2f(error_compensation[0], error_compensation[1]);
    std::memcpy(mThreashold, threashold.ptr<int>(), 3 * sizeof(int));

    mCircleRadius = mImgHeight / 3 - 20;
    mCircleCenter = cv::Point(mImgWidth / 2 - 1, mImgHeight / 2 - 1);

    mNewCameraMatrix = cv::getOptimalNewCameraMatrix(
        mCameraMatrix, mDistCoeffs, cv::Size(mImgWidth, mImgHeight), 1, cv::Size(mImgWidth, mImgHeight));

    fx = mNewCameraMatrix.at<double>(0, 0);
    fy = mNewCameraMatrix.at<double>(1, 1);
    cx = mNewCameraMatrix.at<double>(0, 2);
    cy = mNewCameraMatrix.at<double>(1, 2);

    mLine1[0] = cv::Point(mImgWidth / 2 - 1, 0);
    mLine1[1] = cv::Point(mImgWidth / 2 - 1, mImgHeight / 2 - mCircleRadius - 1);
    mLine2[0] = cv::Point(0, mImgHeight / 2 - 1);
    mLine2[1] = cv::Point(mImgWidth / 2 - mCircleRadius - 1, mImgHeight / 2 - 1);
    mLine3[0] = cv::Point(mImgWidth / 2 - 1, mImgHeight - 1);
    mLine3[1] = cv::Point(mImgWidth / 2 - 1, mImgHeight / 2 + mCircleRadius - 1);
    mLine4[0] = cv::Point(mImgWidth - 1, mImgHeight / 2 - 1);
    mLine4[1] = cv::Point(mImgWidth / 2 + mCircleRadius - 1, mImgHeight / 2 - 1);

    std::string display_mode = std::get<std::string>(param_values[7]);
    if (display_mode == "show_index")
    {
        mDispalyMode = SHOW_INDEX;
    }
    else if (display_mode == "show_area")
    {
        mDispalyMode = SHOW_AREA;
    }
    else if (display_mode == "show_nothing")
    {
        mDispalyMode = SHOW_NOTHING;
    }
    else
    {
        try
        {
            throwError(display_mode + " is a wrong format!");
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            mDispalyMode = SHOW_NOTHING;
        }
    }
}

std::vector<cv::Point2f> Projecting::locateTarget(const cv::Mat& src) const
{
    cv::Mat undistorted_img, img_gray, img_blur, img_binary;
    cv::undistort(src, undistorted_img, mCameraMatrix, mDistCoeffs, mNewCameraMatrix);
    cv::cvtColor(undistorted_img, img_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(img_gray, img_blur, 5);
    cv::threshold(img_blur, img_binary, 100, 255, cv::THRESH_BINARY_INV);

    cv::Mat out, stats, centroids;
    int number = cv::connectedComponentsWithStats(img_binary, out, stats, centroids, 8, CV_16U);

    std::vector<cv::Point2f> points;
    cv::Point2f point;
    for (int i = 0; i < number; ++i)
    {
        /*** Remove small connected areas and oversized areas ***/
        if (stats.at<int>(i, cv::CC_STAT_AREA) < mThreashold[0] || stats.at<int>(i, cv::CC_STAT_AREA) > mThreashold[1])
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

std::unordered_set<uint8_t> Projecting::locateHeap(const cv::Mat& src) const
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
        if (stats.at<int>(i, cv::CC_STAT_AREA) < mThreashold[1])
        {
            continue;
        }
        double u = centroids.ptr<double>(i)[0];   // Unit: pixel
        double v = centroids.ptr<double>(i)[1];
        int x = static_cast<int>(u) - mCircleCenter.x;
        int y = static_cast<int>(v) - mCircleCenter.y;
        if (x * x + y * y < mCircleRadius)
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

void Projecting::locateTarget(const cv::Mat& src, cv::Mat& dst) const
{
    cv::Mat undistorted_img, img_gray, img_blur, img_binary;
    cv::undistort(src, undistorted_img, mCameraMatrix, mDistCoeffs, mNewCameraMatrix);
    cv::cvtColor(undistorted_img, img_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(img_gray, img_blur, 5);
    cv::threshold(img_blur, img_binary, 100, 255, cv::THRESH_BINARY_INV);

    dst = undistorted_img.clone();

    cv::Mat out, stats, centroids;
    int number = cv::connectedComponentsWithStats(img_binary, out, stats, centroids, 8, CV_16U);

    uint16_t index = 0;
    int area = 0;
    for (int i = 0; i < number; ++i)
    {
        area = stats.at<int>(i, cv::CC_STAT_AREA);
        /*** Remove small connected areas and oversized areas ***/
        if (area < mThreashold[0] || area > mThreashold[1])
        {
            continue;
        }

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        cv::rectangle(dst, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2, 8, 0);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        switch (mDispalyMode)
        {
            case SHOW_INDEX:
                cv::putText(dst,
                            std::to_string(++index),
                            cv::Point(x, y - 6),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.8,
                            cv::Scalar(0, 255, 0),
                            2);
                break;
            case SHOW_AREA:
                cv::putText(dst,
                            std::to_string(area),
                            cv::Point(x, y - 6),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.8,
                            cv::Scalar(0, 255, 0),
                            2);
                break;
            default:
                break;
        }
    }
}

void Projecting::locateHeap(const cv::Mat& src, cv::Mat& dst) const
{
    cv::Mat undistorted_img, img_gray, img_blur, img_binary;
    cv::undistort(src, undistorted_img, mCameraMatrix, mDistCoeffs, mNewCameraMatrix);
    cv::cvtColor(undistorted_img, img_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(img_gray, img_blur, 5);
    cv::threshold(img_blur, img_binary, 100, 255, cv::THRESH_BINARY_INV);

    dst = undistorted_img.clone();
    cv::circle(dst, mCircleCenter, mCircleRadius, cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::line(dst, mLine1[0], mLine1[1], cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::line(dst, mLine2[0], mLine2[1], cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::line(dst, mLine3[0], mLine3[1], cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::line(dst, mLine4[0], mLine4[1], cv::Scalar(255, 0, 0), 1, 8, 0);

    cv::Mat out, stats, centroids;
    int number = cv::connectedComponentsWithStats(img_binary, out, stats, centroids, 8, CV_16U);

    uint16_t index = 0;
    int area = 0;
    for (int i = 0; i < number; ++i)
    {
        /*** Find the heaping area. ***/
        area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < mThreashold[1] || area > mThreashold[2])
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
        switch (mDispalyMode)
        {
            case SHOW_INDEX:
                cv::putText(dst,
                            std::to_string(++index),
                            cv::Point(x, y - 6),
                            cv::FONT_HERSHEY_SIMPLEX,
                            1.0,
                            cv::Scalar(0, 0, 255),
                            2);
                break;
            case SHOW_AREA:
                cv::putText(dst,
                            std::to_string(area),
                            cv::Point(x, y - 6),
                            cv::FONT_HERSHEY_SIMPLEX,
                            1.0,
                            cv::Scalar(0, 0, 255),
                            2);
                break;
            default:
                break;
        }
    }
}
}   // namespace avent