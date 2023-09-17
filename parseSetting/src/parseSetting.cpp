#include "parseSetting.h"

namespace avent
{
template<>
int parseSettings(const std::string& setting_file, const std::string& param)
{
    cv::FileStorage fs_settings(setting_file, cv::FileStorage::READ);
    if (!fs_settings.isOpened())
    {
        exitWithInfo("Failed to open settings file at: " + setting_file);
    }

    cv::FileNode node = fs_settings[param];
    double outcome;
    if (!node.empty() && node.isInt())
    {
        outcome = int(node);
    }
    else
    {
        exitWithInfo("\"" + param + "\"" + " parameter doesn't exist or is not a real number!");
    }

    fs_settings.release();
    return outcome;
}

template<>
double parseSettings(const std::string& setting_file, const std::string& param)
{
    cv::FileStorage fs_settings(setting_file, cv::FileStorage::READ);
    if (!fs_settings.isOpened())
    {
        exitWithInfo("Failed to open settings file at: " + setting_file);
    }

    cv::FileNode node = fs_settings[param];
    double outcome;
    if (!node.empty() && node.isReal())
    {
        outcome = node.real();
    }
    else
    {
        exitWithInfo("\"" + param + "\"" + " parameter doesn't exist or is not a real number!");
    }

    fs_settings.release();
    return outcome;
}

template<>
std::string parseSettings(const std::string& setting_file, const std::string& param)
{
    cv::FileStorage fs_settings(setting_file, cv::FileStorage::READ);
    if (!fs_settings.isOpened())
    {
        exitWithInfo("Failed to open settings file at: " + setting_file);
    }

    cv::FileNode node = fs_settings[param];
    std::string outcome;
    if (!node.empty() && node.isString())
    {
        outcome = node.string();
    }
    else
    {
        exitWithInfo("\"" + param + "\"" + " parameter doesn't exist or is not a String!");
    }
    fs_settings.release();
    return outcome;
}

template<>
cv::Mat parseSettings(const std::string& setting_file, const std::string& param)
{
    cv::FileStorage fs_settings(setting_file, cv::FileStorage::READ);
    if (!fs_settings.isOpened())
    {
        exitWithInfo("Failed to open settings file at: " + setting_file);
    }
    cv::FileNode node = fs_settings[param];
    cv::Mat mat;
    if (node.isMap())
    {
        mat = node.mat();
    }
    else
    {
        exitWithInfo("\"" + param + "\"" + " parameter doesn't exist or is not a Mat!");
    }
    fs_settings.release();
    return mat;
}
}   // namespace avent