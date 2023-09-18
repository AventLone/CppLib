/**********************************************************
 * @author AventLone
 * @version 0.1
 * @date 2023-09-18
 * @copyright Copyright (AventLone) 2023
 **********************************************************/
#pragma once
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include <fstream>

namespace tensorRT
{
struct Deleter
{
    template<typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template<typename T>
using UniquePtr = std::unique_ptr<T, Deleter>;

// template<typename T>
// using SharedPtr = std::shared_ptr<T, Deleter>;

/********************************************************************************
 * @brief
 * 用于创建IBuilder、IRuntime或IRefitter实例的记录器用于通过该接口创建的所有对象。
 * 在释放所有创建的对象之前，记录器应一直有效。
 * 主要是实例化ILogger类下的log()方法。
 ********************************************************************************/
class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* message) noexcept override
    {
        // suppress info-level messages
        if (severity != nvinfer1::ILogger::Severity::kINFO)
        {
            std::cout << message << std::endl;
        }
    }
};

class Net
{
public:
    explicit Net(const std::string& setting_fiel, const std::string& model_path);
    virtual ~Net();
    Net(const Net& T) = delete;
    Net& operator=(const Net& T) = delete;

    void run(const cv::Mat& src, std::vector<cv::Mat>& dst);

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;   // The TensorRT engine used to run the network
    UniquePtr<nvinfer1::IExecutionContext> mContext;

    void* mCudaBuffer[2];   // GPU memory buffer.

protected:
    /*** Net parameters ***/
    int mInputTensorIdx, mOutputTensorIdx;
    nvinfer1::Dims mInputDims, mOutputDims;   // The dimensions of the input and output to the network.
    size_t mOutputDataSize;

    /*** Buffer ***/
    cv::Mat mOutput;   // A buffer storing output of the net.

    void doInference(const cv::Mat& input);

    virtual void postprocess(const cv::Mat& src, std::vector<cv::Mat>& dst) = 0;
};
}   // namespace tensorRT