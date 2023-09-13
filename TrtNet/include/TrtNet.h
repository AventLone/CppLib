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
using trtUniquePtr = std::unique_ptr<T, Deleter>;

// template<typename T>
// using trtSharedPtr = std::shared_ptr<T, Deleter>;

/** @brief
 * 用于创建IBuilder、IRuntime或IRefitter实例的记录器用于通过该接口创建的所有对象。
 * 在释放所有创建的对象之前，记录器应一直有效。
 * 主要是实例化ILogger类下的log()方法。
 **/
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

struct NetParams
{
    std::string trt_model_path;
    std::string input_tensor_name, output_tensor_name;
    int input_tensor_index, output_tensor_index;
    nvinfer1::Dims input_dims;    // The dimensions of the input to the network.
    nvinfer1::Dims output_dims;   // The dimensions of the output to the network.
    size_t output_data_size;
};
using NetParams = struct NetParams;

class Net
{
public:
    explicit Net(const NetParams& params);
    virtual ~Net();
    Net(const Net& T) = delete;
    Net& operator=(const Net& T) = delete;

    void run(const cv::Mat& src, cv::Mat& dst);

protected:
    void doInference(const cv::Mat& input);

    virtual void postprocess(cv::Mat& img) = 0;

    cv::Mat mOutput;   // The output of the net/model.

    NetParams mParams;

private:
    // nvinfer1::ICudaEngine* mEngine;   // The TensorRT engine used to run the network
    // trtSharedPtr<nvinfer1::ICudaEngine> mEngine;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    trtUniquePtr<nvinfer1::IExecutionContext> mContext;


    // nvinfer1::IExecutionContext* mContext;

    void* mCudaBuffer[2];   // Create GPU memory buffer.
};
}   // namespace tensorRT