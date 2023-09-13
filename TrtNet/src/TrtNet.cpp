#include "TrtNet.h"

namespace tensorRT
{
Net::Net(const NetParams& params) : mParams(params)
{
    cudaSetDevice(0);
    /*** Read local net file. ***/
    std::ifstream file_ptr(mParams.trt_model_path, std::ios::binary);
    if (!file_ptr.good())
    {
        throw std::runtime_error("\033[31m Failed to load the trt engine file!\033[0m");
    }

    size_t size = 0;
    file_ptr.seekg(0, file_ptr.end);   // 将读指针从文件末尾开始移动0个字节
    size = file_ptr.tellg();           // 返回读指针的位置，此时读指针的位置就是文件的字节数
    file_ptr.seekg(0, file_ptr.beg);   // 将读指针从文件开头开始移动0个字节
    char* model_stream = new char[size];
    file_ptr.read(model_stream, size);
    file_ptr.close();

    Logger logger;   // 日志记录接口
    // nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);   // 反序列化引擎

    trtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

    // 保存模型的模型结构、模型参数以及最优计算kernel配置；
    // 不能跨平台和跨TensorRT版本移植
    // mEngine = runtime->deserializeCudaEngine(model_stream, size);   // 推理引擎
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size),
                                                     Deleter());   // 推理引擎

    // 储存中间值，实际进行推理的对象
    // 由engine创建，可创建多个对象，进行多推理任务
    mContext = trtUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    // mContext = mEngine->createExecutionContext();   // 上下文
    delete[] model_stream;

    // mCudaBuffer = new void*[2];
    /*** Allocate GPU memory on input part of the buffer. ***/
    mParams.input_tensor_index = mEngine->getBindingIndex(mParams.input_tensor_name.c_str());
    mParams.input_dims = mEngine->getBindingDimensions(mParams.input_tensor_index);
    size_t input_data_length = mParams.input_dims.d[1] * mParams.input_dims.d[2] * mParams.input_dims.d[3];
    cudaMalloc(&(mCudaBuffer[mParams.input_tensor_index]), input_data_length * sizeof(float));

    /*** Allocate GPU memory on output part of the buffer. ***/
    mParams.output_tensor_index = mEngine->getBindingIndex(mParams.output_tensor_name.c_str());
    mParams.output_dims = mEngine->getBindingDimensions(mParams.output_tensor_index);
    mParams.output_data_size = mParams.output_dims.d[1] * mParams.output_dims.d[2];
    cudaMalloc(&(mCudaBuffer[mParams.output_tensor_index]), mParams.output_data_size * sizeof(float));
}

Net::~Net()
{
    // cudaFree(mCudaBuffer[mParams.input_tensor_index]);
    // cudaFree(mCudaBuffer[mParams.output_tensor_index]);
}

void Net::run(const cv::Mat& src, cv::Mat& dst)
{
    if (src.empty())
    {
        throw std::runtime_error("\033[31m The input of tensorRT Net is empty!\033[0m");
    }
    dst = src.clone();
    doInference(src);
    postprocess(dst);
}

void Net::doInference(const cv::Mat& input)
{
    /*** The preprocess of the input. ***/
    cv::Mat input_tensor;
    cv::dnn::blobFromImage(input,
                           input_tensor,
                           1 / 255.0,
                           cv::Size(mParams.input_dims.d[2], mParams.input_dims.d[3]),
                           cv::Scalar(0, 0, 0),
                           true,
                           false);

    /*** Transfer the data from cpu memory to gpu memory. ***/
    cudaMemcpy(mCudaBuffer[mParams.input_tensor_index],
               input_tensor.ptr<float>(),
               input_tensor.total() * sizeof(float),
               cudaMemcpyHostToDevice);

    mContext->executeV2(mCudaBuffer);

    /*** Transfer the data from gpu memory to cpu memory. ***/
    float* output_array = new float[mParams.output_data_size];
    cudaMemcpy(output_array,
               mCudaBuffer[mParams.output_tensor_index],
               mParams.output_data_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // mOutput = std::vector<float>(output_array, output_array + mParams.output_data_size);
    mOutput = cv::Mat(mParams.output_dims.d[1], mParams.output_dims.d[2], CV_32F, output_array).clone();
    delete[] output_array;
}
}   // namespace tensorRT
