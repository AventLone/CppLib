#include "TrtNet.h"
#include "parseSetting.h"

namespace tensorRT
{
Net::Net(const std::string& setting_file, const std::string& model_path)
{
    cudaSetDevice(0);
    /*** Read local net file. ***/
    std::ifstream file_ptr(model_path, std::ios::binary);
    if (!file_ptr.good())
    {
        // throw std::runtime_error("\033[31m Failed to load the trt engine file!\033[0m");
        avent::exitWithInfo("Failed to load the trt net file!");
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

    UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

    // 保存模型的模型结构、模型参数以及最优计算kernel配置；
    // 不能跨平台和跨TensorRT版本移植
    // mEngine = runtime->deserializeCudaEngine(model_stream, size);   // 推理引擎
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size),
                                                     Deleter());   // 推理引擎

    // 储存中间值，实际进行推理的对象
    // 由engine创建，可创建多个对象，进行多推理任务
    mContext = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    // mContext = mEngine->createExecutionContext();   // 上下文
    delete[] model_stream;

    // mCudaBuffer = new void*[2];
    std::string input_tensor_name = avent::parseSettings<std::string>(setting_file, "InputTensorName");
    std::string output_tensor_name = avent::parseSettings<std::string>(setting_file, "OutputTensorName");
    /*** Allocate GPU memory on input part of the buffer. ***/
    mInputTensorIdx = mEngine->getBindingIndex(input_tensor_name.c_str());
    mInputDims = mEngine->getBindingDimensions(mInputTensorIdx);
    size_t input_data_length = mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3];
    cudaMalloc(&(mCudaBuffer[mInputTensorIdx]), input_data_length * sizeof(float));

    /*** Allocate GPU memory on output part of the buffer. ***/
    mOutputTensorIdx = mEngine->getBindingIndex(output_tensor_name.c_str());
    mOutputDims = mEngine->getBindingDimensions(mOutputTensorIdx);
    mOutputDataSize = mOutputDims.d[1] * mOutputDims.d[2];
    cudaMalloc(&(mCudaBuffer[mOutputTensorIdx]), mOutputDataSize * sizeof(float));
}


Net::~Net()
{
    // cudaFree(mCudaBuffer[mParams.mInputTensorIdx]);
    // cudaFree(mCudaBuffer[mParams.mOutputTensorIdx]);
}

void Net::run(const cv::Mat& src, std::vector<cv::Mat>& dst)
{
    if (src.empty())
    {
        avent::throwError("The input of tensorRT Net is empty!");
    }
    // dst = src.clone();
    // cv::Mat temp_src = src.clone();
    doInference(src);
    postprocess(src, dst);
}

void Net::doInference(const cv::Mat& input)
{
    /*** The preprocess of the input. ***/
    cv::Mat input_tensor;
    cv::dnn::blobFromImage(
        input, input_tensor, 1 / 255.0, cv::Size(mInputDims.d[2], mInputDims.d[3]), cv::Scalar(0, 0, 0), true, false);

    /*** Transfer the data from CPU memory to GPU memory. ***/
    cudaMemcpy(mCudaBuffer[mInputTensorIdx],
               input_tensor.ptr<float>(),
               input_tensor.total() * sizeof(float),
               cudaMemcpyHostToDevice);

    mContext->executeV2(mCudaBuffer);

    /*** Transfer the data from GPU memory to CPU memory. ***/
    float* output_array = new float[mOutputDataSize];
    cudaMemcpy(output_array, mCudaBuffer[mOutputTensorIdx], mOutputDataSize * sizeof(float), cudaMemcpyDeviceToHost);

    // mOutput = std::vector<float>(output_array, output_array + mParams.mOutputDataSize);
    mOutput = cv::Mat(mOutputDims.d[1], mOutputDims.d[2], CV_32F, output_array).clone();
    delete[] output_array;
}
}   // namespace tensorRT
