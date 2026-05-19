#ifndef _yolo11_det_
#define _yolo11_det_
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

static Logger gLogger_;
using namespace nvinfer1;
const int kOutputSize_ = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class YOLO11_DET{
    private:
        
        std::string engine_name;
        std::string cuda_post_process = "c";
        int model_bboxes;
        
        IRuntime* runtime = nullptr;
        ICudaEngine* engine = nullptr;
        IExecutionContext* context = nullptr;

        cudaStream_t stream;

        float* device_buffers[2];
        float* output_buffer_host = nullptr;
        float* decode_ptr_host = nullptr;
        float* decode_ptr_device = nullptr;

        std::vector<Detection> res_batch;

    public:
    YOLO11_DET(std::string engine_name_input){
        engine_name = engine_name_input;
        cudaSetDevice(kGpuId);
        deserialize_engine(engine_name, &runtime, &engine, &context);
        CUDA_CHECK(cudaStreamCreate(&stream));
        cuda_preprocess_init(kMaxInputImageSize);
        auto out_dims = engine->getBindingDimensions(1);
        model_bboxes = out_dims.d[0];
        prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host,
                   &decode_ptr_device, cuda_post_process);

        cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("road_mask", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Yolo_Detect", cv::WINDOW_AUTOSIZE);
        cv::moveWindow("Trajectory", 2500, 400);
        cv::moveWindow("road_mask", 1200, 400);
        cv::moveWindow("Yolo_Detect", 1850, 400);
    }

    std::vector<Detection>  YOLO11_loop(cv::Mat& frame){
        res_batch.clear();

        cuda_batch_preprocess(frame, device_buffers[0], kInputW, kInputH, stream);

        infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host,
              decode_ptr_device, model_bboxes, cuda_post_process);

        batch_nms(res_batch, output_buffer_host, 1, kOutputSize_, kConfThresh, kNmsThresh);

        draw_bbox(frame, res_batch);
        return res_batch;
    }

    ~YOLO11_DET(){
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(device_buffers[0]));
        CUDA_CHECK(cudaFree(device_buffers[1]));
        CUDA_CHECK(cudaFree(decode_ptr_device));
        delete[] decode_ptr_host;
        delete[] output_buffer_host;
        cuda_preprocess_destroy();
        // Destroy the engine
        delete context;
        delete engine;
        delete runtime;
    }

    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                            IExecutionContext** context) {
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << engine_name << " error!" << std::endl;
            assert(false);
        }
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        char* serialized_engine = new char[size];
        assert(serialized_engine);
        file.read(serialized_engine, size);
        file.close();

        *runtime = createInferRuntime(gLogger_);
        assert(*runtime);
        *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
        assert(*engine);
        *context = (*engine)->createExecutionContext();
        assert(*context);
        delete[] serialized_engine;
    }

    void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                        float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device,
                        std::string cuda_post_process) {
        assert(engine->getNbBindings() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(kInputTensorName);
        const int outputIndex = engine->getBindingIndex(kOutputTensorName);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize_ * sizeof(float)));
        *output_buffer_host = new float[kBatchSize * kOutputSize_];

    }

    void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
            float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process) {
        // infer on the batch asynchronously, and DMA output back to host
        // auto start = std::chrono::system_clock::now();
        context.enqueueV2(buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize_ * sizeof(float), cudaMemcpyDeviceToHost,
                                stream));
        // auto end = std::chrono::system_clock::now();
        // std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                // << "ms" << std::endl;

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

};

#endif
