#ifndef _yolov11_road_seg_
#define _yolov11_road_seg_
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

static Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * (sizeof(Detection) - sizeof(float) * 51) / sizeof(float) + 1;
const static int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);


class YOLO11_ROAD_SEG {
private:
    using Clock = std::chrono::steady_clock;
public:
    struct Timing {
        double preprocess_ms = 0.0;
        double infer_ms = 0.0;
        double nms_ms = 0.0;
        double split_det_ms = 0.0;
        double process_mask_ms = 0.0;
        double draw_mask_ms = 0.0;
        double draw_bbox_ms = 0.0;
        double total_ms = 0.0;
    };
private:
    std::string engine_name ;
    std::string cuda_post_process = "g";
    int model_bboxes;   

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    cudaStream_t stream;

    // Prepare cpu and gpu buffers
    float* device_buffers[3];
    float* output_buffer_host = nullptr;
    float* output_seg_buffer_host = nullptr;
    float* decode_ptr_host = nullptr;
    float* decode_ptr_device =  nullptr;

    std::vector<Detection> res_batch;
    cv::Mat merged_mask;
    Timing last_timing_;
    
public:
    YOLO11_ROAD_SEG(std::string engine_name_input ) {
        engine_name = engine_name_input;
        cudaSetDevice(kGpuId);
        deserialize_engine(engine_name, &runtime, &engine, &context);
        CUDA_CHECK(cudaStreamCreate(&stream));
        cuda_preprocess_init(kMaxInputImageSize);
        auto out_dims = engine->getBindingDimensions(1);
        model_bboxes = out_dims.d[0];
        prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &device_buffers[2], &output_buffer_host,
                    &output_seg_buffer_host, &decode_ptr_host, &decode_ptr_device, cuda_post_process);
    }

    cv::Mat YOLO11_ROAD_SEG_LOOP(cv::Mat& img, std::vector<Detection>& object_batch) {
        const auto t0 = Clock::now();
        res_batch.clear();
        const auto t_pre0 = Clock::now();
        // Preprocess
        cuda_batch_preprocess(img, device_buffers[0], kInputW, kInputH, stream);
        const auto t_pre1 = Clock::now();
        // Run inference
        const auto t_inf0 = Clock::now();
        infer(*context, stream, (void**)device_buffers, output_buffer_host, output_seg_buffer_host, kBatchSize,
              decode_ptr_host, decode_ptr_device, model_bboxes, cuda_post_process);
        const auto t_inf1 = Clock::now();
        // NMS
        const auto t_nms0 = Clock::now();
        // batch_nms(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh);
        batch_process(res_batch, decode_ptr_host, 1, bbox_element, img);
        CUDA_CHECK(cudaMemcpyAsync(output_seg_buffer_host, device_buffers[2], kBatchSize * kOutputSegSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int count = std::min(static_cast<int>(decode_ptr_host[0]), kMaxNumOutputBbox);
        int kept_idx = 0;
        for (int i_det = 0; i_det < count && kept_idx < static_cast<int>(res_batch.size()); i_det++) {
            const int basic_pos = 1 + i_det * bbox_element;
            if (static_cast<int>(decode_ptr_host[basic_pos + 6]) != 1) {
                continue;
            }
            const int src_index = static_cast<int>(decode_ptr_host[basic_pos + 7]);
            const float* src_det = &output_buffer_host[1 + src_index * (sizeof(Detection) / sizeof(float))];
            memcpy(res_batch[kept_idx].mask, src_det + 6, sizeof(float) * 32);
            kept_idx++;
        }

        const auto t_nms1 = Clock::now();

        //存放着
        // 只生成class_id == 0 -road 的所有掩码并合成到一张CV_8UC1的图片中
        const auto t_split0 = Clock::now();
        std::vector<Detection> road_det;
        std::vector<Detection> trafic_det;
        road_det.reserve(res_batch.size());
        for (const auto& det : res_batch) {
            if (static_cast<int>(det.class_id) == 0) {
                road_det.push_back(det);
            } else {
                trafic_det.push_back(det);
            }
        }
        const auto t_split1 = Clock::now();
        const auto t_mask0 = Clock::now();
        auto masks = process_mask(&output_seg_buffer_host[0 * kOutputSegSize], kOutputSegSize, road_det);
        const auto t_mask1 = Clock::now();
        const auto t_draw_mask0 = Clock::now();
        draw_mask_bbox(img, road_det, masks, merged_mask);
        const auto t_draw_mask1 = Clock::now();
        const auto t_draw_bbox0 = Clock::now();
        draw_bbox(img, trafic_det);
        const auto t_draw_bbox1 = Clock::now();
        object_batch = trafic_det;

        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        last_timing_.preprocess_ms = ms(t_pre0, t_pre1);
        last_timing_.infer_ms = ms(t_inf0, t_inf1);
        last_timing_.nms_ms = ms(t_nms0, t_nms1);
        last_timing_.split_det_ms = ms(t_split0, t_split1);
        last_timing_.process_mask_ms = ms(t_mask0, t_mask1);
        last_timing_.draw_mask_ms = ms(t_draw_mask0, t_draw_mask1);
        last_timing_.draw_bbox_ms = ms(t_draw_bbox0, t_draw_bbox1);
        last_timing_.total_ms = ms(t0, Clock::now());
        return merged_mask;
    }

    const Timing& GetLastTiming() const { return last_timing_; }

    ~YOLO11_ROAD_SEG() {
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(device_buffers[0]));
        CUDA_CHECK(cudaFree(device_buffers[1]));
        CUDA_CHECK(cudaFree(device_buffers[2]));
        CUDA_CHECK(cudaFree(decode_ptr_device));
        delete[] decode_ptr_host;
        delete[] output_buffer_host;
        delete[] output_seg_buffer_host;
        cuda_preprocess_destroy();
        // Destroy the engine
        delete context;
        delete engine;
        delete runtime;
    }

    static cv::Rect get_downscale_rect(float bbox[4], float scale) {
        float left = bbox[0];
        float top = bbox[1];
        float right = bbox[0] + bbox[2];
        float bottom = bbox[1] + bbox[3];

        left = left < 0 ? 0 : left;
        top = top < 0 ? 0 : top;
        right = right > kInputW ? kInputW : right;
        bottom = bottom > kInputH ? kInputH : bottom;

        left /= scale;
        top /= scale;
        right /= scale;
        bottom /= scale;
        return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
    }

    std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets) {
        const int mask_h = kInputH / 4;
        const int mask_w = kInputW / 4;
        const int mask_area = mask_h * mask_w;
        (void)proto_size;

        std::vector<cv::Mat> masks;
        masks.reserve(dets.size());

        cv::Mat proto_planes[32];
        for (int j = 0; j < 32; j++) {
            proto_planes[j] = cv::Mat(mask_h, mask_w, CV_32FC1, const_cast<float*>(proto + j * mask_area));
        }

        cv::Mat logits, exp_buf;
        for (size_t i = 0; i < dets.size(); i++) {
            cv::Mat mask_mat = cv::Mat::zeros(mask_h, mask_w, CV_32FC1);
            cv::Rect r = get_downscale_rect(dets[i].bbox, 4);
            r &= cv::Rect(0, 0, mask_w, mask_h);
            if (r.width <= 0 || r.height <= 0) {
                masks.push_back(mask_mat);
                continue;
            }

            logits.create(r.height, r.width, CV_32FC1);
            logits.setTo(0);
            for (int j = 0; j < 32; j++) {
                const float c = dets[i].mask[j];
                if (fabsf(c) < 1e-6f) {
                    continue;
                }
                cv::scaleAdd(proto_planes[j](r), c, logits, logits);
            }

            cv::multiply(logits, -1.0f, logits);
            cv::exp(logits, exp_buf);
            cv::add(exp_buf, 1.0f, exp_buf);
            cv::divide(1.0f, exp_buf, logits);
            logits.copyTo(mask_mat(r));
            masks.push_back(mask_mat);
        }
        return masks;
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

        *runtime = createInferRuntime(gLogger);
        assert(*runtime);
        *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
        assert(*engine);
        *context = (*engine)->createExecutionContext();
        assert(*context);
        delete[] serialized_engine;
    }

    void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                        float** output_seg_buffer_device, float** output_buffer_host, float** output_seg_buffer_host,
                        float** decode_ptr_host, float** decode_ptr_device, std::string cuda_post_process) {
        assert(engine->getNbBindings() == 3);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(kInputTensorName);
        const int outputIndex = engine->getBindingIndex(kOutputTensorName);
        const int outputIndex_seg = engine->getBindingIndex("proto");

        assert(inputIndex == 0);
        assert(outputIndex == 1);
        assert(outputIndex_seg == 2);
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)output_seg_buffer_device, kBatchSize * kOutputSegSize * sizeof(float)));

        *output_buffer_host = new float[kBatchSize * kOutputSize];
        *output_seg_buffer_host = new float[kBatchSize * kOutputSegSize];
        if (kBatchSize > 1) {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));

    }   

    void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, float* output_seg,
            int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes,
            std::string cuda_post_process) {
        // infer on the batch asynchronously, and DMA output back to host
        context.enqueueV2(buffers, stream, nullptr);
        
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);  //cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                   sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                                   stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

#endif
