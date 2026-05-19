#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

class MjpegWebServer {
public:
    explicit MjpegWebServer(int port = 8080, int jpeg_quality = 80);
    ~MjpegWebServer();

    bool Start();
    void Stop();
    void UpdateFrame(const cv::Mat& frame);

private:
    void EncodeLoop();
    void AcceptLoop();
    void HandleClient(int client_fd);
    bool SendAll(int fd, const void* data, size_t len);

private:
    int port_;
    int jpeg_quality_;

    std::atomic<bool> running_{false};
    int listen_fd_ = -1;

    std::thread accept_thread_;
    std::thread encode_thread_;

    std::mutex raw_mtx_;
    std::condition_variable frame_cv_;
    cv::Mat raw_frame_;
    uint64_t raw_seq_ = 0;

    std::mutex jpeg_mtx_;
    std::vector<unsigned char> latest_jpeg_;
    uint64_t jpeg_seq_ = 0;
};
