#include "AutoDriveSystem.h"
#include "mjpeg_web_server.h"
#include "camera.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace {
std::atomic<bool> g_keep_running(true);

void HandleSignal(int) {
    g_keep_running.store(false);
}
}  // namespace

int main() {
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    std::signal(SIGPIPE, SIG_IGN);

    cv::Mat img;
    V4L2Capture cap("/dev/v4l/by-id/usb-LRCP_V1080P_LRCP_V1080P-video-index0", 640, 480);
    if (cap.InitCamera() != 0) {
        std::cerr << "摄像头初始化失败" << std::endl;
        return 1;
    }

    AutoDriveSystem autoDriveSystem;
    MjpegWebServer web_server(8080, 80);
    if (!web_server.Start()) {
        std::cerr << "Web 服务启动失败，请检查端口 8080 是否被占用" << std::endl;
        return 1;
    }

    std::cout << "初始化完成，先暂停3秒..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "开始播报道路语音..." << std::endl;
    autoDriveSystem.PlayRoadAudioBlocking();
    std::cout << "道路语音播报结束，进入主循环..." << std::endl;

    auto last_fps_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    double fps = 0.0;
    int a = 0;
    cv::Mat img_clone ;     

    while (g_keep_running.load()) {
        img = cap.ResultCamera();
        if (img.empty()) {
            std::cerr << "no image" <<std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        // img_clone = img.clone();

        autoDriveSystem.RunOnce(img);

        ++frame_count;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_sec = std::chrono::duration<double>(now - last_fps_time).count();
        if (elapsed_sec >= 0.5) {
            fps = frame_count / elapsed_sec;
            frame_count = 0;
            last_fps_time = now;
        }

        std::ostringstream fps_text;
        fps_text << "FPS: " << std::fixed << std::setprecision(1) << fps;
        cv::putText(img, fps_text.str(), cv::Point(400, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        web_server.UpdateFrame(img);
        // cv::imwrite("./../img/img_" + std::to_string(a++) + ".jpg", img_clone);
    }

    web_server.Stop();
    return 0;
}
