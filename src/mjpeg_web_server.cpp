#include "mjpeg_web_server.h"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

MjpegWebServer::MjpegWebServer(int port, int jpeg_quality)
    : port_(port), jpeg_quality_(jpeg_quality) {}

MjpegWebServer::~MjpegWebServer() {
    Stop();
}

bool MjpegWebServer::Start() {
    if (running_.load()) {
        return true;
    }

    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        std::cerr << "[web] socket create failed" << std::endl;
        return false;
    }

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[web] bind failed on port " << port_ << std::endl;
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (listen(listen_fd_, 8) < 0) {
        std::cerr << "[web] listen failed" << std::endl;
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    std::cout << "[web] 已启动: http://127.0.0.1:" << port_ << " 或 http://<设备IP>:" << port_ << std::endl;

    running_.store(true);
    encode_thread_ = std::thread(&MjpegWebServer::EncodeLoop, this);
    accept_thread_ = std::thread(&MjpegWebServer::AcceptLoop, this);
    return true;
}

void MjpegWebServer::Stop() {
    if (!running_.exchange(false)) {
        return;
    }

    frame_cv_.notify_all();

    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
        close(listen_fd_);
        listen_fd_ = -1;
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }
}

void MjpegWebServer::UpdateFrame(const cv::Mat& frame) {
    if (!running_.load()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lk(raw_mtx_);
        raw_frame_ = frame.clone();
        raw_seq_++;
    }
    frame_cv_.notify_one();
}

bool MjpegWebServer::SendAll(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

void MjpegWebServer::EncodeLoop() {
    uint64_t consumed = 0;
    while (running_.load()) {
        cv::Mat local_frame;
        {
            std::unique_lock<std::mutex> lk(raw_mtx_);
            frame_cv_.wait(lk, [this, &consumed] {
                return !running_.load() || raw_seq_ != consumed;
            });

            if (!running_.load()) {
                break;
            }

            local_frame = raw_frame_.clone();
            consumed = raw_seq_;
        }

        if (local_frame.empty()) {
            continue;
        }

        std::vector<unsigned char> jpeg;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
        if (!cv::imencode(".jpg", local_frame, jpeg, params)) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lk(jpeg_mtx_);
            latest_jpeg_ = std::move(jpeg);
            jpeg_seq_++;
        }
    }
}

void MjpegWebServer::AcceptLoop() {
    if (listen_fd_ < 0) {
        running_.store(false);
        return;
    }

    while (running_.load()) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (running_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            continue;
        }

        std::thread(&MjpegWebServer::HandleClient, this, client_fd).detach();
    }
}

void MjpegWebServer::HandleClient(int client_fd) {
    char req_buf[2048];
    std::memset(req_buf, 0, sizeof(req_buf));
    ssize_t n = recv(client_fd, req_buf, sizeof(req_buf) - 1, 0);
    if (n <= 0) {
        close(client_fd);
        return;
    }

    std::string req(req_buf, static_cast<size_t>(n));
    bool is_get_stream = req.find("GET /stream") == 0;
    bool is_head_stream = req.find("HEAD /stream") == 0;
    bool is_get_root = req.find("GET /") == 0;
    bool is_head_root = req.find("HEAD /") == 0;
    bool is_stream = is_get_stream || is_head_stream;
    bool is_root = is_get_root || is_head_root;

    if (is_stream) {
        const std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Connection: close\r\n"
            "Cache-Control: no-cache\r\n"
            "Pragma: no-cache\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

        if (!SendAll(client_fd, header.data(), header.size())) {
            close(client_fd);
            return;
        }

        if (is_head_stream) {
            close(client_fd);
            return;
        }

        uint64_t sent_seq = 0;
        while (running_.load()) {
            std::vector<unsigned char> jpeg;
            uint64_t seq = 0;
            {
                std::lock_guard<std::mutex> lk(jpeg_mtx_);
                if (!latest_jpeg_.empty() && jpeg_seq_ != sent_seq) {
                    jpeg = latest_jpeg_;
                    seq = jpeg_seq_;
                }
            }

            if (jpeg.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                continue;
            }

            std::string part =
                "--frame\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";

            if (!SendAll(client_fd, part.data(), part.size()) ||
                !SendAll(client_fd, jpeg.data(), jpeg.size()) ||
                !SendAll(client_fd, "\r\n", 2)) {
                break;
            }

            sent_seq = seq;
        }

        close(client_fd);
        return;
    }

    if (is_root) {
        const std::string body =
            "<!doctype html><html><head><meta charset='utf-8'><title>YOLO11 Stream</title></head>"
            "<body style='margin:0;background:#111;color:#fff;font-family:Arial,sans-serif;'>"
            "<div style='padding:10px'>YOLO11 实时画面</div>"
            "<img src='/stream' style='max-width:100vw;height:auto;display:block;margin:0 auto;'/>"
            "</body></html>";

        std::string resp =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n\r\n";

        if (is_get_root) {
            resp += body;
        }

        SendAll(client_fd, resp.data(), resp.size());
        close(client_fd);
        return;
    }

    const std::string not_found =
        "HTTP/1.1 404 Not Found\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n\r\n";
    SendAll(client_fd, not_found.data(), not_found.size());
    close(client_fd);
}
