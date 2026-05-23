#pragma once

#include "yolo11_det.hpp"
#include "usart.h"
#include "road_detection.h"
#include "yolov11_road_seg.hpp"
#include "Control.h"
#include "postprocess.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;

/*========================================================
*                   交通标志类别
========================================================*/
enum Class_Typical {
    Road,
    Change_Lanes,
    Warning_Sign,
    Green_Light,
    Red_Light,
    Cross_Road,
    Turn_Left,
    Turn_Right,
    Remove_Limit_Speed,
    Limit_10_Speed,
    People,
    Dangerous,
    People_disappear,                   //单独新加的小人消失标志位
    CLASS_NUM
};

/*========================================================
*                   车辆状态
========================================================*/
enum class TrajectoryState { 
    NORMAL,
    TURNING_LEFT,
    TURNING_RIGHT
};

enum class SpeedState { 
    FOLLOW_LINE, 
    STOP, 
    LIMIT_SPEED 
};

/*========================================================
*                   配置参数
========================================================*/
namespace Config {
    constexpr double MAX_VX = 0.6;                 // 正常巡线最大线速度上限
    constexpr double MAX_VW = 2;                   // 角速度上限（转向强度上限）
    constexpr int IMAGE_CENTER_X = 320;            // 图像中心 x（用于左右判断）
    constexpr int TURN_TIMEOUT_MS = 550;           // 左/右转状态最大持续时间（毫秒）
    constexpr int PEOPLE_DISAPPEAR_FRAMES = 8;     // 小人连续消失多少帧后恢复行驶
    constexpr int OBSTACLE_TRIGGER_FRAMES = 2;     // 锥桶连续命中多少帧才触发避障
    constexpr int OBSTACLE_MISS_FRAMES = 12;       // 锥桶连续丢失多少帧后退出避障
    constexpr int OBSTACLE_COOLDOWN_FRAMES = 25;   // 避障完成后的冷却帧数（防重复触发）
    constexpr int OBSTACLE_START_AREA = 3000;      // 锥桶面积超过该阈值后开始轨迹避障
    constexpr float OBSTACLE_OFFSET_PX = 20.0f;    // 固定轨迹偏移量（像素）
    constexpr float OBSTACLE_OFFSET_ALPHA = 0.4f; // 偏移平滑系数（越大响应越快）
    constexpr int OBSTACLE_DIRECTION_DEADZONE_PX = 20; // 中线附近死区（防左右抖动切换）
    constexpr int OBSTACLE_MARGIN_PX = 10;         // 轨迹点贴边保护边距（像素）
    constexpr int AUDIO_MIN_INTERVAL_MS = 1200;    // 同一标志最小播报间隔（毫秒）
    constexpr int CROSS_ROAD_STOP_MS = 3000;       // 同时识别到两个人行横道后的停车等待时间（毫秒）
    constexpr int SECOND_TURN_STOP_AREA = 2800;    // 第二次遇见左/右转时触发结束的面积阈值
}

/*========================================================
*                   定时器
========================================================*/
class Timer {
public:
    Timer() { Reset(); }
    void Reset() { start_ = std::chrono::steady_clock::now(); }
    bool Timeout(int ms) const {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_).count();
        return elapsed >= ms;
    }
private:
    std::chrono::steady_clock::time_point start_;
};

/*========================================================
*                   交通标志规则
========================================================*/
struct TrafficRule {
    int class_id;
    float min_area;
    int trigger_count;
    bool need_center_check;
    int center_threshold;
    bool center_inside; // true: abs(x-center)<threshold
    int decision_area;
};

/*========================================================
*                   系统上下文
========================================================*/
struct DriveContext {
    CarState car_state = {0};
    bool isPeopleAppear = 0;                                
    std::array<int, CLASS_NUM> traffic_count = {0};
    std::array<bool, CLASS_NUM> traffic_flag = {false};
    Trajectory_params trajectory;
};

/*========================================================
*               自动驾驶系统类
========================================================*/
class AutoDriveSystem {
public:
    AutoDriveSystem() : segmentor_("./../enigne/train28.engine") {
        usart_init(115200, "/dev/serial/by-path/platform-3610000.xhci-usb-0:2.3:1.0");           //串口通信初始化

        //PD控制器参数：kp, kd, 输出限制在[-max_vw, max_vw], 采样时间10ms
        PD_Init(&pd_controller_, 0.015, 0.0015, -Config::MAX_VW, Config::MAX_VW, 10);   
        InitTrafficRules();
        InitAudioSystem();
    }
    cv::Mat road_mask;
    void RunOnce(cv::Mat& img) {
        if (mission_finish_requested_) {
            FinalizeMissionAndExit();
            return;
        }

        /*-----识别道路，轨迹、目标检测，进行寻线,输出轨迹点、边界点--------- -------*/
        std::vector<Detection> object_batch;
        road_mask = segmentor_.YOLO11_ROAD_SEG_LOOP(img, object_batch);
        const auto& seg_timing = segmentor_.GetLastTiming();

        extract_trajectory.Extract(road_mask, ctx_.trajectory);

        UpdateTrafficSigns(object_batch, img);
        if (mission_finish_requested_) {
            FinalizeMissionAndExit();
            return;
        }
        HandleTrafficAudioEvents();
        UpdateObstacleTrajectoryState(img.cols);
        ApplyObstacleOffsetToTrajectory(img.cols);

        /*-----------------------决策控制-----------------------------*/
        //轨迹决策层---左转、右转、角点识别进行左转右转、遇见锥形桶需要进行避障、遇见变道需要播报声音
        TrajectoryDecision();

        // 变道标志触发后，控制点前瞻距离从60切换到100，并保持到比赛结束.
        if (ctx_.traffic_flag[Change_Lanes]) {
            if (!trajectory_control_changed_) {
                trajectory_control_ = 100;
                trajectory_control_changed_ = true;
                std::cout << "[CONTROL] trajectory_control switched to 100 (Change_Lanes triggered)." << std::endl;
            }
        }

        //根据轨迹计算速度
        computeSimple(img.cols,
                      ctx_.car_state,
                      ctx_.trajectory.road_trajectory,
                      pd_controller_,
                      Config::MAX_VX,
                      trajectory_control_);

        //速度控制层---此处计算出来速度，但是遇到小人，红灯的时候需要继续停车、如果之后看到绿灯就继续行使、遇见限速标识减速、遇见解除限速需要进行提速，    
        SpeedDecision();

        /*--------------------可视化 road和交通标志-------------------------*/
        Visualize(img);

        SetSpeed();
        bluetooth_send();

        if (mission_finish_requested_) {
            FinalizeMissionAndExit();
            return;
        }
    }

    void SetSpeed() {
        send_struct.x_vel_target = ctx_.car_state.vx * 1000;
        send_struct.z_vel_target = ctx_.car_state.vw * 1000;
    }

    void UpdateTrafficSigns(std::vector<Detection>& object_batch, cv::Mat& img) {
        bool people_detected_this_frame = false;
        bool dangerous_detected_this_frame = false;
        bool red_detected_this_frame = false;
        bool green_detected_this_frame = false;
        int cross_road_detect_count = 0;
        int dangerous_center_x = -1;
        int dangerous_area = 0;

        for ( auto& obj : object_batch) {
            TrafficRule rule{};
            bool has_rule = false;
            auto it = traffic_rules_.find(obj.class_id);
            if (it != traffic_rules_.end()) {
                rule = it->second;
                has_rule = true;
            } else if (obj.class_id == Change_Lanes) {
                // 变道规则兜底：YAML 未配置时仍可触发.
                rule = {Change_Lanes, 1200.0f, 2, false, 0, true, 2000};
                has_rule = true;
            } else if (obj.class_id == Warning_Sign) {
                // 注意标识(Attention)规则兜底：不是锥桶(Dangerous).
                rule = {Warning_Sign, 1000.0f, 2, false, 0, true, 1800};
                has_rule = true;
            }
            if (!has_rule)
                continue;

            // 红灯门控：只有在解除限速曾触发后，红灯才允许触发.
            if (obj.class_id == Red_Light && !remove_limit_seen_) {
                ctx_.traffic_count[Red_Light] = 0;
                ctx_.traffic_flag[Red_Light] = false;
                continue;
            }
            if (obj.mianji < rule.min_area)
                continue;

            bool valid = true;
            if (rule.need_center_check) {
                cv::Rect r = get_rect(img, obj.bbox);
                int x_pos = r.x + r.width / 2;
                int diff = abs(x_pos - Config::IMAGE_CENTER_X);
                valid = rule.center_inside ? (diff < rule.center_threshold) : (diff > rule.center_threshold);
            }
            if (!valid) 
                continue;

            // 绿灯只在“已进入红灯停车”阶段才允许触发，避免绿灯提前残留导致后续红灯被立刻抵消.
            if (obj.class_id == Green_Light && !ctx_.traffic_flag[Red_Light]) {
                ctx_.traffic_count[Green_Light] = 0;
                ctx_.traffic_flag[Green_Light] = false;
                continue;
            }

            // 已完成一次转弯后，再次遇见左/右转并且面积足够大，触发“停车+结束播报+退出程序”.
            if (!mission_finish_requested_ &&
                turn_completed_count_ >= 1 &&
                (obj.class_id == Turn_Left || obj.class_id == Turn_Right) &&
                obj.mianji > Config::SECOND_TURN_STOP_AREA) {
                mission_finish_requested_ = true;
                mission_finish_trigger_class_ = obj.class_id;
                mission_finish_trigger_area_ = static_cast<int>(obj.mianji);
                speed_state_ = SpeedState::STOP;
                trajectory_state_ = TrajectoryState::NORMAL;
                ctx_.traffic_flag[Turn_Left] = false;
                ctx_.traffic_flag[Turn_Right] = false;
                ctx_.traffic_count[Turn_Left] = 0;
                ctx_.traffic_count[Turn_Right] = 0;
                std::cout << "[MISSION] SECOND TURN SIGN DETECTED: class="
                          << ClassNameById(obj.class_id)
                          << " area=" << mission_finish_trigger_area_
                          << " -> STOP + PLAY FINISH + EXIT" << std::endl;
                break;
            }



            ctx_.traffic_count[obj.class_id]++;
            const bool was_flag_on = ctx_.traffic_flag[obj.class_id];
            if (ctx_.traffic_count[obj.class_id] > rule.trigger_count && obj.mianji > rule.decision_area) {
                ctx_.traffic_flag[obj.class_id] = true;
                if (obj.class_id == Remove_Limit_Speed) {
                    remove_limit_seen_ = true;
                }
                if (obj.class_id == Red_Light) {
                    red_detected_this_frame = true;
                }
                if (obj.class_id == Green_Light) {
                    green_detected_this_frame = true;
                }
                if (obj.class_id == People)
                    people_detected_this_frame = 1;
                if (obj.class_id == People && !was_flag_on) {
                    std::cout << "[TRAFFIC] PEOPLE FLAG ON"
                              << " area=" << static_cast<int>(obj.mianji)
                              << " count=" << ctx_.traffic_count[People]
                              << std::endl;
                }
                if (obj.class_id == Dangerous && obj.mianji > Config::OBSTACLE_START_AREA) {
                    cv::Rect r = get_rect(img, obj.bbox);
                    dangerous_center_x = r.x + r.width / 2;
                    dangerous_area = static_cast<int>(obj.mianji);
                    dangerous_detected_this_frame = true;
                }
                if (obj.class_id == Cross_Road) {
                    cross_road_detect_count++;
                }
            }
        }

        // 未处于激活状态时，清空红绿灯计数，防止历史残留影响下一轮触发.
        if (!ctx_.traffic_flag[Red_Light] && !red_detected_this_frame) {
            ctx_.traffic_count[Red_Light] = 0;
        }
        if (!ctx_.traffic_flag[Green_Light] && !green_detected_this_frame) {
            ctx_.traffic_count[Green_Light] = 0;
        }

        // 小人连续消失 N 帧后恢复行驶，避免偶发漏检导致抖动.
        if (people_detected_this_frame) {
            ctx_.isPeopleAppear = true;
            people_miss_frames_ = 0;
        } else if (ctx_.traffic_flag[People] || ctx_.isPeopleAppear) {
            people_miss_frames_++;
            if (people_miss_frames_ >= Config::PEOPLE_DISAPPEAR_FRAMES) {
                ctx_.traffic_flag[People] = false;
                ctx_.traffic_count[People] = 0;
                ctx_.isPeopleAppear = false;
                people_miss_frames_ = 0;
                std::cout << "[TRAFFIC] PEOPLE FLAG OFF" << std::endl;
            }
        }

        if (dangerous_detected_this_frame) {
            obstacle_center_x_ = dangerous_center_x;
            obstacle_area_ = dangerous_area;
            obstacle_detect_frames_++;
            obstacle_miss_frames_ = 0;
        } else {
            obstacle_area_ = 0;
            obstacle_detect_frames_ = 0; // 触发按“连续帧”计数
            obstacle_miss_frames_++;
        }

        // 同一帧同时看到两个人行横道，触发一次停车等待.
        if (cross_road_detect_count >= 2) {
            if (!cross_road_double_latched_) {
                cross_road_double_latched_ = true;
                cross_road_waiting_ = true;
                cross_road_timer_.Reset();
                speed_state_ = SpeedState::STOP;
                std::cout << "CROSS_ROAD x2 DETECTED: STOP 3s" << std::endl;
            }
        } else {
            cross_road_double_latched_ = false;
        }
    }
    

    void TrajectoryDecision() {
        if (cooldown_ > 0) {
            cooldown_--;
            return;
        }
        switch (trajectory_state_) {
            case TrajectoryState::NORMAL:
                Follow_Trajectory_Decision();
                break;
            case TrajectoryState::TURNING_LEFT: 
                LeftTurn_Decision(); 
                break;

            case TrajectoryState::TURNING_RIGHT: 
                RightTurn_Decision(); 
                break;
        }
    }

    void SpeedDecision() {
        switch (speed_state_) {
            case SpeedState::FOLLOW_LINE: 
                FollowLine_Speed(); 
                break;
            case SpeedState::STOP: 
                Stop();
                break;
            case SpeedState::LIMIT_SPEED: 
                LimitSpeed(); 
                break;
        }
    }

    void Visualize(Mat& img) {
        for (size_t i = 0; i < ctx_.trajectory.road_trajectory.size(); i++) {
            circle(img, ctx_.trajectory.road_trajectory[i], 5, Scalar(0,255,0), -1);
            circle(img, ctx_.trajectory.left_points[i], 5, Scalar(255,0,0), -1);
            circle(img, ctx_.trajectory.right_points[i], 5, Scalar(0,0,255), -1);
        }
        std::string info = "vx: " + std::to_string(ctx_.car_state.vx).substr(0,4) +
                           " vw: " + std::to_string(ctx_.car_state.vw).substr(0,4);
        putText(img, info, Point(30,30), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,255), 2);
    }

    DriveContext& GetContext() { return ctx_; }

private:
    void Follow_Trajectory_Decision() {
        if (ctx_.traffic_flag[Turn_Left]) {
            trajectory_state_ = TrajectoryState::TURNING_LEFT;
            turn_timer_.Reset();
            std::cout << "TURN LEFT" << std::endl;
        }
        else if (ctx_.traffic_flag[Turn_Right]) {
            trajectory_state_ = TrajectoryState::TURNING_RIGHT; turn_timer_.Reset();
            std::cout << "TURN RIGHT" << std::endl; 
        }
    }

    void LeftTurn_Decision() {
        ProcessTurning(ctx_.trajectory.left_points); 
        if (turn_timer_.Timeout(Config::TURN_TIMEOUT_MS))
            FinishTurn(Turn_Left); 
    }
    void RightTurn_Decision() { 
        ProcessTurning(ctx_.trajectory.right_points); 
        if (turn_timer_.Timeout(Config::TURN_TIMEOUT_MS)) 
            FinishTurn(Turn_Right); 
        }

    void FinishTurn(int class_id) { 
        trajectory_state_ = TrajectoryState::NORMAL;
        ctx_.traffic_flag[class_id] = false;
        ctx_.traffic_count[class_id] = 0;
        cooldown_ = 500;
        turn_completed_count_++;
        std::cout << "[TURN] COMPLETED COUNT = " << turn_completed_count_ << std::endl;
        std::cout << "TURN FINISHED" << std::endl;
    }

    void ProcessTurning(const std::vector<Point>& target_points) {
        if (ctx_.trajectory.road_trajectory.empty() || ctx_.trajectory.road_trajectory.size() != target_points.size())
            return;
        if (ctx_.trajectory.original_trajectory.empty())
            ctx_.trajectory.original_trajectory = ctx_.trajectory.road_trajectory;
        int n = ctx_.trajectory.road_trajectory.size();
        for (int i = 0; i < n; i++) {
            float alpha = static_cast<float>(i)/(n-1);
            ctx_.trajectory.road_trajectory[i].x = static_cast<int>((1-alpha)*ctx_.trajectory.original_trajectory[i].x + alpha*target_points[i].x);
        }
    }

    void FollowLine_Speed() {
        if (ctx_.traffic_flag[Red_Light]) {
            speed_state_ = SpeedState::STOP;
            std::cout << "/*------------Red_Light-----------------------------*/" << std::endl;
        } 
        else if(ctx_.traffic_flag[People]) {
            speed_state_ = SpeedState::STOP;
            std::cout << "/*------------Appear People! Stop-----------------------------*/" << std::endl;
        }
        else if (ctx_.traffic_flag[Limit_10_Speed]) {
            speed_state_ = SpeedState::LIMIT_SPEED;
            std::cout << "/*------------Limit_10_Speed-----------------------------*/" << std::endl;
        }
        else if (ctx_.traffic_flag[Turn_Left]) {        //左直角转弯
            ctx_.car_state.vx = 0.3;
            ctx_.car_state.vw = 2.0; 
        }
        else if( ctx_.traffic_flag[Turn_Right]) {      //右直角转弯
            ctx_.car_state.vx = 0.3;
            ctx_.car_state.vw = -2.0; 
        }
    }

    void Stop() {
        ctx_.car_state.vx = 0; 
        ctx_.car_state.vw = 0;

        // 人行横道双目标停车优先处理：固定等待3秒后恢复.
        if (cross_road_waiting_) {
            if (cross_road_timer_.Timeout(Config::CROSS_ROAD_STOP_MS)) {
                cross_road_waiting_ = false;
                speed_state_ = SpeedState::FOLLOW_LINE;
                std::cout << "CROSS_ROAD WAIT DONE, RESUME FOLLOW LINE" << std::endl;
            }
            return;
        }

        // 红灯优先级更高：必须绿灯才恢复.
        if (ctx_.traffic_flag[Red_Light]) {
            if (ctx_.traffic_flag[Green_Light]) {
                speed_state_ = SpeedState::FOLLOW_LINE;
                ctx_.traffic_flag[Red_Light] = false;
                ctx_.traffic_count[Red_Light] = 0;
                ctx_.traffic_flag[Green_Light] = false;
                ctx_.traffic_count[Green_Light] = 0;
                std::cout << "GREEN LIGHT RESUME" << std::endl;
            }
            return;
        }

        // 小人消失后恢复前进.
        if (!ctx_.traffic_flag[People]) {
            speed_state_ = SpeedState::FOLLOW_LINE;
            std::cout << "PEOPLE DISAPPEARED, RESUME FOLLOW LINE" << std::endl;
        }
    }

    void LimitSpeed() {
        // 限速状态下也要让红灯/行人具备更高优先级，避免错过停车.
        if (ctx_.traffic_flag[Red_Light]) {
            speed_state_ = SpeedState::STOP;
            std::cout << "/*------------Red_Light-----------------------------*/" << std::endl;
            return;
        }
        if (ctx_.traffic_flag[People]) {
            speed_state_ = SpeedState::STOP;
            std::cout << "/*------------Appear People! Stop-----------------------------*/" << std::endl;
            return;
        }

        ctx_.car_state.vx = std::min(ctx_.car_state.vx, Config::MAX_VX*0.5);
        if (ctx_.traffic_flag[Remove_Limit_Speed]) {
            speed_state_ = SpeedState::FOLLOW_LINE;
            ctx_.traffic_flag[Limit_10_Speed] = false;
            ctx_.traffic_flag[Remove_Limit_Speed] = false;
            std::cout << "REMOVE LIMIT SPEED" << std::endl;
        }
    }

    std::unordered_map<int, TrafficRule> DefaultTrafficRules() const;
    int ClassIdFromName(const std::string& name) const;
    bool LoadTrafficRulesFromFile(const std::string& path, std::unordered_map<int, TrafficRule>& out_rules);
    void InitTrafficRules();
    
    //mp3语音播报函数
    const char* ClassNameById(int class_id) const;
    std::string ShellQuote(const std::string& text) const;
    bool FileExists(const std::string& path) const;
    std::string ResolveAudioRootDir() const;
    void ResolveRoadAudioName();
    void DetectAudioPlayer();
    std::string BuildPlayerCommand(const std::string& abs_file_path) const;
    void InitAudioSystem();
    bool CanPlayNow(int class_id, const std::chrono::steady_clock::time_point& now);
    void PlayAudioAsync(int class_id);
    bool PlayAudioBlockingByName(const std::vector<std::string>& file_names);
    void HandleTrafficAudioEvents();
    void FinalizeMissionAndExit();

    //避障函数
    void UpdateObstacleTrajectoryState(int img_width);
    void ApplyObstacleOffsetToTrajectory(int img_width);

private:
    // 模型
    YOLO11_ROAD_SEG segmentor_;
    ExtractTrajectory extract_trajectory;          
    PD_Controller pd_controller_;
    std::unordered_map<int, TrafficRule> traffic_rules_;
    // 上下文
    DriveContext ctx_;
    // 状态机
    TrajectoryState trajectory_state_ = TrajectoryState::NORMAL;
    SpeedState speed_state_ = SpeedState::FOLLOW_LINE;
    // 转向计时器
    Timer turn_timer_;
    // 冷却时间
    int cooldown_ = 0;
    // 已完成转弯次数（左/右共用）
    int turn_completed_count_ = 0;
    // 小人消失帧计数（防抖）
    int people_miss_frames_ = 0;
    // 锥桶避障（轨迹偏移）
    bool obstacle_traj_active_ = false;
    float obstacle_offset_target_px_ = 0.0f;
    float obstacle_offset_current_px_ = 0.0f;
    int obstacle_detect_frames_ = 0;
    int obstacle_miss_frames_ = 0;
    int obstacle_center_x_ = -1;
    int obstacle_area_ = 0;
    int obstacle_offset_sign_ = 0; // -1: 左偏, 1: 右偏
    int obstacle_cooldown_frames_ = 0;
    bool obstacle_detect_logged_ = false; // 同一轮障碍仅打印一次 DETECTED
    // 人行横道双目标停车
    bool cross_road_waiting_ = false;
    bool cross_road_double_latched_ = false;
    Timer cross_road_timer_;
    // 红灯触发门控：必须先触发过一次解除限速.
    bool remove_limit_seen_ = false;
    // 播报
    std::array<std::string, CLASS_NUM> audio_file_map_;
    std::array<bool, CLASS_NUM> audio_prev_flags_;
    std::array<std::chrono::steady_clock::time_point, CLASS_NUM> audio_last_play_tp_;
    std::array<bool, CLASS_NUM> audio_has_played_;
    std::string audio_root_dir_ = "./mp3/";
    int audio_player_type_ = 0; // 0:none 1:mpg123 2:mplayer 3:ffplay
    // 第二次转弯结束流程
    bool mission_finish_requested_ = false;
    bool mission_finish_handled_ = false;
    int mission_finish_trigger_class_ = -1;
    int mission_finish_trigger_area_ = 0;
    // 动态控制点：初始60，变道后切换到100并保持.
    int trajectory_control_ = 80;
    bool trajectory_control_changed_ = false;

};
