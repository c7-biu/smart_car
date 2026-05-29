#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

typedef struct {
    double Kp, Kd;
    double error, last_error, derivative;
    double output;
    double out_min, out_max;
    uint32_t dt_ms;
} PD_Controller;

struct CarState { double vx; double vw; };

const double a_max = 0.35;  // 最大向心加速度 m/s² a = v*w

// ---------------- PD角度环控制器 ----------------
inline void PD_Init(PD_Controller* pd, double kp, double kd, double out_min, double out_max, uint32_t dt_ms);
inline double constrain_f(double val, double min, double max);
inline double PD_Calculate(PD_Controller* pd, double setpoint, double feedback);
inline double V_planning(double omega, double V_max);
inline void computeSimple(int img_width,
                          CarState& car_vec,
                          const vector<Point>& road_trajectory,
                          PD_Controller& pd_controller,
                          double V_max,
                          int trajectory_control);


// ---------------- 速度计算 ----------------
inline void computeSimple(int img_width,
                   CarState& car_vec,
                   const vector<Point>& road_trajectory,
                   PD_Controller& pd_controller,
                   double V_max,
                   int trajectory_control)
{
    if (road_trajectory.size() == 0) {
        car_vec.vx = 0;
        car_vec.vw = 0;
        return;
    }

    // ================== ① 转向控制（只用单点）==================
    int ctrl_idx = std::min((int)road_trajectory.size() - 1, trajectory_control);
    int center_x = road_trajectory[ctrl_idx].x;

    int error = center_x - img_width / 2;

    car_vec.vw = PD_Calculate(&pd_controller, 0, error);

    // ================== ② 基础速度 ==================
    static double last_vx = 0.0;
    double alpha = 0;

    double V_target = V_planning(car_vec.vw, V_max);

    if (V_target < last_vx)
        alpha = 0.9;
    else
        alpha = 0.5;

    car_vec.vx = alpha * V_target + (1 - alpha) * last_vx;
    last_vx = car_vec.vx;

    // ================== ③ 弯道检测（只影响速度，不影响转向）==================

    int far_idx = trajectory_control + 120;
    if (far_idx >= road_trajectory.size()) {
        far_idx = road_trajectory.size() - 1;
    }

    int far_x = road_trajectory[far_idx].x;

    double far_error = fabs(far_x - img_width / 2);

    // 阈值（要调）
    double curve_threshold = 80.0;

    if (far_error > curve_threshold) {
        double ratio = far_error / (img_width / 2.0);
        ratio = std::min(ratio, 1.0);
        double k = 1.0 - 0.6 * ratio;
        car_vec.vx *= k;
    }

    // 防止速度过低
    car_vec.vx = std::max(car_vec.vx, 0.2);
}

// ---------------- 速度规划 ----------------
inline double V_planning(double omega, double V_max)
{
    double V_safe = (fabs(omega) < 1e-3) ? V_max : a_max / fabs(omega);
    double V = min(V_safe, V_max);
    if (V < 0.05) V = 0.0;  // 极端弯道停车
    return V;
}

inline double PD_Calculate(PD_Controller* pd, double setpoint, double feedback)
{
    pd->error = setpoint - feedback;
    double raw_d = (pd->error - pd->last_error) / (pd->dt_ms / 1000.0);
    pd->derivative = 0.5 * raw_d + 0.5 * pd->derivative;
    pd->output = pd->Kp * pd->error + pd->Kd * pd->derivative;
    pd->output = constrain_f(pd->output, pd->out_min, pd->out_max);
    pd->last_error = pd->error;
    return pd->output;
}

inline void PD_Init(PD_Controller* pd, double kp, double kd, 
             double out_min, double out_max, uint32_t dt_ms)
{
    pd->Kp = kp; pd->Kd = kd;
    pd->error = pd->last_error = pd->derivative = pd->output = 0;
    pd->out_min = out_min; pd->out_max = out_max;
    pd->dt_ms = dt_ms;
}

inline double constrain_f(double val, double min, double max)
{
    if (val < min) return min;
    if (val > max) return max;
    return val;
}
