#include "AutoDriveSystem.h"

void AutoDriveSystem::UpdateObstacleTrajectoryState(int img_width) {
    if (obstacle_cooldown_frames_ > 0) {
        obstacle_cooldown_frames_--;
    }

    if (ctx_.traffic_flag[Red_Light] || ctx_.traffic_flag[People]) {
        obstacle_traj_active_ = false;
        obstacle_offset_target_px_ = 0.0f;
    }

    if (!obstacle_traj_active_ &&
        obstacle_detect_frames_ >= Config::OBSTACLE_TRIGGER_FRAMES &&
        obstacle_cooldown_frames_ == 0) {
        obstacle_traj_active_ = true;
        if (obstacle_offset_sign_ == 0) {
            obstacle_offset_sign_ = -1;
        }
        if (!obstacle_detect_logged_) {
            std::cout << "OBSTACLE DETECTED: APPLY TRAJECTORY OFFSET" << std::endl;
            obstacle_detect_logged_ = true;
        }
    }

    if (obstacle_traj_active_) {
        const float abs_offset = Config::OBSTACLE_OFFSET_PX;
        const int center = img_width / 2;
        const int diff = obstacle_center_x_ - center;
        if (diff > Config::OBSTACLE_DIRECTION_DEADZONE_PX) {
            obstacle_offset_sign_ = -1; // 锥桶在右，轨迹左偏
        } else if (diff < -Config::OBSTACLE_DIRECTION_DEADZONE_PX) {
            obstacle_offset_sign_ = 1;  // 锥桶在左，轨迹右偏
        } else if (obstacle_offset_sign_ == 0) {
            obstacle_offset_sign_ = -1;
        }
        obstacle_offset_target_px_ = obstacle_offset_sign_ * abs_offset;
    }

    if (obstacle_traj_active_ && obstacle_miss_frames_ >= Config::OBSTACLE_MISS_FRAMES) {
        obstacle_traj_active_ = false;
        obstacle_offset_target_px_ = 0.0f;
        obstacle_detect_frames_ = 0;
        obstacle_miss_frames_ = 0;
        obstacle_center_x_ = -1;
        obstacle_cooldown_frames_ = Config::OBSTACLE_COOLDOWN_FRAMES;
        obstacle_offset_sign_ = 0;
        obstacle_detect_logged_ = false;
        ctx_.traffic_flag[Dangerous] = false;
        ctx_.traffic_count[Dangerous] = 0;
        std::cout << "OBSTACLE CLEARED: TRAJECTORY RECOVER" << std::endl;
    }

    obstacle_offset_current_px_ += Config::OBSTACLE_OFFSET_ALPHA *
                                   (obstacle_offset_target_px_ - obstacle_offset_current_px_);
    if (std::fabs(obstacle_offset_current_px_) < 1.0f && !obstacle_traj_active_) {
        obstacle_offset_current_px_ = 0.0f;
    }
}

void AutoDriveSystem::ApplyObstacleOffsetToTrajectory(int img_width) {
    auto& traj = ctx_.trajectory.road_trajectory;
    if (traj.empty() || std::fabs(obstacle_offset_current_px_) < 1e-3f) {
        return;
    }

    const int n = static_cast<int>(traj.size());
    for (int i = 0; i < n; ++i) {
        const float s = (n <= 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(n - 1);

        float w = 0.0f;
        if (s < 0.10f) {
            w = 0.0f;
        } else if (s < 0.55f) {
            w = (s - 0.20f) / 0.35f;
        } else if (s < 0.80f) {
            w = 1.0f;
        } else {
            w = (1.0f - s) / 0.20f;
            if (w < 0.0f) {
                w = 0.0f;
            }
        }

        const float shifted_x = static_cast<float>(traj[i].x) + obstacle_offset_current_px_ * w;
        const int x_upper = std::max(Config::OBSTACLE_MARGIN_PX, img_width - Config::OBSTACLE_MARGIN_PX);
        const int clamped_x = std::max(Config::OBSTACLE_MARGIN_PX,
                                       std::min(static_cast<int>(std::lround(shifted_x)), x_upper));
        traj[i].x = clamped_x;
    }
}
