#include "AutoDriveSystem.h"

#include <cstdlib>
#include <fstream>
#include <thread>

const char* AutoDriveSystem::ClassNameById(int class_id) const {
    static const std::array<const char*, CLASS_NUM> kClassNames = {{
        "Road",
        "Change_Lanes",
        "Warning_Sign",
        "Green_Light",
        "Red_Light",
        "Cross_Road",
        "Turn_Left",
        "Turn_Right",
        "Remove_Limit_Speed",
        "Limit_10_Speed",
        "People",
        "Dangerous",
        "People_disappear"
    }};
    if (class_id < 0 || class_id >= CLASS_NUM) {
        return "";
    }
    return kClassNames[class_id];
}

std::string AutoDriveSystem::ShellQuote(const std::string& text) const {
    std::string out = "'";
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\'') {
            out += "'\\''";
        } else {
            out.push_back(text[i]);
        }
    }
    out += "'";
    return out;
}

bool AutoDriveSystem::FileExists(const std::string& path) const {
    std::ifstream ifs(path.c_str());
    return ifs.good();
}

std::string AutoDriveSystem::ResolveAudioRootDir() const {
    const std::vector<std::string> candidate_dirs = {
        "./mp3/",
        "./../mp3/",
        "/home/jd/a/yolo11/mp3/"
    };
    for (size_t i = 0; i < candidate_dirs.size(); ++i) {
        if (FileExists(candidate_dirs[i] + "Turn_Left.mp3")) {
            return candidate_dirs[i];
        }
    }
    return "./mp3/";
}

void AutoDriveSystem::ResolveRoadAudioName() {
    const std::string lower_name = "road.mp3";
    const std::string upper_name = "Road.mp3";
    if (FileExists(audio_root_dir_ + lower_name)) {
        audio_file_map_[Road] = lower_name;
    } else if (FileExists(audio_root_dir_ + upper_name)) {
        audio_file_map_[Road] = upper_name;
    }
}

void AutoDriveSystem::DetectAudioPlayer() {
    if (std::system("command -v mpg123 >/dev/null 2>&1") == 0) {
        audio_player_type_ = 1;
    } else if (std::system("command -v mplayer >/dev/null 2>&1") == 0) {
        audio_player_type_ = 2;
    } else if (std::system("command -v ffplay >/dev/null 2>&1") == 0) {
        audio_player_type_ = 3;
    } else {
        audio_player_type_ = 0;
    }
}

std::string AutoDriveSystem::BuildPlayerCommand(const std::string& abs_file_path) const {
    const std::string qpath = ShellQuote(abs_file_path);
    if (audio_player_type_ == 1) {
        return "mpg123 -q " + qpath + " >/dev/null 2>&1";
    }
    if (audio_player_type_ == 2) {
        return "mplayer -really-quiet " + qpath + " >/dev/null 2>&1";
    }
    if (audio_player_type_ == 3) {
        return "ffplay -nodisp -autoexit -loglevel quiet " + qpath + " >/dev/null 2>&1";
    }
    return "";
}

void AutoDriveSystem::InitAudioSystem() {
    audio_file_map_.fill("");
    audio_prev_flags_.fill(false);
    audio_has_played_.fill(false);

    for (int class_id = 0; class_id < CLASS_NUM; ++class_id) {
        const std::string class_name = ClassNameById(class_id);
        if (!class_name.empty() && class_id != People_disappear) {
            audio_file_map_[class_id] = class_name + ".mp3";
        }
    }

    audio_root_dir_ = ResolveAudioRootDir();
    ResolveRoadAudioName();

    // 文件名兼容：优先使用新命名，其次回退旧命名.
    auto resolve_alias = [this](const std::vector<std::string>& names) -> std::string {
        for (const auto& name : names) {
            if (FileExists(audio_root_dir_ + name)) {
                return name;
            }
        }
        return "";
    };

    const std::string change_lanes_name = resolve_alias({
        "Change_lanes.mp3",
        "Change_Lanes.mp3",
        "change_lanes.mp3"
    });
    if (!change_lanes_name.empty()) {
        audio_file_map_[Change_Lanes] = change_lanes_name;
    }

    const std::string warning_sign_name = resolve_alias({
        "Dangerous.mp3",
        "dangerous.mp3"
    });
    if (!warning_sign_name.empty()) {
        audio_file_map_[Warning_Sign] = warning_sign_name;
    }

    DetectAudioPlayer();

    if (audio_player_type_ == 0) {
        std::cout << "[AUDIO] no player found. install mpg123/mplayer/ffplay." << std::endl;
    } else {
        std::cout << "[AUDIO] root: " << audio_root_dir_ << std::endl;
    }
}

void AutoDriveSystem::PlayRoadAudioBlocking() {
    const std::string& mapped = audio_file_map_[Road];
    std::vector<std::string> candidates;
    if (!mapped.empty()) {
        candidates.push_back(mapped);
    }
    candidates.push_back("road.mp3");
    candidates.push_back("Road.mp3");
    const bool played = PlayAudioBlockingByName(candidates);
    if (!played) {
        std::cout << "[AUDIO] road audio not found (road.mp3 / Road.mp3)." << std::endl;
    }
}

bool AutoDriveSystem::CanPlayNow(int class_id) {
    if (class_id < 0 || class_id >= CLASS_NUM) {
        return false;
    }
    if (audio_has_played_[class_id]) {
        return false;
    }
    audio_has_played_[class_id] = true;
    return true;
}

void AutoDriveSystem::PlayAudioAsync(int class_id) {
    if (audio_player_type_ == 0 || class_id < 0 || class_id >= CLASS_NUM) {
        return;
    }

    const std::string& file_name = audio_file_map_[class_id];
    if (file_name.empty()) {
        return;
    }
    const std::string path = audio_root_dir_ + file_name;
    if (!FileExists(path)) {
        std::cout << "[AUDIO] missing file: " << path << std::endl;
        return;
    }

    const std::string cmd = BuildPlayerCommand(path);
    if (cmd.empty()) {
        return;
    }
    std::thread([cmd]() {
        const int ret = std::system(cmd.c_str());
        (void)ret;
    }).detach();
}

bool AutoDriveSystem::PlayAudioBlockingByName(const std::vector<std::string>& file_names) {
    if (audio_player_type_ == 0) {
        std::cout << "[AUDIO] no player for blocking play." << std::endl;
        return false;
    }
    for (const auto& name : file_names) {
        const std::string path = audio_root_dir_ + name;
        if (!FileExists(path)) {
            continue;
        }
        const std::string cmd = BuildPlayerCommand(path);
        if (cmd.empty()) {
            continue;
        }
        const int ret = std::system(cmd.c_str());
        (void)ret;
        return true;
    }
    return false;
}

void AutoDriveSystem::HandleTrafficAudioEvents() {
    for (int class_id = 0; class_id < CLASS_NUM; ++class_id) {
        const bool flag = ctx_.traffic_flag[class_id];
        if (flag && !audio_prev_flags_[class_id] && CanPlayNow(class_id)) {
            PlayAudioAsync(class_id);
        }
        audio_prev_flags_[class_id] = flag;
    }
}

void AutoDriveSystem::FinalizeMissionAndExit() {
    if (mission_finish_handled_) {
        return;
    }
    mission_finish_handled_ = true;

    // 立即停车并下发到下位机.
    ctx_.car_state.vx = 0.0;
    ctx_.car_state.vw = 0.0;
    speed_state_ = SpeedState::STOP;
    send_struct.x_vel_target = 0;
    send_struct.z_vel_target = 0;
    bluetooth_send();

    std::cout << "[MISSION] FINISH FLOW START: trigger="
              << ClassNameById(mission_finish_trigger_class_)
              << " area=" << mission_finish_trigger_area_
              << std::endl;

    const bool played = PlayAudioBlockingByName({"finish.mp3", "Finish.mp3"});
    if (!played) {
        std::cout << "[MISSION] finish audio not found (finish.mp3 / Finish.mp3)." << std::endl;
    } else {
        std::cout << "[MISSION] finish audio played. exit now." << std::endl;
    }

    std::exit(0);
}
