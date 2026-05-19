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
    DetectAudioPlayer();

    if (audio_player_type_ == 0) {
        std::cout << "[AUDIO] no player found. install mpg123/mplayer/ffplay." << std::endl;
    } else {
        std::cout << "[AUDIO] root: " << audio_root_dir_ << std::endl;
        PlayAudioAsync(Road);
    }
}

bool AutoDriveSystem::CanPlayNow(int class_id, const std::chrono::steady_clock::time_point& now) {
    if (class_id < 0 || class_id >= CLASS_NUM) {
        return false;
    }
    if (!audio_has_played_[class_id]) {
        audio_last_play_tp_[class_id] = now;
        audio_has_played_[class_id] = true;
        return true;
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - audio_last_play_tp_[class_id]).count();
    if (elapsed_ms < Config::AUDIO_MIN_INTERVAL_MS) {
        return false;
    }
    audio_last_play_tp_[class_id] = now;
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

void AutoDriveSystem::HandleTrafficAudioEvents() {
    const auto now = std::chrono::steady_clock::now();
    for (int class_id = 0; class_id < CLASS_NUM; ++class_id) {
        const bool flag = ctx_.traffic_flag[class_id];
        if (flag && !audio_prev_flags_[class_id] && CanPlayNow(class_id, now)) {
            PlayAudioAsync(class_id);
        }
        audio_prev_flags_[class_id] = flag;
    }
}
