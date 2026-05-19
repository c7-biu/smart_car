#include "AutoDriveSystem.h"

std::unordered_map<int, TrafficRule> AutoDriveSystem::DefaultTrafficRules() const {
    return {
        {Turn_Left, {Turn_Left, 1900, 5, true, 50, true, 3000}},
        {Turn_Right, {Turn_Right, 1900, 5, true, 50, true, 3000}},
        {Limit_10_Speed, {Limit_10_Speed, 1000, 5, true, 50, false, 6000}},
        {Remove_Limit_Speed, {Remove_Limit_Speed, 7500, 5, true, 100, false, 0}},
        {Red_Light, {Red_Light, 1000, 5, true, 50, false, 2000}},
        {Green_Light, {Green_Light, 1000, 5, true, 50, false, 2000}},
        {People, {People, 20000, 1, false, 0, true, 0}},
        {Dangerous, {Dangerous, 1200, 2, false, 0, true, 3500}},
    };
}

int AutoDriveSystem::ClassIdFromName(const std::string& name) const {
    static const std::unordered_map<std::string, int> kNameToId = {
        {"Road", Road},
        {"Change_Lanes", Change_Lanes},
        {"Warning_Sign", Warning_Sign},
        {"Green_Light", Green_Light},
        {"Red_Light", Red_Light},
        {"Cross_Road", Cross_Road},
        {"Turn_Left", Turn_Left},
        {"Turn_Right", Turn_Right},
        {"Remove_Limit_Speed", Remove_Limit_Speed},
        {"Limit_10_Speed", Limit_10_Speed},
        {"People", People},
        {"Dangerous", Dangerous}
    };
    const auto it = kNameToId.find(name);
    if (it == kNameToId.end()) {
        return -1;
    }
    return it->second;
}

bool AutoDriveSystem::LoadTrafficRulesFromFile(
    const std::string& path, std::unordered_map<int, TrafficRule>& out_rules) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    cv::FileNode rules_node = fs["traffic_rules"];
    if (rules_node.empty() || rules_node.type() != cv::FileNode::SEQ) {
        return false;
    }

    std::unordered_map<int, TrafficRule> loaded;
    for (cv::FileNodeIterator it = rules_node.begin(); it != rules_node.end(); ++it) {
        cv::FileNode node = *it;
        if (node.type() != cv::FileNode::MAP) {
            continue;
        }

        int class_id = -1;
        cv::FileNode id_node = node["class_id"];
        if (!id_node.empty()) {
            if (id_node.isInt()) {
                class_id = static_cast<int>(id_node);
            } else if (id_node.isString()) {
                class_id = ClassIdFromName(static_cast<std::string>(id_node));
            }
        }
        if (class_id < 0 || class_id >= CLASS_NUM) {
            continue;
        }

        TrafficRule rule;
        rule.class_id = class_id;
        node["min_area"] >> rule.min_area;
        node["trigger_count"] >> rule.trigger_count;
        node["need_center_check"] >> rule.need_center_check;
        node["center_threshold"] >> rule.center_threshold;
        node["center_inside"] >> rule.center_inside;
        node["decision_area"] >> rule.decision_area;
        loaded[class_id] = rule;
    }

    if (loaded.empty()) {
        return false;
    }

    out_rules.swap(loaded);
    return true;
}

void AutoDriveSystem::InitTrafficRules() {
    traffic_rules_ = DefaultTrafficRules();
    const std::vector<std::string> candidate_paths = {
        "./config/traffic_rules.yaml",
        "./../config/traffic_rules.yaml",
        "/home/jd/a/yolo11/config/traffic_rules.yaml"
    };

    for (size_t i = 0; i < candidate_paths.size(); ++i) {
        std::unordered_map<int, TrafficRule> loaded_rules;
        if (LoadTrafficRulesFromFile(candidate_paths[i], loaded_rules)) {
            traffic_rules_.swap(loaded_rules);
            // YAML 未配置的类别回退到默认规则（例如 Dangerous）
            const auto defaults = DefaultTrafficRules();
            for (auto it = defaults.begin(); it != defaults.end(); ++it) {
                if (traffic_rules_.find(it->first) == traffic_rules_.end()) {
                    traffic_rules_[it->first] = it->second;
                }
            }
            std::cout << "traffic rules loaded from: " << candidate_paths[i] << std::endl;
            return;
        }
    }

    std::cout << "traffic rules config not found/invalid, use defaults." << std::endl;
}
