#include "postprocess.h"
#include "utils.h"

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);

    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (kInputH - r_w * img.rows) / 2;
        b = bbox[3] - (kInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (kInputW - r_h * img.cols) / 2;
        r = bbox[2] - (kInputW - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

cv::Rect get_rect_adapt_landmark(cv::Mat& img, float bbox[4], float lmk[kNumberOfPoints * 3]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (kInputH - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (kInputH - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < kNumberOfPoints * 3; i += 3) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (kInputH - r_w * img.rows) / 2) / r_w;
            // lmk[i + 2]
        }
    } else {
        l = (bbox[0] - (kInputW - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (kInputW - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < kNumberOfPoints * 3; i += 3) {
            lmk[i] = (lmk[i] - (kInputW - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
            // lmk[i + 2]
        }
    }
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0], rbox[0]),
            (std::min)(lbox[2], rbox[2]),
            (std::max)(lbox[1], rbox[1]),
            (std::min)(lbox[3], rbox[3]),
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionBoxS = (lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS;
    return interBoxS / unionBoxS;
}

static bool cmp(const Detection& a, const Detection& b) {
    if (a.conf == b.conf) {
        return a.bbox[0] < b.bbox[0];
    }
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh)
            continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void batch_nms(std::vector<Detection>& res_batch, float* output, int batch_size, int output_size,
               float conf_thresh, float nms_thresh) {

    nms(res_batch, &output[0 * output_size], conf_thresh, nms_thresh);
}

void process_decode_ptr_host(std::vector<Detection>& res, const float* decode_ptr_host, int bbox_element, cv::Mat& img,
                             int count) {
    Detection det;
    for (int i = 0; i < count; i++) {
        int basic_pos = 1 + i * bbox_element;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1) {
            det.bbox[0] = decode_ptr_host[basic_pos + 0];
            det.bbox[1] = decode_ptr_host[basic_pos + 1];
            det.bbox[2] = decode_ptr_host[basic_pos + 2];
            det.bbox[3] = decode_ptr_host[basic_pos + 3];
            det.conf = decode_ptr_host[basic_pos + 4];
            det.class_id = decode_ptr_host[basic_pos + 5];
            res.push_back(det);
        }
    }
}

void batch_process(std::vector<Detection>& res_batch, const float* decode_ptr_host, int batch_size,
                   int bbox_element, cv::Mat& img) {
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, kMaxNumOutputBbox);
    process_decode_ptr_host(res_batch, &decode_ptr_host[0 * count], bbox_element, img, count);
}

cv::Scalar getColor(int id) {
    switch(id) {
        case 1: return cv::Scalar(255, 0, 0);      // 蓝
        case 2: return cv::Scalar(0, 255, 0);      // 绿
        case 3: return cv::Scalar(0, 0, 255);      // 红
        case 4: return cv::Scalar(255, 255, 0);    // 青
        case 5: return cv::Scalar(0, 255, 255);    // 黄
        case 6: return cv::Scalar(255, 0, 255);    // 紫
        case 7: return cv::Scalar(0, 128, 255);    // 橙
        case 8: return cv::Scalar(255, 255, 0);    // 浅青
        case 9: return cv::Scalar(255, 0, 128);    // 粉
        default: return cv::Scalar(128, 128, 128); // 灰
    }
}


enum Class_Typical {
    road,
    Turn_Road ,
    Attention ,
    Green_Light,
    Red_Light,
    Passage_Road ,
    Turn_Left ,
    Turn_Right,
    Remove_Limit_Speed,
    Limit_Speed,
    Person,
    zhuitong
};

std::string get_Name(int id) {
    switch(id) {
        case Red_Light: return "Red_Light";
        case Turn_Road: return "Turn_Road";
        case Limit_Speed: return "Limit_Speed";
        case Passage_Road: return "cross_the_road";
        case Attention: return "Attention";
        case Turn_Left: return "Turn_Left";
        case Turn_Right: return "Turn_Right";
        case Remove_Limit_Speed: return "Remove_Limit_Speed";
        case Green_Light: return "Green_Light";
        case Person: return "Person";
        case zhuitong: return "zhuitong";
        default: return "";
    }
}

void draw_bbox(cv::Mat& img, std::vector<Detection>& res_batch) {
    auto& res = res_batch;
    for (size_t j = 0; j < res.size(); j++) {
        cv::Rect r = get_rect(img, res[j].bbox);
        int id_num = (int)res[j].class_id;
        res[j].mianji = r.height * r.width;
        cv::rectangle(img, r, getColor(id_num), 2);
        std::string label = get_Name(id_num) + " " + to_string_with_precision(res[j].mianji) + "  " + to_string_with_precision(res[j].conf);
        // Get the size of the text
        cv::putText(img, label, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.4,
                    getColor(id_num), 2);
    }
}

cv::Mat scale_mask(cv::Mat mask, cv::Mat img) {
    int x, y, w, h;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    if (r_h > r_w) {
        w = kInputW;
        h = r_w * img.rows;
        x = 0;
        y = (kInputH - h) / 2;
    } else {
        w = r_h * img.cols;
        h = kInputH;
        x = (kInputW - w) / 2;
        y = 0;
    }
    cv::Rect r(x, y, w, h);
    cv::Mat res;
    cv::resize(mask(r), res, img.size());
    return res;
}

void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks, cv::Mat& yellow_mask_out) {
    static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17,
                                           0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF,
                                           0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};

    // 初始化输出掩码：CV_8UC1，全黑（复用已有内存）
    yellow_mask_out.create(img.rows, img.cols, CV_8UC1);
    yellow_mask_out.setTo(0);

    // 预先计算 letterbox 映射参数（将原图坐标映射到 kInputW x kInputH 坐标）
    const float r_w = kInputW / (img.cols * 1.0f);
    const float r_h = kInputH / (img.rows * 1.0f);
    float sx = 0.0f, sy = 0.0f, sw = 0.0f, sh = 0.0f;
    if (r_h > r_w) {
        sw = static_cast<float>(kInputW);
        sh = r_w * img.rows;
        sx = 0.0f;
        sy = (kInputH - sh) * 0.5f;
    } else {
        sw = r_h * img.cols;
        sh = static_cast<float>(kInputH);
        sx = (kInputW - sw) * 0.5f;
        sy = 0.0f;
    }

    cv::Mat resized_mask, bin_mask, overlay, blended;
    for (size_t i = 0; i < dets.size(); i++) {
        if (i >= masks.size()) {
            break;
        }

        cv::Rect r = get_rect(img, dets[i].bbox);
        r &= cv::Rect(0, 0, img.cols, img.rows);
        if (r.width <= 0 || r.height <= 0) {
            continue;
        }

        const int mask_w = masks[i].cols;
        const int mask_h = masks[i].rows;
        if (mask_w <= 0 || mask_h <= 0) {
            continue;
        }
        const float scale_x = static_cast<float>(mask_w) / kInputW;
        const float scale_y = static_cast<float>(mask_h) / kInputH;

        // 将原图 ROI 反映射到 mask 对应 letterbox ROI，只做局部 resize
        float mx0_f = (sx + (r.x * sw) / img.cols) * scale_x;
        float my0_f = (sy + (r.y * sh) / img.rows) * scale_y;
        float mx1_f = (sx + ((r.x + r.width) * sw) / img.cols) * scale_x;
        float my1_f = (sy + ((r.y + r.height) * sh) / img.rows) * scale_y;

        int mx0 = std::max(0, std::min(mask_w - 1, static_cast<int>(floorf(mx0_f))));
        int my0 = std::max(0, std::min(mask_h - 1, static_cast<int>(floorf(my0_f))));
        int mx1 = std::max(mx0 + 1, std::min(mask_w, static_cast<int>(ceilf(mx1_f))));
        int my1 = std::max(my0 + 1, std::min(mask_h, static_cast<int>(ceilf(my1_f))));

        cv::Rect mask_roi(mx0, my0, mx1 - mx0, my1 - my0);
        if (mask_roi.width <= 0 || mask_roi.height <= 0) {
            continue;
        }

        cv::resize(masks[i](mask_roi), resized_mask, r.size(), 0, 0, cv::INTER_LINEAR);
        cv::compare(resized_mask, 0.5f, bin_mask, cv::CMP_GT);  // CV_8UC1, 0 or 255
        if (cv::countNonZero(bin_mask) == 0) {
            continue;
        }

        auto color = colors[static_cast<int>(dets[i].class_id) % colors.size()];
        auto bgr = cv::Scalar(color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF);
        cv::Mat img_roi = img(r);

        overlay.create(r.height, r.width, CV_8UC3);
        overlay.setTo(bgr);
        cv::addWeighted(img_roi, 0.5, overlay, 0.5, 0.0, blended);
        blended.copyTo(img_roi, bin_mask);

        yellow_mask_out(r).setTo(255, bin_mask);
    }
}

cv::Mat ExtractRoadMask(const cv::Mat& img, const std::vector<cv::Mat>& masks) {
    cv::Mat prob = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

    for (const auto& m : masks) {
        cv::Mat mm = scale_mask(m, img);   // 你的现有函数，映射回原图尺寸
        cv::max(prob, mm, prob);           // 多个 road 实例做并集（取最大概率）
    }

    cv::Mat road_mask_u8;
    cv::threshold(prob, road_mask_u8, 0.5f, 255.0f, cv::THRESH_BINARY);
    road_mask_u8.convertTo(road_mask_u8, CV_8UC1);
    return road_mask_u8;
}
