#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

struct Trajectory_params
{
    std::vector<Point> left_points;
    std::vector<Point> right_points;
    std::array<bool, 480> left_losts;
    std::array<bool, 480> right_losts;
    std::vector<Point> road_trajectory;
    std::vector<Point> original_trajectory;
};

// ---------------- 道路检测类 ----------------
class ExtractTrajectory {
public:
    ExtractTrajectory() {}

    // 道路检测主函数
    void Extract(cv::Mat mask, Trajectory_params& trajectory) {
        trajectory.left_points.clear();
        trajectory.right_points.clear();
        trajectory.road_trajectory.clear();
        generateFullTrajectory(mask, trajectory);
        return;
    }

private:
   int generateFullTrajectory(const Mat& mask, Trajectory_params& trajectory) {
        int height = mask.rows;
        int width  = mask.cols;

        trajectory.left_points.reserve(height);
        trajectory.right_points.reserve(height);
        trajectory.road_trajectory.reserve(height);

        for (int i = height - 1; i >= 0; i--)
        {
            int left_border = -1;
            int right_border = -1;

            // ==============================
            // 扫描这一行所有白点
            // ==============================
            for (int j = 20; j < width - 1; j++)
            {
                if (mask.at<uchar>(i, j) > 0)
                {
                    if (left_border == -1)
                        left_border = j;

                    right_border = j;
                }
            }

            // ==============================
            // 过滤无效行
            // ==============================
            if (left_border == -1 || right_border == -1)
                continue;
            
            trajectory.left_points.emplace_back(left_border, i);
            trajectory.right_points.emplace_back(right_border, i);
            trajectory.road_trajectory.emplace_back((left_border + right_border) / 2, i);
        }

        return 0;
    }

};



    




 // ---------------- 轨迹生成函数 ----------------
    // int generateFullTrajectory(const Mat& mask, Trajectory_params& trajectory){
    //     int height = mask.rows;
    //     int width = mask.cols;
    //     int start_col = 20;
    //     int end_col = width - 20;

    //     // 1. 统计每列白点数量
    //     vector<int> white_column(width, 0);
    //     for (int y = height - 1; y >= 0; y--) {
    //         for (int x = start_col; x < end_col; x++) {
    //             if (mask.at<uchar>(y, x) > 0) white_column[x]++;
    //         }
    //     }

    //     // 2. 找左右最长白列
    //     int longest_left[2] = {0, 0};
    //     int longest_right[2] = {0, 0};
    //     for (int j = start_col; j < end_col; j++) {
    //         if (longest_left[0] < white_column[j]) {
    //             longest_left[0] = white_column[j];
    //             longest_left[1] = j;
    //         }
    //     }
    //     for (int j = end_col - 1; j >= start_col; j--) {
    //         if (longest_right[0] < white_column[j]) {
    //             longest_right[0] = white_column[j];
    //             longest_right[1] = j;
    //         }
    //     }

    //     // 3. 搜索范围
    //     int search_stop_line = longest_left[0];
    //     int search_top = max(0, height - search_stop_line);

    //     // 4. 生成轨迹点
    //     for (int i = height - 1; i >= search_top; i--) {
    //         // 搜索左边界
    //         int left_border = start_col;
    //         for (int j = longest_left[1]; j >= start_col; j--) {
    //             if (mask.at<uchar>(i, j) > 0 &&
    //                 mask.at<uchar>(i, j - 1) > 0 &&
    //                 mask.at<uchar>(i, j - 2) == 0) {
    //                 left_border = j-1;
    //                 trajectory.left_losts[i] = false;
    //                 break;
    //             }
    //         }

    //         // 搜索右边界
    //         int right_border = end_col;
    //         for (int j = longest_right[1]; j <= end_col; j++) {
    //             if (mask.at<uchar>(i, j) > 0 &&
    //                 mask.at<uchar>(i, j + 1) > 0 &&
    //                 mask.at<uchar>(i, j + 2) == 0) {
    //                 right_border = j+1;
    //                 trajectory.right_losts[i] = false;
    //                 break;
    //             }
    //         }

    //         trajectory.left_points.push_back(Point(left_border, i));
    //         trajectory.right_points.push_back(Point(right_border, i));
    //         trajectory.road_trajectory.push_back(Point((left_border + right_border) / 2, i));
    //     }

    //     return 0;
    // }



///      
        /*⬇--------------------进入十字路口的右上角点检测--------------------⬇*/
        // int Right_Up_Find = -1;                 //右侧角点的y值
        // int Left_Up_Find = -1;                  //左侧角点的y值

        // //统计有边界丢线的数量
        // int right_lost_count = count(right_lost.begin()+50, right_lost.end(), true);
        //     // std::cout << "Right lost count: " << right_lost_count  << std::endl;
        
        // if(right_lost_count >= 300) {                 //如果右边界丢线数量较多，说明可能进入了十字路口，进行右侧角点检测
        //     for(int i = height -1 ;i >= search_top; i--) {           //从图像底部向上寻找
        //         int border_idx = height- 1- i;                      // 因为left_points和right_points是从下往上存的，所以索引需要转换
        //         //进行左侧角点检测                  //上册平稳，下册撕裂
        //         if (Left_Up_Find == -1 && border_idx >= 5 && border_idx <= left_points.size() - 5)
        //         {
        //             int curr = left_points[border_idx].x;                  //此处防止leftpoint益处越界，idx需要大于3且小于left_points.size() - 3
        //             int up1  = left_points[border_idx + 3].x;              //当前点的上一个点              
        //             int up2  = left_points[border_idx + 4].x;              //当前点的上二个点
        //             int up3  = left_points[border_idx + 5].x;              //当前点的上三个点
        //             int dn1  = left_points[border_idx - 3].x;              //当前点的下一个点
        //             int dn2  = left_points[border_idx - 4].x;              //当前点的下两个点
        //             int dn3  = left_points[border_idx - 5].x;              //当前点的下三个点
        //             if( (abs(curr - up1) <= 6 && abs(curr - up2) <= 10 && abs(curr - up3) <= 15) &&
        //                 ((curr - dn1) >= 10 && (curr - dn2) >= 15 && (curr - dn3) >= 15))
        //             {
        //                 Left_Up_Find = i;                                  //记录第i行
        //             }
        //         }
        //         //进行右侧角点检测
        //         if (Right_Up_Find == -1 && border_idx >= 5 && border_idx <= right_points.size() - 5)
        //         {
        //             int curr = right_points[border_idx].x;                  //此处防止rightpoint益处越界，idx需要大于3且小于right_points.size() - 3
        //             int up1  = right_points[border_idx + 3].x;              //当前点的上一个点              
        //             int up2  = right_points[border_idx + 4].x;              //当前点的上二个点
        //             int up3  = right_points[border_idx + 5].x;              //当前点的上三个点
        //             int dn1  = right_points[border_idx - 3].x;              //当前点的下一个点
        //             int dn2  = right_points[border_idx - 4].x;              //当前点的下两个点
        //             int dn3  = right_points[border_idx - 5].x;              //当前点的下三个点
        //             if( (abs(curr - up1) <= 6 && abs(curr - up2) <= 6 && abs(curr - up3) <= 10) &&
        //                 ((curr - dn1) <= -10 && (curr - dn2) <= -15 && (curr - dn3) <= -15) )
        //             {
        //                 Right_Up_Find = i;                                  //记录第i行
        //             }
        //         }
        //         if(Left_Up_Find != -1 && Right_Up_Find != -1) {
        //             break;
        //         }
        //     }
        //     if(Left_Up_Find != -1){
        //         // cv::circle(original_image, Point(left_points[height-1-Left_Up_Find]), 10, Scalar(255, 0, 0), -1);   // 左上角点
        //         //补线显示
        //         // cv::line(original_image, Point(left_points[height-1-Left_Up_Find]), Point(left_points[height-1-Left_Up_Find+3]), Scalar(255, 0, 0), 5);
        //         //计算斜率画出整个一条直线而不是线段        有报错说是浮点数例外，可能是因为除数为0了，说明这条线是垂直的，我们直接画一条竖线就行了

        //         // double slope = (left_points[height-1-Left_Up_Find+3].y - left_points[height-1-Left_Up_Find].y) /
        //         //                (left_points[height-1-Left_Up_Find+3].x - left_points[height-1-Left_Up_Find].x);
        //         // if(isinf(slope)) {  // 斜率无穷大，说明是垂直线
        //         //     for (int y = left_points[height-1-Left_Up_Find].y; y >= left_points[height-1-Left_Up_Find+3].y; y--) {
        //         //         cv::circle(original_image, Point(left_points[height-1-Left_Up_Find].x, y), 1, Scalar(255, 0, 0), -1);
        //         //     }
        //         // } else {
        //         //     double intercept = left_points[height-1-Left_Up_Find].y - slope * left_points[height-1-Left_Up_Find].x;
        //         //     for (int x = left_points[height-1-Left_Up_Find].x; x <= left_points[height-1-Left_Up_Find+3].x; x++) {
        //         //         int y = slope * x + intercept;
        //         //         cv::circle(original_image, Point(x, y), 1, Scalar(255, 0, 0), -1);
        //         //     }
        //         // }

        //         // std::cout << "Left_Up_Find: " << Left_Up_Find << " Right_Up_Find: " << Right_Up_Find << std::endl;
        //         std::cout <<std::endl;
        //     }
        //     if(Right_Up_Find != -1 )
        //     {   
        //     // std::cout << "Right lost count: " << right_lost_count << "        sssssssssssssss" << std::endl;

        //         cv::circle(original_image, Point(right_points[height-1-Right_Up_Find]), 10, Scalar(0, 255, 0), -1); // 右上角点
        //         //补线显示
        //         // cv::line(original_image, Point(right_points[height-1-Right_Up_Find]), Point(right_points[height-1-Right_Up_Find+3]), Scalar(0, 255, 0), 5);

        //         //计算斜率画出整个一条直线而不是线段
        //         double dx = right_points[height-1-Right_Up_Find+3].x - right_points[height-1-Right_Up_Find].x;
        //         if(dx == 0) {  // 斜率无穷大，说明是垂直线
        //             for (int y = right_points[height-1-Right_Up_Find].y; y <= right_points[0].y; y++) {
        //                 cv::circle(original_image, Point(right_points[height-1-Right_Up_Find].x, y), 1, Scalar(0, 255, 0), -1);
        //                 std::cout << "dx == 0: " << dx << std::endl;
        //             }
        //         } else {
        //             double slope = (right_points[height-1-Right_Up_Find+5].y - right_points[height-1-Right_Up_Find+3].y) / dx;
        //             double intercept = right_points[height-1-Right_Up_Find+3].y - slope * right_points[height-1-Right_Up_Find+3].x;
        //             //反了，应该遍历y值，求出x值
        //             for (int y = right_points[height-1-Right_Up_Find].y; y <= right_points[0].y; y++) {
        //                 int x = (y - intercept) / slope;
        //                 cv::circle(original_image, Point(x, y), 3, Scalar(0, 255, 0), -1);
        //                 std::cout << "y " << y << std::endl;
        //                 std::cout << "right_points_now: " << right_points[height-1-Right_Up_Find] << " right_points: " << right_points[0] << std::endl;
        //             }
        //         }
                
        //         std::cout << "Right_Up_Find: " << Right_Up_Find << " Left_Up_Find: " << Left_Up_Find << std::endl;
        //         std::cout <<std::endl;
        //     }
        // }
        //进行补线操作
