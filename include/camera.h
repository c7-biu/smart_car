#pragma once
#include <unistd.h>
#include <error.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <pthread.h>
#include <linux/videodev2.h>
// #include "opencv2/imgcodecs/legacy/constants_c.h"
#include <sys/mman.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <mutex>

// #include "common.h"
// using namespace std;
// using namespace cv;

#define CLEAR(x) memset(&(x), 0, sizeof(x))

#define IMAGEWIDTH 640
#define IMAGEHEIGHT 480

class V4L2Capture {
public:
    struct CameraConfig {
        int brightness = 128;
        int contrast = 50;
        int saturation = 60;
        int hue = 0;
        int sharpness = 50;
        int gamma = 120;
        int gain = 0;
        bool auto_white_balance = false;
        int white_balance_temperature = 4600;
        bool manual_exposure = true;
        int exposure_absolute = 95;
        int frame_rate_numerator = 1;
        int frame_rate_denominator = 30;
        bool enable_tuning = false;
    };

    V4L2Capture();
	V4L2Capture(const char *devName, int width, int height);
	virtual ~V4L2Capture();
	int openDevice();
	int closeDevice();
	int initDevice();
	int startCapture();
	int stopCapture();
	int freeBuffers();
	int getFrame(void **,size_t *);
	int backFrame();
	static void test();
	int InitCamera();
	cv::Mat ResultCamera();
	struct cam_buffer
	{
		void* start;
		unsigned int length;
	};
private:
	int initBuffers();
	// V4L2Capture* vcap;
	int fd_cam;
	const char *devName;
	int capW;
	int capH;
	cam_buffer *buffers;
	unsigned int n_buffers;
	int frameIndex;
	unsigned char *yuv422frame = NULL;
	unsigned long yuvframeSize = 0;
};
