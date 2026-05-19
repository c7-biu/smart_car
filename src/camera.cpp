#include "camera.h"
#include <cctype>
#include <cstdlib>
#include <fstream>
// V4L2Capture vcap;
static unsigned int *fb_base = NULL;                        /* Framebuffer映射基地址 */
static int screen_size;                                      /* 整个Framebuffer大小*/
int lcd_w = 800 ,lcd_h= 480;

namespace {
using CameraParams = V4L2Capture::CameraConfig;

std::string Trim(const std::string& s) {
	const char* ws = " \t\r\n";
	const auto begin = s.find_first_not_of(ws);
	if (begin == std::string::npos) return "";
	const auto end = s.find_last_not_of(ws);
	return s.substr(begin, end - begin + 1);
}

std::string ToLower(std::string s) {
	for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return s;
}

bool ParseBool(const std::string& value, bool* out) {
	std::string v = ToLower(Trim(value));
	if (v == "1" || v == "true" || v == "yes" || v == "on") {
		*out = true;
		return true;
	}
	if (v == "0" || v == "false" || v == "no" || v == "off") {
		*out = false;
		return true;
	}
	return false;
}

std::string FindConfigPath() {
	const char* env_path = std::getenv("CAMERA_CONFIG_PATH");
	if (env_path && access(env_path, R_OK) == 0) {
		return env_path;
	}

	const char* candidates[] = {
		"./camera_params.ini",
		"../camera_params.ini",
		"../../camera_params.ini"
	};
	for (const char* path : candidates) {
		if (access(path, R_OK) == 0) {
			return path;
		}
	}
	return "";
}

bool LoadCameraParams(const std::string& path, CameraParams* params) {
	std::ifstream in(path);
	if (!in.is_open()) {
		return false;
	}

	std::string line;
	while (std::getline(in, line)) {
		line = Trim(line);
		if (line.empty() || line[0] == '#' || line[0] == ';') {
			continue;
		}

		const auto eq = line.find('=');
		if (eq == std::string::npos) {
			continue;
		}

		std::string key = ToLower(Trim(line.substr(0, eq)));
		std::string value = Trim(line.substr(eq + 1));

		try {
			if (key == "brightness") params->brightness = std::stoi(value);
			else if (key == "contrast") params->contrast = std::stoi(value);
			else if (key == "saturation") params->saturation = std::stoi(value);
			else if (key == "hue") params->hue = std::stoi(value);
			else if (key == "sharpness") params->sharpness = std::stoi(value);
			else if (key == "gamma") params->gamma = std::stoi(value);
			else if (key == "gain") params->gain = std::stoi(value);
			else if (key == "white_balance_temperature") params->white_balance_temperature = std::stoi(value);
			else if (key == "exposure_absolute") params->exposure_absolute = std::stoi(value);
			else if (key == "frame_rate_numerator") params->frame_rate_numerator = std::stoi(value);
			else if (key == "frame_rate_denominator") params->frame_rate_denominator = std::stoi(value);
			else if (key == "enable_tuning") ParseBool(value, &params->enable_tuning);
			else if (key == "auto_white_balance") ParseBool(value, &params->auto_white_balance);
			else if (key == "manual_exposure") ParseBool(value, &params->manual_exposure);
		} catch (...) {
			std::cerr << "[camera] 配置项解析失败: " << key << "=" << value << std::endl;
		}
	}

	return true;
}

bool ApplyCtrl(int fd, __u32 id, int value, const char* name) {
	struct v4l2_control ctrl;
	CLEAR(ctrl);
	ctrl.id = id;
	ctrl.value = value;
	if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) == -1) {
		std::cerr << "[camera] 设置 " << name << " 失败，忽略继续" << std::endl;
		return false;
	}
	std::cout << "[camera] " << name << " = " << value << std::endl;
	return true;
}
}  // namespace

V4L2Capture::V4L2Capture()
{

}
V4L2Capture::V4L2Capture(const char *devName, int width, int height) {
	// TODO Auto-generated constructor stub
	this->devName = devName;
	this->fd_cam = -1;
	this->buffers = NULL;
	this->n_buffers = 0;
	this->frameIndex = -1;
	this->capW=width;
	this->capH=height;
}

V4L2Capture::~V4L2Capture() {
	// TODO Auto-generated destructor stub
	stopCapture();
	freeBuffers();
	closeDevice();
}

int V4L2Capture::openDevice() {
	/*设备的打开*/
	printf("video dev : %s\n", devName);
	// 阻塞模式打开，避免 DQBUF 在主循环中长期返回 EAGAIN 导致拿不到帧
	fd_cam = open(devName, O_RDWR);
	if (fd_cam < 0) {
		perror("Can't open video device");
		std::cout <<"god!" <<std::endl;
		return -1;
	}
	return 0;
}

int V4L2Capture::closeDevice() {
	if (fd_cam > 0) {
		int ret = 0;
		if ((ret = close(fd_cam)) < 0) {
			perror("Can't close video device");
		}
		return 0;
	} else {
		return -1;
	}
}

int V4L2Capture::initDevice() {
	if (fd_cam < 0) {
		std::cerr << "camera fd invalid, call openDevice() first" << std::endl;
		return -1;
	}

	int ret;
	struct v4l2_capability cam_cap;		//显示设备信息
	struct v4l2_cropcap cam_cropcap;	//设置摄像头的捕捉能力
	struct v4l2_fmtdesc cam_fmtdesc;	//查询所有支持的格式：VIDIOC_ENUM_FMT
	struct v4l2_crop cam_crop;			//图像的缩放
	struct v4l2_format cam_format;		//设置摄像头的视频制式、帧格式等

	/* 使用IOCTL命令VIDIOC_QUERYCAP，获取摄像头的基本信息*/
	ret = ioctl(this->fd_cam, VIDIOC_QUERYCAP, &cam_cap);
	if(fd_cam<0)
	perror("on my ");
	if (ret < 0) {
		perror("Can't get device information: VIDIOCGCAP");
	}
	printf(
			"Driver Name:%s\nCard Name:%s\nBus info:%s\nDriver Version:%u.%u.%u\n",
			cam_cap.driver, cam_cap.card, cam_cap.bus_info,
			(cam_cap.version >> 16) & 0XFF, (cam_cap.version >> 8) & 0XFF,
			cam_cap.version & 0XFF);

	/* 使用IOCTL命令VIDIOC_ENUM_FMT，获取摄像头所有支持的格式*/
	cam_fmtdesc.index = 0;
	cam_fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	printf("Support format:\n");
	while (ioctl(fd_cam, VIDIOC_ENUM_FMT, &cam_fmtdesc) != -1) {
		printf("\t%d.%s\n", cam_fmtdesc.index + 1, cam_fmtdesc.description);
		cam_fmtdesc.index++;
	}

	/* 使用IOCTL命令VIDIOC_CROPCAP，获取摄像头的捕捉能力*/
	cam_cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (0 == ioctl(fd_cam, VIDIOC_CROPCAP, &cam_cropcap)) {
		printf("Default rec:\n\tleft:%d\n\ttop:%d\n\twidth:%d\n\theight:%d\n",
				cam_cropcap.defrect.left, cam_cropcap.defrect.top,
				cam_cropcap.defrect.width, cam_cropcap.defrect.height);
				// cam_cropcap.defrect.width=640;
				// cam_cropcap.defrect.height=480;
		/* 使用IOCTL命令VIDIOC_S_CROP，获取摄像头的窗口取景参数*/
		cam_crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		cam_crop.c = cam_cropcap.defrect;		//默认取景窗口大小
		if (-1 == ioctl(fd_cam, VIDIOC_S_CROP, &cam_crop)) {
			//printf("Can't set crop para\n");
		}
	} else {
		printf("Can't set cropcap para\n");
	}



	struct v4l2_streamparm Stream_Parm;
	memset(&Stream_Parm, 0, sizeof(struct v4l2_streamparm));
	Stream_Parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; 

	CameraParams params;
	const std::string config_path = FindConfigPath();
	if (!config_path.empty()) {
		if (LoadCameraParams(config_path, &params)) {
			std::cout << "[camera] 已加载参数文件: " << config_path << std::endl;
		} else {
			std::cerr << "[camera] 参数文件读取失败，使用默认值" << std::endl;
		}
	} else {
		std::cout << "[camera] 未找到 camera_params.ini，使用默认值" << std::endl;
	}

	Stream_Parm.parm.capture.timeperframe.numerator = params.frame_rate_numerator;
	Stream_Parm.parm.capture.timeperframe.denominator = params.frame_rate_denominator;

	ret = ioctl(fd_cam, VIDIOC_S_PARM, &Stream_Parm);

	if (ret < 0){
		perror("设置帧率失败，继续使用默认帧率");
	}

	std::cout << "-------------------------------------设置基础图像参数-------------------------------------" << std::endl;
	ApplyCtrl(fd_cam, V4L2_CID_BRIGHTNESS, params.brightness, "brightness");
	ApplyCtrl(fd_cam, V4L2_CID_CONTRAST, params.contrast, "contrast");
	ApplyCtrl(fd_cam, V4L2_CID_SATURATION, params.saturation, "saturation");
	ApplyCtrl(fd_cam, V4L2_CID_HUE, params.hue, "hue");
	ApplyCtrl(fd_cam, V4L2_CID_SHARPNESS, params.sharpness, "sharpness");
	ApplyCtrl(fd_cam, V4L2_CID_GAMMA, params.gamma, "gamma");
	ApplyCtrl(fd_cam, V4L2_CID_GAIN, params.gain, "gain");
	ApplyCtrl(fd_cam, V4L2_CID_AUTO_WHITE_BALANCE, params.auto_white_balance ? 1 : 0, "auto_white_balance");
	if (!params.auto_white_balance) {
		ApplyCtrl(fd_cam, V4L2_CID_WHITE_BALANCE_TEMPERATURE, params.white_balance_temperature, "white_balance_temperature");
	}
	if (params.manual_exposure) {
		ApplyCtrl(fd_cam, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL, "exposure_auto(manual)");
		ApplyCtrl(fd_cam, V4L2_CID_EXPOSURE_ABSOLUTE, params.exposure_absolute, "exposure_absolute");
	}
	std::cout << "-------------------------------------基础参数设置完成------------------------------------- " << std::endl;

	/* 使用IOCTL命令VIDIOC_S_FMT，设置摄像头帧信息*/
	cam_format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	cam_format.fmt.pix.width = capW;
	cam_format.fmt.pix.height = capH;
	cam_format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;		//要和摄像头支持的类型对应
	cam_format.fmt.pix.field = V4L2_FIELD_INTERLACED;
	ret = ioctl(fd_cam, VIDIOC_S_FMT, &cam_format);
	if (ret < 0) {
		perror("Can't set frame information");
	}
	/* 使用IOCTL命令VIDIOC_G_FMT，获取摄像头帧信息*/
	cam_format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ret = ioctl(fd_cam, VIDIOC_G_FMT, &cam_format);
	if (ret < 0) {
		perror("Can't get frame information");
	}
	printf("Current data format information:\n\twidth:%d\n\theight:%d\n",
			cam_format.fmt.pix.width, cam_format.fmt.pix.height);
	ret = initBuffers();
	if (ret < 0) {
		perror("Buffers init error");
		return -1;
	}
	return 0;
}

int V4L2Capture::initBuffers() {
	int ret;
	/* 使用IOCTL命令VIDIOC_REQBUFS，申请帧缓冲*/
	struct v4l2_requestbuffers req;
	CLEAR(req);
	req.count = 4;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;
	ret = ioctl(fd_cam, VIDIOC_REQBUFS, &req);
	if (ret < 0) {
		perror("Request frame buffers failed");
	}
	if (req.count < 2) {
		perror("Request frame buffers while insufficient buffer memory");
	}
	buffers = (struct cam_buffer*) calloc(req.count, sizeof(*buffers));
	if (!buffers) {
		perror("Out of memory");
	}
	for (n_buffers = 0; n_buffers < req.count; n_buffers++) {
		struct v4l2_buffer buf;
		CLEAR(buf);
		// 查询序号为n_buffers 的缓冲区，得到其起始物理地址和大小
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = n_buffers;
		ret = ioctl(fd_cam, VIDIOC_QUERYBUF, &buf);
		if (ret < 0) {
			printf("VIDIOC_QUERYBUF %d failed\n", n_buffers);
			return -1;
		}
		buffers[n_buffers].length = buf.length;
		//printf("buf.length= %d\n",buf.length);
		// 映射内存
		buffers[n_buffers].start = mmap(
				NULL, // start anywhere
				buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_cam,
				buf.m.offset);
		if (MAP_FAILED == buffers[n_buffers].start) {
			printf("mmap buffer%d failed\n", n_buffers);
			return -1;
		}
	}
	return 0;
}

int V4L2Capture::startCapture() {
	unsigned int i;
	for (i = 0; i < n_buffers; i++) {
		struct v4l2_buffer buf;
		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;
		if (-1 == ioctl(fd_cam, VIDIOC_QBUF, &buf)) {
			printf("VIDIOC_QBUF buffer%d failed\n", i);
			return -1;
		}
	}
	enum v4l2_buf_type type;
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == ioctl(fd_cam, VIDIOC_STREAMON, &type)) {
		printf("VIDIOC_STREAMON error");
		return -1;
	}
	return 0;
}

int V4L2Capture::stopCapture() {
	enum v4l2_buf_type type;
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == ioctl(fd_cam, VIDIOC_STREAMOFF, &type)) {
		printf("VIDIOC_STREAMOFF error\n");
		return -1;
	}
	return 0;
}

int V4L2Capture::freeBuffers() {
	unsigned int i;
	for (i = 0; i < n_buffers; ++i) {
		if (-1 == munmap(buffers[i].start, buffers[i].length)) {
			printf("munmap buffer%d failed\n", i);
			return -1;
		}
	}
	free(buffers);
	return 0;
}

int V4L2Capture::getFrame(void **frame_buf, size_t* len) {
	struct v4l2_buffer queue_buf;
	CLEAR(queue_buf);
	queue_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	queue_buf.memory = V4L2_MEMORY_MMAP;
	if (-1 == ioctl(fd_cam, VIDIOC_DQBUF, &queue_buf)) {
		if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
			return 1;  // 暂时没有可用帧，或被信号中断
		}
		perror("VIDIOC_DQBUF error");
		return -1;
	}
	if (queue_buf.index >= n_buffers) {
		std::cerr << "[camera] DQBUF index 越界: " << queue_buf.index << "/" << n_buffers << std::endl;
		return -1;
	}
	*frame_buf = buffers[queue_buf.index].start;
	*len = static_cast<size_t>(queue_buf.bytesused);
	frameIndex = queue_buf.index;
	if (*len == 0 || *len > buffers[queue_buf.index].length) {
		std::cerr << "[camera] bytesused 异常: " << *len
		          << ", buffer_len=" << buffers[queue_buf.index].length << std::endl;
		backFrame();
		return 1;
	}
	// std::cout <<"frameIndex=" << frameIndex <<std::endl;
	// std::cout <<"len=" << *len <<std::endl;
	return 0;
}

int V4L2Capture::backFrame() {
	if (frameIndex != -1) {
		struct v4l2_buffer queue_buf;
		CLEAR(queue_buf);
		queue_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		queue_buf.memory = V4L2_MEMORY_MMAP;
		queue_buf.index = frameIndex;
		if (-1 == ioctl(fd_cam, VIDIOC_QBUF, &queue_buf)) {
			printf("VIDIOC_QBUF error\n");
			return -1;
		}
		return 0;
	}
	return -1;
}

int V4L2Capture::InitCamera(){
	if (openDevice() != 0) {
		return -1;
	}
	if (initDevice() != 0) {
		return -1;
	}
	if (startCapture() != 0) {
		return -1;
	}
	return 0;
}

cv::Mat V4L2Capture::ResultCamera(){
	const int ret = getFrame((void **) &yuv422frame, (size_t *)&yuvframeSize);
	if (ret != 0) {
		if (ret < 0) {
			std::cerr << "[camera] getFrame 失败" << std::endl;
		}
		return cv::Mat();
	}
	std::vector<uchar> buff(yuv422frame,yuv422frame+yuvframeSize);
    // cv::imencode(".jpg", *yuv422frame,buff,param); 
    //解码
	cv::Mat image=cv::imdecode(cv::Mat(buff),cv::IMREAD_COLOR); 
	backFrame();
	if (image.empty()) {
		std::cerr << "[camera] MJPEG decode failed, bytesused=" << yuvframeSize << std::endl;
	}
	return image;
}
