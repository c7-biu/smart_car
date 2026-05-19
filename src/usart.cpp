#include "usart.h"
#include <time.h>

int usart_0;

void usart_init(int bt, const char* dev)
{
    if (chmod(dev, 0666) != 0) {
        // 权限不足时继续运行，避免卡死在 sudo 密码输入。
        perror("serial chmod failed");
    }

    struct termios signal_usart;
    struct sigaction saio; /* definition of signal action */

    /* open the device to be non-blocking (read will return immediatly) */
    usart_0 = open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (usart_0 <= 0)
    {
        std::cout << "Can not open " << std::endl;
        exit(-1);
    }

    //设置中断处理函数，就是接收到数据后会执行的函数，定义在后面
    saio.sa_handler = bluetooth_receive;

    sigemptyset(&saio.sa_mask);
    saio.sa_flags = 0;
    saio.sa_restorer = NULL;
    sigaction(SIGIO, &saio, NULL);

    // fcntl(usart_0, F_SETOWN, getpid()); //allow the process to receive SIGIO
    int flags = fcntl(usart_0, F_GETFL, 0);
    if (flags < 0) {
        perror("串口读取标志失败");
    } else if (fcntl(usart_0, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("串口设置非阻塞失败");
    }
    // fcntl(usart_0, F_SETFL, 0);    //恢复串口的状态为阻塞状态，用于等待串口数据的读入
    set_opt(signal_usart, usart_0, bt);
}

int set_opt(struct termios newtio, int fd, int nSpeed)
{
    /*步骤一，设置字符大小*/
    newtio.c_cflag |= CLOCAL | CREAD;
    newtio.c_cflag &= ~CSIZE;
    /*设置数据位*/
    newtio.c_cflag |= CS8;

    /*设置奇偶校验位*/
    //无奇偶校验位
    newtio.c_cflag &= ~PARENB;

    /*设置波特率*/
    switch (nSpeed)
    {
    case 9600:
        cfsetispeed(&newtio, B9600);
        cfsetospeed(&newtio, B9600);
        break;
    case 115200:
        cfsetispeed(&newtio, B115200);
        cfsetospeed(&newtio, B115200);
        break;
    default:
        cfsetispeed(&newtio, B9600);
        cfsetospeed(&newtio, B9600);
        break;
    }
    /*设置停止位*/
    newtio.c_cflag &= ~CSTOPB;

    /*设置等待时间和最小接收字符*/
    newtio.c_cc[VTIME] = 1;
    newtio.c_cc[VMIN] = 0;

    //使用原始模式，如果以字符串形式接收下面两行可以去掉
    newtio.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    newtio.c_oflag &= ~OPOST;

    /*处理未接收字符*/
    tcflush(fd, TCIFLUSH);
    /*激活新配置*/
    if ((tcsetattr(fd, TCSANOW, &newtio)) < 0)
    {
        perror("com set error");
        return -1;
    }
    cout << "串口开启成功" << endl;
    return 0;
}

Send_struct send_struct;
Receive_struct receive_struct;
void bluetooth_send()
{
    uint8_t buffer[11];
    int idx = 0;    
    // 头部
    buffer[idx++] = send_struct.head;
    
    // 保留字节
    buffer[idx++] = send_struct.reserved_1;
    buffer[idx++] = send_struct.reserved_2;
    
    // 速度值 - 明确指定字节序（假设目标是大端）
    // x_vel_target = 100 (0x0064)
    buffer[idx++] = (send_struct.x_vel_target >> 8) & 0xFF;  // 高字节: 0x00
    buffer[idx++] = send_struct.x_vel_target & 0xFF;          // 低字节: 0x64
    // y_vel_target = 100 (0x0064)  
    buffer[idx++] = (send_struct.y_vel_target >> 8) & 0xFF;
    buffer[idx++] = send_struct.y_vel_target & 0xFF;
    
    // z_vel_target = 100 (0x0064)
    buffer[idx++] = (send_struct.z_vel_target >> 8) & 0xFF;
    buffer[idx++] = send_struct.z_vel_target & 0xFF;
    
    // 校验和（如果需要计算）
    buffer[idx++] = send_struct.check_num;

    // 尾部
    buffer[idx++] = send_struct.tail;


    if (usart_0 < 0) {
        return;
    }

    int sent = 0;
    while (sent < idx) {
        ssize_t n = write(usart_0, buffer + sent, idx - sent);
        if (n > 0) {
            sent += static_cast<int>(n);
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // 非阻塞模式下串口暂时不可写，直接丢弃当前控制帧，避免主线程卡住。
            return;
        }
        return;
    }
}


void bluetooth_receive(int status)
{
    
    int stu_size = sizeof(receive_struct);
    uint8_t cnt = 0, Rxmsg[stu_size];

    //检查针头，中间直接读取, 针尾再次核对
    read(usart_0, &Rxmsg[cnt++], sizeof(uint8_t));

    if (Rxmsg[0] != head)
    {
        std::cout<<"error head"<<std::endl;
        return ;
    }
    else
    {
        while(cnt < stu_size)
            read(usart_0, &Rxmsg[cnt++], sizeof(uint8_t));
        
        if (Rxmsg[stu_size-1] != tail)
        {
            cnt = 0;
            std::cout<<"error tail"<<std::endl;
            return;
        }
    }
    memset(&receive_struct, 0, stu_size);
    memcpy(&receive_struct, Rxmsg, stu_size);

    cnt = 0;
}

    
