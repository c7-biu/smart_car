#pragma once

#include <signal.h>
#include <vector>
#include <fcntl.h>
#include <stdio.h> /*标准输入输出定义*/
#include <stdlib.h> /*标准函数库定义*/
#include <unistd.h> /*Unix 标准函数定义*/
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h> /*文件控制定义*/
#include <termios.h> /*PPSIX 终端控制定义*/
#include <errno.h> /*错误号定义*/
#include <iostream>
#include <string.h>

using namespace std;

#pragma pack(1)
struct Send_struct
{
    uint8_t head = 0X7B;      //head
    uint8_t reserved_1 = 0;   //不可动，是否回充
    uint8_t reserved_2 = 0;    
    short x_vel_target =0;
    short y_vel_target = 0;
    short z_vel_target = 0;
    uint8_t check_num = 0;  
    uint8_t tail = 0X7D;      //tail
};
#pragma pack()

#pragma pack(1)
struct Receive_struct
{
    uint8_t head = 0X7B;      //head
    uint8_t reserved_1 = 0;   //不可动，是否回充
    uint8_t reserved_2 = 0;
    short x_vel_target =0;
    short y_vel_target = 0;
    short z_vel_target = 0;
    uint8_t check_num = 0;  
    uint8_t tail = 0X7D;      //tail
};
#pragma pack()


void usart_init(int bt, const char* dev);
int set_opt(struct termios newtio, int fd, int nSpeed);
void bluetooth_send();
void bluetooth_receive(int status);


static char head = 'J';        //74
static char tail = 'D';        //68
extern Send_struct send_struct;
extern Receive_struct receive_struct;


extern int usart_0;
