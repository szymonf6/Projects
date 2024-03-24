#include <iostream>
#include <cstring>

#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <unistd.h>

void readFrame(int can_socket)
{
    struct can_frame frame;
    ssize_t bytesRead = read(can_socket, &frame, sizeof(frame));

    if (bytesRead >= 0)
    {
        std::cout << "Received CAN frame:" << std::endl;
        std::cout << "ID: 0x" << std::hex << frame.can_id << std::dec << std::endl;
        std::cout << "Data Length: " << static_cast<int>(frame.can_dlc) << std::endl;

        std::cout << "Data:";
        for (int i = 0; i < frame.can_dlc; i++)
        {
            std::cout << ' ' << std::hex << static_cast<int>(frame.data[i]);
        }
        std::cout << std::endl;
    }
    else
    {
        std::cout << "Error receiving data";
    }
}


int main()
{
    // Set the CAN interface identifier
    const char* can_interface = "can0";

    // Configure CAN interface
    struct sockaddr_can addr;
    struct ifreq ifr;

    // Create CAN socket
    int can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    //can_socket = socket(PF_CAN, SOCK_DGRAM, CAN_BCM);
    if (can_socket == -1)
    {
        perror("Error creating CAN socket");
        return -1;
    }

    std::strcpy(ifr.ifr_name, can_interface);
    ioctl(can_socket, SIOCGIFINDEX, &ifr);

    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    // Bind socket to CAN interface
    bind(can_socket, (struct sockaddr*)&addr, sizeof(addr));

    // CAN frame identifier
    canid_t can_id = 0x400;

    // Data to send
    //uint8_t data[] = {0x01};

    struct can_frame frame;
    frame.can_id = can_id;
    frame.can_dlc = 4; // Fix: use can_dlc instead of can_length
    frame.data[0] = 1;
    frame.data[1] = 2;
    frame.data[2] = 3;
    frame.data[3] = 4;
    

    std :: cout << sizeof(struct can_frame) << std :: endl;

    // Send frame
    /*
    while(true)
    {
        write(can_socket, &frame, sizeof(struct can_frame));
        usleep(2400);
    }
    
        //sendFrame(can_id, data, sizeof(data), can_socket);
    */
    // Receive frame
    while(true)
        readFrame(can_socket);

    // Close CAN socket
    close(can_socket);

    return 0;
}
