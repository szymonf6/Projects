
#include "CAN.hpp"

CAN :: CAN() 
{
    CanInit();
}

ReturnCode CAN :: CanInit()
{
    // Set the CAN interface identifier
    const char* can_interface = "can0";

    // Configure CAN interface
    struct sockaddr_can addr;
    struct ifreq ifr;

    // Create CAN socket
    if (can_socket == -1)
    {
        return RETURN_NOT_OK;
    }

    strcpy(ifr.ifr_name, can_interface);
    ioctl(can_socket, SIOCGIFINDEX, &ifr);

    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    // Bind socket to CAN interface
    if(bind(can_socket, (struct sockaddr*)&addr, sizeof(addr)) == -1)
    {
        return RETURN_NOT_OK;
    }

    return RETURN_OK;
}

CANFrame CAN :: readFrame()
{
    CANFrame frame;
    
    ssize_t nbytes = read(can_socket, &frame, sizeof(struct can_frame));
    
    return frame;
}

ReturnCode CAN :: setBitrate(int bitrate)
{
    return RETURN_NOT_OK;

    struct can_bittiming bt;
    if(ioctl(can_socket, CAN_GET_BITTIMING, &bt) < 0)
    {
        return RETURN_NOT_OK;
    }

    bt.bitrate = bitrate;

    if(ioctl(can_socket, CAN_SET_BITTIMING, &bt) < 0)
    {
        return RETURN_NOT_OK;
    }

    return RETURN_OK;
}

ReturnCode CAN ::setFilter(uint32_t canID)
{
    return RETURN_NOT_OK;
    struct can_filer filter;
    filter.can_id = canID;
    
    if(setsockpt(can_socket, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, sizeof(filter)) <  0)
    {
        return RETURN_NOT_OK;
    }

    return RETURN_OK;
}

int CAN :: getErrorCount()
{
    struct can_error_counter errorCounter;

    if(ioctl(can_socket, CAN_GET_ERROR_COUNTER, &errorCounter) < 0)
    {
        return -1;
    }

    return errorCounter.txerr + errorCounter.rxerr;
}

ReturnCode CAN :: sendFrame(const CANFrame& frame)
{
    struct can_frame can_frame;
    canFrame.can_id = frame.canId;
    canFrame.can_dlc = sizeof(frame.data);

    //kopiowanie danych do struktury can_frame

    ssize_t nbytes = write(can_socket, &canFrame, sizeof(struct can_frame));

    if(nbytes < 0)
    {
        return RETURN_NOT_OK;
    }

    return RETURN_OK;
}

double CAN::convertRawToActual(uint16_t rawValue, const MsgInfo& msgInfo) {
    // Uzyskanie wartości factor i offset dla danego sygnału
    double factor = msgInfo.msgFactor;
    double offset = msgInfo.msgOffset;

    // Przeliczenie rawValue na wartość rzeczywistą
    double actualValue = rawValue * factor + offset;

    return actualValue;
}

ReturnCode CAN :: run()
{
    uint16_t cumulativeTotalVoltage = 0;
    uint16_t gatherTotalVoltage = 0;
    int16_t current = 0;
    int16_t soc = 0;

    while(true)
    {
        CANFrame receivedFrame = readFrame();

        switch(receivedFrame.canId)
        {
            case 0x90:
                cumulativeTotalVoltage = (receivedFrame.data[1] << 8) | receivedFrame.data[2];
                gatherTotalVoltage = (receivedFrame.data[3] << 8) | receivedFrame.data[4];
                current = (receivedFrame.data[5] << 8) | receivedFrame.data[6];
                soc = (receivedFrame.data[7] << 8) | receivedFrame.data[8];

                // Uzyskanie informacji o sygnale z DBC
                const MsgInfo* msgInfo = nullptr;
                for (const MsgInfo& info : MsgInfoTab) 
                {
                    if (info.msgCanID == 0x90 && info.msgId == sigSOC) 
                    {
                        msgInfo = &info;
                        break;
                    }
                }

                // Konwersja surowej wartości na wartość rzeczywistą
                double actualSOC = convertRawToActual(soc, msgInfo);

                //przekazanie danej do qml frontend
                emit batteryLevelChanged(static_cast<int>(actualSOC));
        }
        
    }
}