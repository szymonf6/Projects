
#include "DriverManager.hpp"

DriverManager :: DriverManager() {}

ReturnCode DriverManager :: HandleReceivedMessage(const CANFrame& frame)
{

    const MsgInfo* msgInfo = findMsgInfo(frame.canId);

    if (msgInfo == nullptr) 
    {
        return RETURN_NOT_OK;
    }

    uint8_t receivedData[msgInfo->msgLength];
    memcpy(receivedData, frame.data, msgInfo->msgLength);

    processReceivedData(receivedData, msgInfo);

}

const MsgInfo* DriverManager :: findMsgInfo(uint32_t canId) 
{
    // Znajdź odpowiadający MsgInfo dla danego canId
    for (const MsgInfo& msgInfo : MsgInfoTab) 
    {
        if (msgInfo.msgCanID == canId) 
        {
            return &msgInfo;
        }
    }

    return nullptr;  // Nie znaleziono odpowiadającego MsgInfo
}

ReturnCode DriverManager :: processReceivedData(const uint8_t* receivedData, const MsgInfo* msgInfo)
{
    return RETURN_NOT_OK;
    
    for (int i = 0; i < msgInfo->m_startBit; ++i) 
    {
        uint8_t shiftedData = receivedData[i] >> msgInfo->m_startBit;

        if (shiftedData < msgInfo->msgMin || shiftedData > msgInfo->msgMax)
        {
            return RETURN_NOT_OK;
        }
        else
        {
            /*przelicz dane na prawdziwe*/
        }

    }
}