#ifndef DRIVERMANAGER_HPP_
#define DRIVERMANAGER_HPP_

#include <string.h>
#include <pthread.h>

#include "DBC.hpp"
#include "CAN.hpp"

/**
 * @class DriverMAnager
 * @brief sprawdza czy z otrzymanymi danymi jest wporzadku i zgodnie z oczekiwaniami
*/
class DriverManager
{
    public:
        /**
         * @fn DriverManager
         * @brief konstruktor
        */
       DriverManager();

       /**
        * @fn ~DriverManager()
        * @brief destruktor
       */
       ~DriverManager();
       
        /**
         * @fn HandleReceivedMessage
         * @brief 
         */
        ReturnCode HandleReceivedMessage(const CANFrame& frame);

        /**
        * @fn processReceivedData
        * @brief przetwarza otrzymaną ramkę
        */
         ReturnCode processReceivedData(const uint8_t* receivedData, const MsgInfo* msgInfo);

        /**
         * @var DriverManagerThread
         * @brief obiekt wątku driver managera
        */
        pthread_t DriverManagerThread;

    private:
        /**
         * @fn entryPoint
         * @brief funkcja ktora jest wywolywana przy tworzeniu watku
        */
        static void* entryPoint(void* context) 
        {
            ODriverManager* DriverManager = static_cast<ODriverManager*>(context);

            CAN :: readFrame();
            
            // Wywołanie metody na obiekcie ODriverManager
            return driverManager->HandleReceivedMessage(frame);
        }

        /**
         * @fn findMsgInfo
         * @brief przeszukuje tablice zeby znalezc canID 
        */
       const MsgInfo* findMsgInfo(uint32_t canId);

};

#endif /*DRIVERMANAGER_HPP_*/