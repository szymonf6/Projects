#ifndef CAN_HPP_
#define CAN_HPP_

#include <pthread.h>
#include <string.h>

#include "DBC.hpp"
#include "ReturnCode.hpp"

struct CANFrame {
    uint32_t canId;
    uint8_t data[8];
};

class CAN
{
    public:
        /**
         * @fn CAN
         * @brief konstruktor
        */
        CAN();

       /**
        * @fn ~CAN
        * @brief destruktor
       */
        ~CAN();

        /**
         * @fn CanInit
         * @brief inicjalizacja cana
        */
        ReturnCode CanInit();

        /**
        * @fn setBitrate
        * @brief ustaw bitrate cana
        */
        ReturnCode setBitrate(int bitrate);

        /**
        * @fn setFilter
        * @brief ustaw filtr dla konkretnego identyfikatora zeby otrzymywac tylko okreslone ramki
       */
        ReturnCode setFilter(uint32_t canID);

        /**
        * @fn getErrorCount
        * @brief zwraca liczbę błędów na interfejsie
       */
        int getErrorCount();

        /**
         * @fn readFrame
         * @brief odczytuje ramki can
        */
        CANFrame readFrame();

        /**
         * @fn sendFrame
         * @brief wysyła ramkę can
        */
        ReturnCode sendFrame(const CANFrame& frame);

        /**
         * @fn convertRawToActual
         * @brief przelicza wartosci z raw na rzeczywiste
        */
        double convertRawToActual(uint16_t rawValue, const MsgInfo& msgInfo);

        double returnActualSOC();

        /**
         * @fn run()
         * @brief petla glowna cana
        */
        ReturnCode run();

        /**
         * @var CANThread
         * @brief obiekt wątku cana
        */
         pthread_t CANThread;

         /**
         * @fn entryPoint
         * @brief funkcja ktora jest wywolywana przy tworzeniu watku
        */
        static void *entryPoint(void* context) 
        {
            return ((OCAN *)context)->run();
        }

    private:
        /**
         * @var
         * @brief trzyma nazwe interfejsu can
        */
       const char* can0 = "can0";

       /**
        * @var 
        * @brief socket cana
       */
        const int can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);

        double actualSOC;

};

#endif /*CAN_HPP_*/