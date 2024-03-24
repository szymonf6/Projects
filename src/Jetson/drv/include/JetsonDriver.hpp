/**
* @file CJetsonDriver.hpp
* @brief klasa z deklaracją funkcji odpowiedzialna za inicjalizacje pinów jetsona
*/

#ifndef CJETSONDRIVER_HPP_
#define CJETSONDRIVER_HPP_

#include <pthread.h>

#include "DBC.hpp"

/**
 * @fn CJetsonDriver
 * @brief klasa przechowująca funkcje inicjalizujące Jetsona 
*/
class CJetsonDriver
{
    public:
        /**
         * @fn CJetsonDriver
         * @brief konstruktor
        */
        CJetsonDriver();

       /**
        * @fn ~CJetsonDriver
        * @brief destruktor
       */
        ~CJetsonDriver();
       
        /** 
         * @fn init()
         * @brief inicjalizacja pinów jetsona
         */ 
        void init();

        /** 
         * @fn deinit()
         * @brief deinicjalizacja pinów jetsona
         */ 
        void deinit();

        /**
         * @fn sendDataToDriver()
         * @brief wysyła dane do driverManagera
        */
        void sendDataToDriver(const MsgGeneric& a_roMsg);

        /**
        * @fn run()
        * @brief wywołuje inicjalizację
        */
        void run();

        /**
          * @var DisplayThread
          * @brief obiekt wątku wyświetlacza
        */
        pthread_t JetsonDriverThread;

        /**
         * @fn entryPoint
         * @brief funkcja ktora jest wywolywana przy tworzeniu watku
        */
        static void *entryPoint(void* context) 
        {
            return ((OJetsonDriver*)context)->run();
        }

    private:

        /**
         * @var  pathToCanFile
         * @brief przechowuje sciezke do pliku inicjalizacyjnego can
        */
        const char* pathToCanFile = "/";   //sys/devices/platform/can0, /boot/extlinux/extlinux.conf lub /boot/efi/etxlinux                                    // DO UZUPEŁNIENIA

        /**
         * @var  pathToPinsFile
         * @brief przechowuje sciezke do pliku inicjalizacyjnego 40 pinowego headera
        */
        const char* pathToPinsFile = "/";          //sys/class/gpio/gpioXX                             // DO UZUPEŁNIENIA

        /**
         * @var  pathToCameraPinFile
         * @brief przechowuje sciezke do pliku inicjalizacyjnego wejscia do kamerki jetsona
        */
        const char* pathToCameraPinFile = "/";        //etc/X11                               // DO UZUPEŁNIENIA

        /**
         * @var  pathToDisplayPinFile
         * @brief przechowuje sciezke do pliku inicjalizacyjnego wejscia do ekranu jetsona
        */
        const char* pathToDisplayPinFile = "/";    //etc/X11                                     // DO UZUPEŁNIENIA

        /**
         * @var  pathToDeinitCanFile
         * @brief przechowuje sciezke do pliku deinicjalizacyjnego can
        */
        const char* pathToDeinitCanFile = "/";  

        /**
         * @var  pathToDeinitPinsFile
         * @brief przechowuje sciezke do pliku deinicjalizacyjnego 40 pinowego headera
        */
        const char* pathToDeinitPinsFile = "/";          //sys/class/gpio/gpioXX                             // DO UZUPEŁNIENIA

        /**
         * @var  pathToDeinitCameraPinFile
         * @brief przechowuje sciezke do pliku deinicjalizacyjnego wejscia do kamerki jetsona
        */
        const char* pathToDeinitCameraPinFile = "/";        //etc/X11                               // DO UZUPEŁNIENIA

        /**
         * @var  pathToDeinitDisplayPinFile
         * @brief przechowuje sciezke do pliku deinicjalizacyjnego wejscia do ekranu jetsona
        */
        const char* pathToDeinitDisplayPinFile = "/";    //etc/X11                                     // DO UZUPEŁNIENIA

        /**
        * @var resultEnableDisplay
        * @brief przechowuje info czy wyswietlacz zostal zainicjalizowany
        */
        bool resultEnableDisplay;

        /**
          * @var resultEnableCamera
          * @brief przechowuje info czy kamerka zostala zainicjalizowana
        */
        bool resultEnableCamera;

        /**
          * @var resultEnableHeader
          * @brief przechowuje info czy 40pinowy header zostal zainicjalizowany
        */
        bool resultEnableHeader;

        /**
          * @var resultEnableCAN
          * @brief przechowuje info czy can zostal zainicjalizowany
        */
        bool resultEnableCAN;

        /**
          * @var resultDisableDisplay
          * @brief przechowuje info czy wyswietlacz zostal zdeinicjalizowany
        */
        bool resultDisableDisplay;

        /**
          * @var resulDisableCamera
          * @brief przechowuje info czy kamerka zostala zdeinicjalizowana
        */
        bool resulDisableCamera;

        /**
          * @var resultDisableHeader
          * @brief przechowuje info czy 40pinowy header zostal zdeinicjalizowany
        */
        bool resultDisableHeader;

        /**
          * @var resultDisableCAN
          * @brief przechowuje info czy can zostal zdeinicjalizowany
        */
        bool resultDisableCAN;
};

#endif /*JETSON_DRIVER_HPP_*/