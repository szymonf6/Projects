#ifndef MOTORDRIVER_HPP__
#define MOTORDRIVER_HPP__

#include "CAN.hpp"
#include "pthread.h"
#include "ReturnCode.hpp"

class MotorDriver
{
    public:
    /**
     * @fn MotorDriver()
     * @brief konsyruktor
    */
    MotorDriver();

    /**
     * @fn ~MotorDriver()
     * @brief destruktor
    */
    ~MotorDriver();

    /**
     * @var DisplayThread
     * @brief obiekt wątku motora
    */
   pthread_t MotorThread;
    
    /**
     * @var run()
     * @brief odpalanie działania liczenia dyferencjalu motoru
    */
    ReturnCode run();

    private:
    /**
     * @fn entryPoint
     * @brief funkcja ktora jest wywolywana przy tworzeniu watku
    */
    static void *entryPoint(void* context) 
    {
        return ((OMotorDriver *)context)->run();
    }
};

#endif /*MOTORDRIVER_HPP__*/