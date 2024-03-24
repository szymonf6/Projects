/**
* @file DisplayDriver.hpp
* @brief klasa z deklaracją drivera wyswietlacza
*/

#ifndef DISPLAYDRIVER_HPP_
#define DISPLAYDRIVER_HPP_

#include <pthread.h>
#include "ReturnCode.hpp"

class DisplayDriver :
{
public:
    /**
     * @fn DisplayDriver()
     * @brief konstruktor
    */
    DisplayDriver();

    /**
     * @fn ~DisplayDriver()
     * @brief destruktor
    */
    ~DisplayDriver();

    /**
     * @var DisplayThread
     * @brief obiekt wątku wyświetlacza
    */
   pthread_t DisplayThread;
    
    /**
     * @var run()
     * @brief odpalanie działania wyświetlacza
    */
    ReturnCode run();

private:
    /**
     * @fn entryPoint
     * @brief funkcja ktora jest wywolywana przy tworzeniu watku
    */
    static void *entryPoint(void* context) 
    {
        return ((ODisplayDriver *)context)->run();
    }
};

#endif /* DISPLAYDRIVER_HPP_ */