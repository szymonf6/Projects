#ifndef CLASSMANAGER_HPP
#define CLASSMANAGER_HPP

#include "pthread.h"

#include "JetsonDriver.hpp"
#include "CAN.hpp"
#include "DBC.hpp"
#include "DriverManager.hpp"
#include "DisplayDriver.hpp"
#include "CameraDriver.hpp"
#include "MotorDriver.hpp"

#include "ReturnCode.hpp"

/**
 * @brief one to rule the all...
 * @class ClassManager
*/
class ClassManager
{
    public:
    /**
     * @fn getInstance
     * @brief 
     * @return CEcuClassManager& 
     */
    static ClassManager& getInstance();

    /**
     * @fn ClassManager
     * @brief konstruktor inicjuje inne klasy
    */
    ClassManager();

    /**
    * @fn ~ClassMAnaegr
    * @brief destruktor zabija watki i deinicjalizuje jetsonka
    */
    ~ClassManager();

    /**
     * @fn run
     * @brief rozpoczyna pętle softu
     */
    ReturnCode run();

private:

    /**
     * @var OCJetsonDriver
     * @brief obiekt CJetsonDriver
     */
    CJetsonDriver OCJetsonDriver;

    /**
     * @var ODriverManager
     * @brief obiekt DriverManager
     */
    DriverManager ODriverManager;

    /**
     * @var OCAN
     * @brief obiekt CAN
     */
    CAN OCAN;

    /**
     * @var ODisplayDriver
     * @brief obiekt wyswietlacza
    */
    DisplayDriver ODisplayDriver;

    /**
    * @var OCameraDriver
    * @brief obiekt kamerki
    */
    CameraDriver OCameraDriver;

    /**
    * @var OCameraDriver
    * @brief obiekt kamerki
    */
    CameraDriver OMotorDriver;
};

#endif /*CLASSMANAGER_HPP*/