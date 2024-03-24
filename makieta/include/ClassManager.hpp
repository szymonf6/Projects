#ifndef CLASSMANAGER_HPP
#define CLASSMANAGER_HPP

#include "CAN.hpp"
#include "DBC.hpp"

#include "ReturnCode.hpp"

/**
 * @brief one to rule the all...
 * @class ClassManager
*/
class ClassManager
{
    public:
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
     * @var OCAN
     * @brief obiekt CAN
     */
    CAN OCAN;
};

#endif /*CLASSMANAGER_HPP*/