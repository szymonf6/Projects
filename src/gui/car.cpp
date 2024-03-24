#include "car.hpp"

Car::Car(QObject *parent) : QObject(parent) 
{   
    connect(&updateTimer, QTimer::timeout, this, &Car::updateSpeedPeriodically);
    connect(&updateTimer, QTimer::timeout, this, &Car::updateBatteryLevelPeriodically);
    connect(&updateTimer, QTimer::timeout, this, &Car::updateAvgSpeedPeriodically);
    connect(&updateTimer, QTimer::timeout, this, &Car::updateIsWarningPeriodically);
    connect(&updateTimer, QTimer::timeout, this, &Car::updateIsLeftPeriodically);
    connect(&updateTimer, QTimer::timeout, this, &Car::updateIsRightPeriodically);
    connect(&updateTimer, QTimer::timeout, this, &Car::updateIsPosPeriodically);
    connect(&updateTimer, QTimer::timeout, this, &Car::updateLeftDistPeriodically);

    updateTimer.setInterval(100);

    updateTimer.start();

    updateCurrentSpeed();
    updateCurrentBatteryLevel();
    updateCurrentFullDist();
    updateCurrentIsWarning();
    updateCurrentIsRight();
    updateCurrentIsLeft();
    updateCurrentIsPos();
    updateCurrentLeftDist();
}

Car::~Car(){}

int Car :: getCurrentSpeed()
{
    return speed;
}

int Car:: getCurrentBatteryLevel()
{
    batteryLevel =  CAN :: returnActualSOC();
    return batteryLevel;
}

int Car :: getCurrentFullDist()
{
    return fullDist;
}

int Car :: getCurrentAvgSpeed()
{
    return avgSpeed;
}

bool Car :: getCurrentIsWarning()
{
    return isWarning;
}

bool Car :: getCurrentIsRight()
{
    return isRight;
}

bool Car :: getCurrentIsLeft()
{
    return isLeft;
}

bool Car :: getCurrentIsPos()
{
    return isPos;
}

int Car :: getCurrentLeftDist()
{
    return leftDist;
}