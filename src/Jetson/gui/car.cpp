#include "car.hpp"

Car::Car() {}

Car::~Car(){}

int Car :: speed()
{
    return speed_;
}

int Car:: batteryLevel()
{
    return batteryLevel_;
}

int Car :: fullDist()
{
    return fullDist_;
}

int Car :: avgSpeed()
{
    return avgSpeed_;
}

bool Car :: isWarning()
{
    return isWarning_;
}

bool Car :: isRight()
{
    return isRight_;
}

bool Car :: isLeft()
{
    return isLeft_;
}

bool Car :: isPos()
{
    return isPos_;
}

int Car :: leftDist()
{
    return leftDist_;
}
