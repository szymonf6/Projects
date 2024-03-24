#include "ClassManager.hpp"

ClassManager :: ClassManager() 
: OCJetsonDriver(), ODriverManager(), OCAN(), ODisplayDriver(), OCameraDriver() 
{
    pthread_create(&JetsonDriverThread, nullptr, &OJetsonDriver.entryPoint, &OCJetsonDriver);
    pthread_detach(JetsonDriverThread);

    pthread_create(&CANThread, nullptr, &OCAN.entryPoint, &OCAN);
    pthread_detach(CANThread);

    pthread_create(&DriverManaegrThread, nullptr, &ODriverManager.entryPoint, &ODriverManager);
    pthread_detach(DriverManagerThread);

    pthread_create(&DisplayThread, nullptr, &ODisplayDriver.entryPoint, &ODisplayDriver);
    pthread_detach(DisplayThread);

    pthread_create(&CameraThread, nullptr, &OCameraDriver.entryPoint, &OCameraDriver);
    pthread_detach(CameraDrievr);

    pthread_create(&MotorThread, nullptr, &OMotorDriver.entryPoint, &OMotorDriver);
    pthread_detach(MotorDrievr);
}

ClassManager& ClassManager::getInstance()
{
    static ClassManager ClassInstance;
    return ClassInstance;
}

void ClassManager :: run()
{

    while(true)
    {

    }
}