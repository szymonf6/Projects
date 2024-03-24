#ifndef ENUMS_HPP_
#define ENUMS_HPP_

/**
 * @enum Drivers
 * @brief trzyma liste wszystkich driverów w systemie
*/
typedef enum Drivers
{
    LightsDriver = 0,
    
    MotorDriver,
    
    ButtonDriver,
    
    DisplayDriver,
    
    CameraDriver,

    BmsDriver,

    //liczba driverow
    totalDrivers

} DriverId;


/**
 * @enum msgID
 * @brief przechowuje tylko id ramek ktore otrzymuje jetson
*/
typedef enum msgID
{
    
} msgID;

/**
 * @enum Messages
 * @brief trzyma liste wszystkich sygnałów ze sterowników i jetsona
*/
typedef enum Messages
{
    //sterownik swiatel
    sigLightsInit = 0,           //0x167

    sigLightsPos,                //0x168
    sigLightsBack,
    sigLightsStop,
    sigLightsDirLeft,
    sigLightsDirRight,
    
    //sterownik silników
    sigMotorInit,                //0x130

    sigMotorDir,                 //0x131

    sigMotorRotLeft,             //0x132
    sigMotorRotRight,

    //sterownik przyciskow
    sigButtonInit,               //0x200

    sigButtonLightsPos,          //0x201
    sigButtonLightsBack,
    sigButtonLightsDirLeft,
    sigButtonLightsDirRight,
    sigButtonWarning, 

    sigButtonPedalGas,           //0x202

    sigButtonPedalBrake,         //0x203

    sigButtonPRND,               //0x204
    
    sigButtonSteeringWheel,      //0x205

    //wyświetlacz
    sigDisplayInit,              //0x240

    //kamerka
    sigCameraInit,               //0x241

    //bms
    sigBmsInit,                  //0x89

    sigSOCCumulativeTotalVoltage,//0x90
    sigSOCGatherTotalVoltage,
    sigSOCCurrent,
    sigSOC,
    
    sigBmsMAxMinVoltage,         //0x91
    sigBmsMAxMinTemp,            //0x92
    sigBmsChargeDischargeStatus, //0x93
    sigBmsStatus,                //0x94
    sigBmsCellVoltage,           //0x95
    sigBmsCellTemperature,       //0x96
    sigBmsCellBalanceState,      //0x97
    sigBmsBatteryFailureStatus,  //0x98

    //licznik sygnałów w enumie
    totalSignals

} messages;

#endif /*ENUMS_HPP_*/