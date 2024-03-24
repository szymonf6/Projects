/**
 * @file DBC.hpp
 * @brief trzyma info jak wygladaja messagi
*/

#ifndef DBC_HPP_
#define DBC_HPP_

#include "Constants.hpp"
#include "Enums.hpp"
#include <stdint.h>

/**
 * @struct msgReceived
 * @brief przechowuje tylko id ramki i dane ktore dostala ramka
*/
struct msgReceived
{
    uint32_t canId;
    uint8_t data[8];
};

/**
 * @struct MsgInfo
 * @brief przechowuje info o messagach, jest uzywana zeby sprawdzac czy info ktore przychodzi os stmow ma sens
*/
struct MsgInfo
{
    uint32_t msgCanID;
    messages msgId;
    uint8_t m_startBit;
    uint8_t msgLength;
    double msgFactor;
    double msgOffset;
    double msgMin;
    double msgMax;
};

const MsgInfo MsgInfoTab[totalSignals]
{
     //przyciski
    {0x89,        sigBmsInit,                   0,      1,      0,      0,      0,      1           },

    {0x90,        sigSOCCumulativeTotalVoltage, 0,      2,      0,      0,      21,     29.4        },
    {0x90,        sigSOCGatherTotalVoltage,     2,      2,      0,      0,      21,     29.4        },
    {0x90,        sigSOCCurrent,                4,      2,      30000,  0,      0,      10          },
    {0x90,        sigSOC,                       6,      2,      0,      0,      0,      100         },

    {0x91,        sigBmsMAxMinVoltage,          0, 0, 0, 0, 0},
    
    {0x92,        sigBmsMAxMinTemp,             0, 0, 0, 0, 0},
    
    {0x93,        sigBmsChargeDischargeStatus,  0, 0, 0, 0, 0},
    
    {0x94,        sigBmsStatus,                 0, 0, 0, 0, 0},
    
    {0x95,        sigBmsCellVoltage,            0, 0, 0, 0, 0},
    
    {0x96,        sigBmsCellTemperature,        0, 0, 0, 0, 0},
    
    {0x97,        sigBmsCellBalanceState,       0, 0, 0, 0, 0},
    
    {0x98,        sigBmsBatteryFailureStatus,   0, 0, 0, 0, 0},

    //silniczki
    {0x130,       sigMotorInit,                 0, 0, 0, 0, 1},

    {0x131,       sigMotorDir,                  0, 0, 0, 0, 69},

    {0x132,       sigMotorRotLeft,              0, 0, 0, 0, 1},
    {0x132,       sigMotorRotRight,             1, 0, 0, 0, 69},

    //światła
    {0x167,       sigLightsInit,                0, 0, 0, 0, 1},

    {0x168,       sigLightsPos,                 0, 0, 0, 0, 1},
    {0x168,       sigLightsBack,                1, 0, 0, 0, 1},
    {0x168,       sigLightsStop,                2, 0, 0, 0, 1},
    {0x168,       sigLightsDirLeft,             3, 0, 0, 0, 1},
    {0x168,       sigLightsDirRight,            4, 0, 0, 0, 1},

    //przyciski
    {0x200,       sigButtonInit,                0, 0, 0, 0, 1},

    {0x201,       sigButtonLightsPos,           0, 0, 0, 0, 1},
    {0x201,       sigButtonLightsBack,          1, 0, 0, 0, 1},
    {0x201,       sigButtonLightsDirLeft,       2, 0, 0, 0, 1},
    {0x201,       sigButtonLightsDirRight,      3, 0, 0, 0, 1},
    {0x201,       sigButtonWarning,             4, 0, 0, 0, 1},

    {0x202,       sigButtonPedalGas,            0, 0, 0, 0, 1},
    {0x203,       sigButtonPedalBrake,          0, 0, 0, 0, 1},

    {0x204,       sigButtonPRND,                0, 0, 0, 0, 1},

    {0x205,       sigButtonSteeringWheel,       0, 0, 0, 0, 1},

    //wyswietlacz
    {0x240,       sigDisplayInit,               0, 0, 0, 0, 1},

    //kamerka
    {0x241,       sigCameraInit,                0, 0, 0, 0, 1},
};

#endif /*DBC_HPP_*/