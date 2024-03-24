#ifndef CAR_H_
#define CAR_H_

#include <QObject>
#include <QTimer>
#include "CAN.hpp"

/**
 * @class Car
 * @brief backendowa klasa do qt, przekazuje do main.qml wartości
*/
class Car : public QObject, public CAN
{
    /**
     * @var Q_OBJECT
     * @brief makro używane w klasach dziedziczących po QObject, używane do dodania funkcjonalności
    */
    Q_OBJECT

    /**
     * @var Q_PROPERTY
     * @brief makro w Qt, które jest używane do deklarowania właściwości obiektów, 
     * umożliwiając integrację z systemem QML, obsługę sygnałów i slotów, a 
     * także umożliwia dostęp do właściwości obiektów w czasie wykonania.
    */
    Q_PROPERTY(int  iSpeed           MEMBER speed               NOTIFY speedChanged         )
    Q_PROPERTY(int  iBatteryLevel    MEMBER batteryLevel        NOTIFY batteryLevelChanged  )
    Q_PROPERTY(int  iFullDist        MEMBER fullDist            NOTIFY fullDistChanged      )
    Q_PROPERTY(int  iAvgSpeed        MEMBER avgSpeed            NOTIFY avgSpeedChanged      )
    Q_PROPERTY(int  iLeftDist        MEMBER leftDist            NOTIFY leftDistChanged      )
    Q_PROPERTY(bool iIsWarning       MEMBER isWarning           NOTIFY isWarningChanged     )
    Q_PROPERTY(bool iIsRight         MEMBER isRight             NOTIFY isRightChanged       )
    Q_PROPERTY(bool iIsLeft          MEMBER isLeft              NOTIFY isLeftChanged        )
    Q_PROPERTY(bool iIsPos           MEMBER speed               NOTIFY isPosChanged         )


public:
    /**
     * @fn Car()
     * @brief konstruktor
    */
    Car(QObject *parent = nullptr);

    /**
     * @fn ~Car()
     * @brief destruktor
    */
    ~Car();

    /**
     * @brief funkcje które zwracają rzeczy
    */
    int getCurrentSpeed();
    int getCurrentBatteryLevel();
    int getCurrentFullDist();
    int getCurrentAvgSpeed();
    bool getCurrentIsWarning();
    bool getCurrentIsRight();
    bool getCurrentIsLeft();
    bool getCurrentIsPos();
    int getCurrentLeftDist();

    /**
     * @brief funkcje które updatują rzeczy
    */
    void updateCurrentSpeed();
    void updateCurrentBatteryLevel();
    void updateCurrentFullDist();
    void updateCurrentIsWarning();
    void updateCurrentIsRight();
    void updateCurrentIsLeft();
    void updateCurrentIsPos();
    void updateCurrentLeftDist();

/**
 * @brief sygnałki przekazywane do qml
*/
signals:
    /**
     * @fn sygnały
     * @brief sygnał przekazuje do qmla wartości rzeczy
    */
    void speedChanged(int speed);
    void batteryLevelChanged(int batteryLevel);
    void fullDistChanged(int fullDist);
    void avgSpeedChanged(int avgSpeed);
    void isWarningChanged(bool isWarningChanged);
    void isLeftChanged(bool isLeft);
    void isRightChanged(bool isRight);
    void isPosChanged(bool isPos);
    void leftDistChanged(int leftDist);

private slots:
    void updateSpeedPeriodically();
    void updateBatteryLevelPeriodically();
    void updateAvgSpeedPeriodically();
    void updateIsWarningPeriodically();
    void updateIsLeftPeriodically();
    void updateIsRightPeriodically();
    void updateIsPosPeriodically();
    void updateLeftDistPeriodically();

private:
    /**
     * @var speed_
     * @brief on jest szybkoscią perły
    */
    int speed;

    /**
     * @var batteryLevel_
     * @brief poziom naładowania baterii
    */
    int batteryLevel;

    /**
     * @var fullDist_
     * @brief całkowity przebyty dystans
    */
    int fullDist;

    /**
     * @var avgSpeed_
     * @brief średnia prędkość
    */
     int avgSpeed;

    /**
     * @var isWarning_
     * @brief czy światła warning są odpalone
    */
    bool isWarning;

    /**
     * @var isRight_
     * @brief czy kierunkowskaz prawy jest włączony
    */
    bool isRight;

    /**
     * @var isLeft_
     * @brief czy kierunkowskaz lewy jest włączony
    */
    bool isLeft;

    /**
     * @var isPos_
     * @brief czy światła mijania są włączone
    */
    bool isPos;

    /**
     * @var leftDist_
     * @brief dystant ktory został
    */
    int leftDist;

    /**
     * @var timer
     * @brief timer z QT który jest potrzebny aby odswiezac rzeczy
    */
    QTimer updateTimer;
};

#endif // CAR_H_