#ifndef CAR_H_
#define CAR_H_

#include <QObject>

/**
 * @class Car
 * @brief backendowa klasa do qt, przekazuje do main.qml wartości
*/
class Car : public QObject
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
    Q_PROPERTY(int  speed           READ speed      NOTIFY speedChanged         )
    Q_PROPERTY(int  batteryLevel    READ speed      NOTIFY batteryLevelChanged  )
    Q_PROPERTY(int  fullDist        READ fullDist   NOTIFY fullDistChanged      )
    Q_PROPERTY(int  avgSpeed        READ avgSpeed   NOTIFY avgSpeedChanged      )
    Q_PROPERTY(int  leftDist        READ leftDist   NOTIFY leftDistChanged      )
    Q_PROPERTY(bool isWarning       READ isWarning  NOTIFY isWarningChanged     )
    Q_PROPERTY(bool isRight         READ isRight    NOTIFY isRightChanged       )
    Q_PROPERTY(bool isLeft          READ isLeft     NOTIFY isLeftChanged        )
    Q_PROPERTY(bool isPos           READ speed      NOTIFY isPosChanged         )


public:
    /**
     * @fn Car()
     * @brief konstruktor
    */
    Car();

    /**
     * @fn ~Car()
     * @brief destruktor
    */
    ~Car();

    /**
     * @fn speed
     * @brief zwraca rzeczy
    */
    int speed();
    int batteryLevel();
    int fullDist();
    int avgSpeed();
    bool isWarning();
    bool isRight();
    bool isLeft();
    bool isPos();
    int leftDist();

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

private:
    /**
     * @var speed_
     * @brief on jest szybkoscią perły
    */
    int speed_;

    /**
     * @var batteryLevel_
     * @brief poziom naładowania baterii
    */
    int batteryLevel_;

    /**
     * @var fullDist_
     * @brief całkowity przebyty dystans
    */
    int fullDist_;

    /**
     * @var avgSpeed_
     * @brief średnia prędkość
    */
     int avgSpeed_;

    /**
     * @var isWarning_
     * @brief czy światła warning są odpalone
    */
    bool isWarning_;

    /**
     * @var isRight_
     * @brief czy kierunkowskaz prawy jest włączony
    */
    bool isRight_;

    /**
     * @var isLeft_
     * @brief czy kierunkowskaz lewy jest włączony
    */
    bool isLeft_;

    /**
     * @var isPos_
     * @brief czy światła mijania są włączone
    */
    bool isPos_;

    /**
     * @var leftDist_
     * @brief dystant ktory został
    */
    int leftDist_;

};

#endif // CAR_H_
