#include <iostream>
#include <wiringPi.h>
#include <wiringSerial.h>

const int TX_PIN = 8;
const int RX_PIN = 10;

//struktura reprezentująca ramkę danych do wysłania
struct BMSFrame
{
    uint8_t startFlag;
    uint8_t pcAddress;
    uint8_t dataID;
    uint8_t dataLength;
    uint8_t data[8];
    uint8_t checksum;
};

// funkcja do wysyłania ramki danych przez UART
void sendBMSFrame(int serialPort, const BMSFrame &frame)
{
    serialPutchar(serialPort, frame.startFlag);
    serialPutchar(serialPort, frame.pcAddress);
    serialPutchar(serialPort, frame.dataID);
    serialPutchar(serialPort, frame.dataLength);

    for (int i = 0; i < 8; i++)
    {
        serialPutchar(serialPort, frame.data[i]);
    }

    serialPutchar(serialPort, frame.checksum);
}

//funkcja do wysyłania zapytania i odbioru odpowiedzi
void sendQueryAndReceive(int serialPort, uint8_t queryID)
{
    BMSFrame queryFrame;
    queryFrame.startFlag = 0x45;
    queryFrame.pcAddress = 0x40;
    queryFrame.dataID = queryID;
    queryFrame.dataLength = 0;
    queryFrame.checksum = 0;

    //wysylanie zapytania
    sendBMSFrame(serialPort, queryFrame);

    //oczekiwanie na odp
    while(!serialDataAvail(serialPort))
    {
        delay(10);
    }

    //odczytywanie odp
    BMSFrame responseFrame;
    serialGetchar(serialPort); //odczyt startFlag, ignorujemy go
    serialGetchar(SerialPort); // odczyt pcAddress, ignorujemy go
    responseFrame.dataID = serialGetchar(serialPort);
    responseFrame.dataLength = serialGetchar(serialPort);

    for(int i = 0; i < responseFrame.dataLength; i++)
    {
        responseFrame.data[i] = serialGetchar(serialPort);
    }

    responseFrame.checksum = serialGetchar(serialPort);

    //przetwarzanie danych
    switch (responseFrame.dataID)
    {
        case 0x90:
            if(responseFrame.dataLength == 6)
            {
                uint16_t cumulativeTotalVoltage = (responseFrame.data[0] << 8) | responseFrame.data[1];
                uint16_t gatherTotalVoltage = (responseFrame.data[2] << 8) | responseFrame.data[3];
                int16_t current = static_cast<int16_t>((responseFrame.data[4] << 8) | responseFrame.data[5]) - 30000;

                //print odczytywanych danych
                std :: cout << "Cumulative total voltage: " << cumulativeTotalVoltage * 0.1 << " V" << std :: endl;
                std :: cout << "Gather total voltage: " << gatherTotalVoltage * 0.1 << " V" << std :: endl;
                std :: cout << "Current: " << current * 0.1 << " A" << std :: endl;
            }
            break;

        case 0x91: //maximum and minimum voltage
            if(responseFrame.dataLength == 5)
            {
                int16_t maxCellVoltage = (responseFrame.data[0] << 8) | responseFrame.data[1];
                uint16_t noOfCellWithMaxVoltage = responseFrame.data[2];
                int16_t minCellVoltage = (responseFrame.data[3] << 8) | responseFrame.data[4];

                //print odczytwanych danych
                std :: cout << "Max cell voltage: " << maxCellVoltage << "mV" << std :: endl;
                std :: cout << "Cell No with Max Voltage: " << static_cast<int>(noOfCellWithMaxVoltage) << std :: endl;
                std :: cout << "Min Cell Voltage: " << minCellVoltage << "mV" << std::endl;
            }
            break;

        case 0x92: //max and min temperature
            if(responseFrame.dataLength == 4)
            {
                int8_t maxTemperature = responseFrame.data[0] - 40; //skalowanie offsetu
                uint8_t maxTemperatureCellNo = responseFrame.data[1];
                int8_t minTemperature = responseFrame.data[2] - 40; //skalowanie offsetu
                uint8_t minTemperatureCellNo = responseFrame.data[3];

                //print danych odczytanych
                std :: cout << "Max temp: " << maxTemperature << " C" << std :: endl;
                std :: cout << "Cell No with Max Temperature: " << static_cast<int>(maxTemperatureCellNo) << std :: endl;
                std :: cout << "Min temperature: " << minTemperature << " C" << std :: endl;
                std :: cout << "Cell No with Min Temperature: " << static_cast<int>(minTemperatureCellNo) << std :: endl; 
            }
            break;

            //OBSLUGA INNYCH KOMUNIKATÓW...

            default:
                std :: cout << "Unhandled response ID: 0x" << std :: hex << static_cast<int>(responseFrame.dataID) << std :: endl;
                break;
    }

}

int main()
{
    //inicjalizacja WiringPi
    if(wiringPiSetup() == -1)
    {
        std :: cout << "Error initializing WiringPi" << std :: endl;
        return 1;
    }

    //otwarcie portu szeregowego
    int seiralPort = serialOpen("/dev/ttyS0", 9600);

    if(serialPort == -1)
    {
        std :: cout << "Error opening serial port" << std :: endl;
        return -1;
    }

    //ustawienie pinów jako I/O
    pinMode(TX_PIN, OUTPUT);
    pinMode(RX_PIN, INPUT);

    //petla główna
    while(true)
    {
        //wyslanie zapytania o min i max voltage 0x91
        sendQueryAndReceive(serialPort, 0x91);

        //wyslanie zapytania o min i max temp 0x92
        sendQueryAndReceive(serialPort, 0x92);

        //odczyt danych z bmsa
        std :: cout << "-------------------" << std :: endl;
        std :: cout << "Processing BMS Data" << std :: endl;
        std :: cout << "-------------------" << std :: endl;

        delay(5000);

    }
}
