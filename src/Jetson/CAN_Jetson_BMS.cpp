//do std :: cout, cin
#include <iostream>
//do usleep
#include <unistd.h>
// do std::strcpy
#include <cstring>
//zawiera definicje struktur zwiazanych z socketami
#include <sys/socket.h>
//zawiera definicje zwiazane z interfejsem sieciowym
#include <net/if.h>
//zawiera definicje zwiazane z wartowa protokołu CAN
#include <linux/can.h>
//zawiera definicje definicje dla surowego interfejsu CAN
#include <linux/can/raw.h>
//do ioctl
#include <sys/ioctl.h>

const char *CAN_INTERFACE = "can0"; //do ustawienia odpowiednia nazwa interfejsu CAN
const int CAN_FRAME_SIZE = 8; //rozmiar ramki CAN 
const int CAN_FRAME_LEN = 8;

const int BMSMasterAddress = 0x01; //adres BMSa
const uint8_t BluetoothAppAddress = 0x80; //adres bluetooth
const uint8_t GPRSAddress = 0x20; //gprs adres
const uint8_t UpperComputerAddress = 0x40; //

//inicjalizacja struktury sockaddr_can do funkcji bind. addrprzechowuje informacje o adresie interfejsu CAN
struct sockaddr_can addr;

// struktura używana do identyfikacji interfejsu can
struct ifreq ifr;

//struktura reprezentująca ramkę danych do wysłania od PC
struct JetsonFrame
{
    //pole przechowujące identyfikator ramki CAN jetsona
    uint32_t canID;

    //tablica przechowująca dane ramki CAN o stałej wielkości - CAN_FRAME_SIZE
    uint8_t data[CAN_FRAME_SIZE];
};

//struktura reprezentująca ramkę danych do wysłania od BMS
struct BMSResponseFrame
{
    //pole przechowujące identyfikator ramki CAN jetsona
    uint32_t canID;

    //tablica przechowująca dane ramki CAN o stałej wielkości - CAN_FRAME_SIZE
    uint8_t data[CAN_FRAME_SIZE];
};

//funkcja do wysyłania ramki danych przez CAN od PC
void sendCANFrame(uint32_t canId, uint8_t *data, uint8_t length,int can_socket)
{
    //tworzymy strukturę can_frame która reprezentuje ramkę CAN
    struct can_frame frame;

    //ustawiamy identyfikator ramki CAN
    frame.can_id = canId;

    //ustawiamy długość danych w race
    frame.can_dlc = length;

    //kopiujemy dane do pola danych ramki
    for(int i = 0; i < length; i++)
    {
        frame.data[i] = data[i];
    }

    //wywołujemy funkcję write, aby wysłać ramkę can
    //can_socket: identyfikator interfejsu CAN
    //&frame: referencja do struktury can_frame, któa zawiera ramkę do wysłania
    //sizeof(): rozmiar struktury can w bajtach
    write(can_socket, &frame, sizeof(struct can_frame));
}

// funkcja wysyłająca ramkę CAN z danymi związanymi z Jetsonem
void sendJetsonFrame(int can_socket, const JetsonFrame &frame)
{
    // wrtie - funkcja służąca do zapisywania danych do can_socket
    // can_socket - identyfikator interjejsu CAN, przez ktory beda wysylane dane
    // &frame - referencja do struktury JetsonFrame, któa zawiera dane do wyslania
    //sizeof() - rozmiar struktury JetsonFrame

    struct can_frame canFrame;
    canFrame.can_id = frame.canID;

    ssize_t nbytes = write(can_socket, &frame, sizeof(struct can_frame));
}

//funkcja odbierająca dane z BMS z cana i zwraca je jako wynik
BMSResponseFrame receiveBMSResponse(int can_socket)
{
    //obiekt służący do przechowywyania odebranej ramki danych
    BMSResponseFrame responseFrame;

    // pętla wykonuje się dopóki funkcja read zwraca wartość ujemną
    // can_socket - id cana
    // responseFrame - wskaźnik od miejsca gdzie będą zapisywane dane. są to dane ramki BMSResponseFrame
    // sizeof() rozmiar danych do odcyztu
    while(read(can_socket, &responseFrame, sizeof(responseFrame)) < 0)
    {
        //oczekiwanie na dane
        usleep(10);
    }
    return responseFrame;
}

//funkcja do przetwarzania odpowiedzi bms w zależności od typu wiadomości, przyjmuje stałą referencję do obiektu typu BMSResponse
void processBMSResponse(const BMSResponseFrame &responseFrame)
{
    //zmienne
    uint16_t cumulativeTotalVoltage = 0;
    uint16_t gatherTotalVoltage = 0;
    int16_t current = 0;
    int16_t soc = 0;
    int16_t maxCellVoltage = 0;
    uint16_t noOfCellWithMaxVoltage = 0;
    int16_t minCellVoltage = 0;
    int8_t maxTemperature = 0;
    uint8_t maxTemperatureCellNo = 0;
    int8_t minTemperature = 0;
    uint8_t minTemperatureCellNo = 0;

    //przetwarzamy canId otrzymane od BMSa
    switch (responseFrame.canID)
    {
        case 0x90: //SOC of total voltage current
            cumulativeTotalVoltage = (responseFrame.data[1] << 8) | responseFrame.data[2];
            gatherTotalVoltage = (responseFrame.data[3] << 8) | responseFrame.data[4];
            current = (responseFrame.data[5] << 8) | responseFrame.data[6];
            soc = (responseFrame.data[7] << 8) | responseFrame.data[8];

            //przetwarzanie danych
            std :: cout << "Cumulative total voltage: " << cumulativeTotalVoltage * 0.1 << " V" << std :: endl;
            std :: cout << "Gather total voltage: " << gatherTotalVoltage * 0.1 << " V" << std :: endl;
            std :: cout << "Current: " << (current - 30000) * 0.1 << " A" << std :: endl;
            std :: cout << "SOC: " << soc * 0.1 << "%" << std :: endl;
        break;

        case 0x91: //max and min voltage
            maxCellVoltage = (responseFrame.data[0] << 8) | responseFrame.data[1];
            noOfCellWithMaxVoltage = responseFrame.data[2];
            minCellVoltage = (responseFrame.data[3] << 8) | responseFrame.data[4];

            // print odczytwanych danych
            std ::cout << "Max cell voltage: " << maxCellVoltage << "mV" << std ::endl;
            std ::cout << "Cell No with Max Voltage: " << static_cast<int>(noOfCellWithMaxVoltage) << std ::endl;
            std ::cout << "Min Cell Voltage: " << minCellVoltage << "mV" << std::endl;
        break;

        case 0x92: // max and min temp
            maxTemperature = responseFrame.data[0] - 40; // skalowanie offsetu
            maxTemperatureCellNo = responseFrame.data[1];
            minTemperature = responseFrame.data[2] - 40; // skalowanie offsetu
            minTemperatureCellNo = responseFrame.data[3];

            // print danych odczytanych
            std ::cout << "Max temp: " << maxTemperature << " C" << std ::endl;
            std ::cout << "Cell No with Max Temperature: " << static_cast<int>(maxTemperatureCellNo) << std ::endl;
            std ::cout << "Min temperature: " << minTemperature << " C" << std ::endl;
            std ::cout << "Cell No with Min Temperature: " << static_cast<int>(minTemperatureCellNo) << std ::endl;
        break;

        //...

        default:
            std :: cout << "Unhandled response ID: 0x" << std :: hex << responseFrame.data[0] << std :: endl;
        break;
    }
}

//wyswietlanie ramek CAN
void printCANFrame(const struct can_frame &frame) 
{
    //wyswietlamy szestnastkowo CAN ID i dziesietnie dane
    std :: cout << "CAN ID: 0x" << std :: hex << frame.can_id << std :: dec << ", Data: ";

    //iteruje przez dane ramki CAN do dlugosci danych w ramce CAN
    for (int i = 0; i < frame.can_dlc; ++i) 
    {
        //wypisujemy kolejne bajty danych ramki CAN. rzutujemy wartość bajtu na liczbe całkowitą int
        std :: cout << "0x" << std :: hex << static_cast<int>(frame.data[i]) << " ";
    }
    std :: cout << std :: endl;
}

//funkcja do wyysłania zapytania i odbioru odpowiedzi
void sendQueryAndReceive(int can_socket, uint32_t queryID, uint8_t bmsAddress, uint8_t pcAddress)
{
    //obiekt ramki danych do wysłania z zapytaniem od JEtsona
    JetsonFrame JetsonQueryFrame;

    //konstrukcja identyfikatora ramki na podstawie uzyskanych danych
    JetsonQueryFrame.canID = queryID | (UpperComputerAddress << 16) | (bmsAddress << 24);

    //wysyłanie zapytania
    sendJetsonFrame(can_socket, JetsonQueryFrame);

    // Odczyt odpowiedzi
    struct can_frame receivedFrame;

    //odczyt
    ssize_t nbytes = read(can_socket, &receivedFrame, sizeof(struct can_frame));

    do
    {
        if (nbytes < 0)
        {
            //jeśli brak odpowiedzi, czekaj i ponów próbę
            usleep(1000);
        }
        else
        {
            //odczyt się powiódł
            break;
        }
    } while(true);

    //paranoid check
    if(nbytes < sizeof(struct can_frame))
    {
        std :: cout << "Incomplete CAN frame" << std :: endl; 
    }

    // Wypisz odebraną ramkę CAN
    printCANFrame(receivedFrame);

    //przygotowanie ramki danych do przetworzenia przez funkcję processBMSResponse
    BMSResponseFrame responseFrame;
    responseFrame.canID = receivedFrame.can_id;
    for(int i = 0; i < CAN_FRAME_SIZE; i++)
    {
        responseFrame.data[i] = receivedFrame.data[i];
    }

    //przetworzenie odebranej odpowiedzi
    //processBMSResponse(responseFrame);
}

int main()
{
    // 1. otwarcie interfejsu CAN
    // 2. pobranie indeksu interfejsu
    // 3. konfiguracja adresu interfejsu CAN
    // 4. przypisanie adresu interfejsu CAN do gniazda
    //utworzenie socketu dla interfejsu CAN, can_socket - przechowuje identyfikator gniazda
    // PF_CAN - rodzaj protokołu w systemie
    //SOCK_RAW - gniazdo będzie używać surowego dostępu do danych, CAN_RAW - gniazdo obsługuje surowe ramki
    // surowe gniazda umożliwiają dostęp do warstwy transportowej bezposrednio
    int can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);

    //sprawdzenie czy otwarcie gniazda sie utworzylo
    if (can_socket < 0)
    {
        std :: cout << "Error opening CAN interface" << std :: endl;
        return -1;
    }

    // uzyskanie indeksu interfejsu, który jest poźniej używany w konfiguracji interfejsu 
    // kopiuje nazwę interfejsu CAN (CAN_INTERFACE) do struktury ifr
    std::strcpy(ifr.ifr_name, "can0");

    // ustawienie indeksu interfejsu na podstawie nazy interfejsu. 
    // SIOCGIFINDEX - kod operacji ioctl używany do uzyskania indeksu interfejsu dla danej nazwy interfejsu
    ioctl(can_socket, SIOCGIFINDEX, &ifr);

    // ustawienie rodziny adresowej na AF_CAN
    addr.can_family = AF_CAN;

    // ustawienie indeksu interfejsu na indeks uzyskany wcześniej poprzez funkcję ioctl
    addr.can_ifindex = ifr.ifr_ifindex;

    //przypisanie gniazdu określony interfejs CAN
    //adres interfejsu jest dostarczony przez strukturę addr
    if(bind(can_socket, (struct sockaddr*)&addr, sizeof(addr)) == -1)
    {
        std :: cout << "Error binding CAN socket" << std :: endl;
        close(can_socket);
        return -1;
    }

    //pętla główna
    while (true)
    {
        // Wysłanie ramki 0x18100140 do BMS
        //ustalenie identyfikatora ramki can na 0x18100140
        uint32_t canId = 0x18100140;

        // tablica któa zostanie wysłana w ramce can
        uint8_t data[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        // określa dlugosc danych w ramce dzięki czemu sendCANFRame bedzie wiedziala ile bajtow nalezy wyslac
        uint8_t length = sizeof(data) / sizeof(data[0]);

        //BMS Address: 0x01, Jetson Address: 0x40
        sendQueryAndReceive(can_socket, canId, BMSMasterAddress, UpperComputerAddress);

        // Odczyt danych z BMSa
        std::cout << "-------------------" << std::endl;
        std::cout << "Processing BMS Data" << std::endl;
        std::cout << "-------------------" << std::endl;

        usleep(5000000);
    }

    //zamknięcie interfejsu CAN
    close(can_socket);

    return 0;
}