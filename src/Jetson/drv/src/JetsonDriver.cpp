#include "JetsonDriver.hpp"
//do otwierania plikow systemowych
#include <cstdlib>
//do cout
#include <iostream>
//do cana
#include "includes.hpp"

CJetsonDriver :: CJetsonDriver() {}

void CJetsonDriver :: init()
{
    //init pinow headera, uprawnienia 755 do bezproblemowego otwierania pliku

    resultEnableHeader = system(pathToPinsFile);
    if(resultEnableHeader == 1)
    {
        std :: cout << "błąd inicjalizacji pinów 40pinowego headera" << std :: endl;
    }
    else
    {
        std :: cout << "inicjalizacja 40pinowego headera zakonczona pomyslnie" << std :: endl;
    }

    //init cana
    resultEnableCAN = system(pathToCanFile);
    if(resultEnableCAN == 1)
    {
        std :: cout << "błąd inicjalizacji pinów cana" << std :: endl;
    }
    else
    {
        std :: cout << "inicjalizacja cana zakonczona pomyslnie" << std :: endl;
    }

    //init kamery
    resultEnableCamera = system(pathToCameraPinFile);
    if(resultEnableCamera == 1)
    {
        std :: cout << "błąd inicjalizacji kamerki" << std :: endl;
    }
    else
    {
        std :: cout << "inicjalizacja kamerki zakonczona pomyslnie" << std :: endl;
    }

    //init wyświetlacza
    resultEnableDisplay = system(pathToDisplayPinFile);
    if(resultEnableDisplay == 1)
    {
        std :: cout << "błąd inicjalizacji wyświetlacza" << std :: endl;
    }
    else
    {
        std :: cout << "inicjalizacja wyświetlacza zakonczona pomyslnie" << std :: endl;
    }
}

void CJetsonDriver :: deinit()
{
    //deinit wyswietlacza
    resultDisableDisplay = system(pathToDeinitDisplayPinFile);
    if(resultDisableDisplay == 1)
    {
        std :: cout << "błąd deinicjalizacji" << std :: endl;
    }
    else
    {
        std :: cout << "deinicjalizacja zakonczona pomyslnie" << std :: endl;
    }

    //deinit kamerki
    resulDisableCamera = system(pathToDeinitCameraPinFile);
    if(resulDisableCamera == 1)
    {
        std :: cout << "błąd deinicjalizacji" << std :: endl;
    }
    else
    {
        std :: cout << "deinicjalizacja zakonczona pomyslnie" << std :: endl;
    }

    //deinit headera
    resultDisableHeader = system(pathToDeinitPinsFile);
    if(resultDisableHeader == 1)
    {
        std :: cout << "błąd deinicjalizacji" << std :: endl;
    }
    else
    {
        std :: cout << "deinicjalizacja zakonczona pomyslnie" << std :: endl;
    }

    //deinit cana
    resultDisableCAN = system(pathToDeinitCanFile);
    if(resultDisableCAN == 1)
    {
        std :: cout << "błąd deinicjalizacji" << std :: endl;
    }
    else
    {
        std :: cout << "deinicjalizacja zakonczona pomyslnie" << std :: endl;
    }
}

void CJetsonDriver :: run()
{
    CJetsonDriver :: init();
}

void CJetsonDriver :: sendDataToDriver()
{
    //DO UZUPELNIENIA
}