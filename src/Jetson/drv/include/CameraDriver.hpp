/**
* @file CameraDriver.hpp
* @brief klasa z deklaracją funkcji odpowiedzialna za kamerke
*/

#ifndef CAMERADRIVER_HPP_
#define CAMERADRIVER_HPP_

#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <string.h>

class CameraDriver:
{
public:
    /**
     * @fn CameraDriver()
     * @brief konstruktor
    */
    CameraDriver();

    /**
     * @fn ~CameraDriver
     * @brief destruktor
    */
    ~CameraDriver();

    /**
     * @fn initializeCamera()
     * @brief inicjalizuje kamerke
    */
    bool initializeCamera();

    /**
     * @fn run()
     * @brief Metoda do rozpoczęcia przechwytywania obrazu z kamery
    */
    void run();

    /**
     * @var CameraThread
     * @brief obiekt wątku kamerki
    */
    pthread_t CameraThread;

private:
    /**
     * @var gstreamer_pipeline
     * @brief przeklejone z neta, generuje ciąg znaków ktory rozpoczyna korzystanie z kamerki, określa parametry przechwytywania obrazu, dodaje przekształcenia obrazu 
    */
    std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    }

    /**
     * @var capture_width
     * @brief szerokość przechwytywania
    */
    const int capture_width = 1280;

    /**
     * @var capture_height
     * @brief wysyokość przechwytywania
    */
    const int capture_height = 720;

    /**
     * @var display_width
     * @brief szerokość wyświetlania
    */
    const int display_width = 1280;

    /**
     * @var display_height
     * @brief wysokość przechwytywania
    */
    const int display_height = 720;

    /**
     * @var framerate
     * @brief ilość klatek na sekundę
    */
    const int framerate = 30;

    /**
     * @car flip_method
     * @brief przekształcenia obrazu
    */
    const int flip_method = 0;

    /**
     * @var cap
     * @brief obiekt openCV ktory umozliwia dostep do kamerki
    */
    cv::VideoCapture cap;

    /**
     * @var img
     * @brief obiekt ktory reprezentuje obraz w programie, uzywany do przechwytywania pojedynczej klatki obrazu pochodzacej z kamery
    */
    cv::Mat img;

    /**
     * @fn entryPoint
     * @brief funkcja ktora jest wywolywana przy tworzeniu watku
    */
    static void *entryPoint(void* context) 
    {
        return ((OCameraDriver *)context)->run();
    }
};

#endif /* CAMERADRIVER_HPP_ */