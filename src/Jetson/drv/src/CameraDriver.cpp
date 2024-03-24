#include "CameraDriver.hpp"

CameraDriver::CameraDriver() 
{
}

bool CameraDriver::initializeCamera() {

    std::string pipeline = gstreamer_pipeline(capture_width,
	                                        capture_height,
                                            display_width,
                                            display_height,
                                            framerate,
                                            flip_method);
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if(!cap.isOpened()) 
    {
	    return (-1);
    }

    cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat img;
}

void CameraDriver::run() {
    while(true)
    {
    	if (!cap.read(img))
        {
		    break;
	    }
	
	cv::imshow("CSI Camera",img);
	int keycode = cv::waitKey(10) & 0xff ; 
        if (keycode == 27) break ;
    }
}

CameraDriver::~CameraDriver() 
{
    cap.release();
    cv::destroyAllWindows();
}