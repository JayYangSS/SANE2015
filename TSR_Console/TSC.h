#pragma once
#include "kalmanFilter.h"
#include "structure.h"
#include "ConvNet.h"
#include "opencv/highgui.h"
#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cv.h>       
#include "TST.h"
using namespace std;
using namespace cv;

class CTSC
{
public:
	CTSC();
	~CTSC();
	void Tracking_Validation(Mat& imgshow, Mat &srcImage, SVecMSERs& MserVec, vector<SKalman>& MultiKF, SVecTracking& High, int& cntframe, Rect& ROIset, float& fscale, vector<Rect>& vecValidRec, Mat& imgROImask);
private:
	void CNN(Mat& imgshow, Mat& imgsrc, Rect & rec);
	

};

