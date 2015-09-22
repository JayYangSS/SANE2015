#include <iostream>
#include <stdio.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
//#include "structure.h"
//#include "ConvNet.h"
//#include "kalmanFilter.h"
//#include "EM.h"
#include "TSD.h"
#include "TST.h"
#include "TSC.h"
//using namespace std;
//using namespace cv;


Mat imgshow;
Point ptVanishing;

CvNormalBayesClassifier *bayesian = new CvNormalBayesClassifier;
Mat testest;
//

#define scaledist 0.4
#define cntCandidate 2
#define cntBefore 4
#define frameCandiate 5

int main()
{

	// ------------------------------------------------------------ Kalman Filter setup
	vector<SKalman> MultiKF_U;
	vector<SKalman> MultiKF_R;


	///////////////////////////////////////////////////////////////
	char Xmlfilename[100] = "20classTS.xml";
	initRead(Xmlfilename);


	double fps = 15;

	int fourcc = CV_FOURCC('X', 'V', 'I', 'D'); // codec
	bool isColor = true;
	VideoWriter *video = new VideoWriter;
	if (!video->open("result.avi", fourcc, fps, Size(1280, 720), isColor)){
		delete video;
	}
	//invTri = imread("invtri.png",0);
	//RectangleM = imread("rectangle.png",0);
	/* Load Video */
	//cv::VideoCapture m_videoCapture("sunny.wmv");
	//cv::VideoCapture m_videoCapture("blackbox2hd.wmv");
	//	string source = "2015-03-05-11h-52m-26s_F_normal.mp4"; //2015-03-18-17h-26m-33s_F_normal//
	//string source = "backlight_01.wmv";
	string source = "cloudy_03_expressway.wmv";
	//	string source = "2015-03-18-10h-10m-20s_F_normal.mp4";
	//	string source = "2015-03-18-17h-20m-32s_F_normal.mp4";
	//	string source = "2015-03-18-17h-26m-33s_F_normal.mp4";
	//	string source = "2015-03-07-09h-18m-05s_F_event.mp4";
	//	string source = "2015-03-15-15h-08m-11s_F_event.mp4";
	//	string source = "2015-03-18-10h-08m-20s_F_normal.mp4";
	cv::VideoCapture m_videoCapture(source);
	if (!m_videoCapture.isOpened()){
		cout << "Could not load the video file." << endl;
		return -1;
	}
	cout << "Successfully loaded the video file." << endl;
	///////////////////////////////////////////////////

	unsigned int frameCount = m_videoCapture.get(CV_CAP_PROP_FRAME_COUNT);

	Mat srcImage;
	Mat imgROImask;
	Mat imgDifference;
	Mat imgTempDiff;
	Mat imgDetect;
	int cntframe = 0;

	float fscale = 1.0f;

	vector<Rect> vecValidRec;

	ptVanishing.x = 640 - 20;
	ptVanishing.y = 360 - 20;

	CTSD TSdetector;
	
	CTSC TSclassifier;
	bayesian->load("trainDtBs.xml");
	TSdetector.m_bayesian = bayesian;
	
	TSdetector.SetMinArea(400);
	TSdetector.SetMaxArea(4000);
	TSdetector.SetScale(1.0);

	CTST TStracker(TSdetector);
	TStracker.SetVanishingPT(ptVanishing);
	while (true)
	{

		cntframe++;
		cout << cntframe << endl;
		
		/* Capture Image Frame */
		m_videoCapture >> srcImage;
		if (srcImage.empty())	break;
		///////////////////

		imgshow = srcImage.clone();

		double t = (double)getTickCount();
		///////////////////
		circle(imgshow, ptVanishing, 1, CV_RGB(255, 0, 0), 2);
		line(imgshow, Point(ptVanishing.x, 0), Point(ptVanishing.x, ptVanishing.y), CV_RGB(0, 255, 0));
		line(imgshow, Point(imgshow.cols, ptVanishing.y), Point(ptVanishing.x, ptVanishing.y), CV_RGB(255, 255, 255), 2);
		line(imgshow, Point((ptVanishing.x - (ptVanishing.y*tan(ToRadian(50))) > imgshow.cols ? imgshow.cols : ptVanishing.x - (ptVanishing.y*tan(ToRadian(50)))), 0), Point(ptVanishing.x, ptVanishing.y), CV_RGB(255, 255, 255), 2);

		srcImage.copyTo(imgDetect);
		imgROImask = Mat::zeros(srcImage.size(), CV_8UC1);

		vector<Rect> vecRectTracking;
		Rect ROIset_U = Rect_<int>(srcImage.cols / 3 - srcImage.cols / 8, srcImage.rows / 4 - srcImage.rows / 8 + srcImage.rows / 32, srcImage.cols * 5 / 8, srcImage.rows / 4 - srcImage.rows / 8 + srcImage.rows / 32);
		Rect ROIset_R = Rect_<int>(srcImage.cols * 2 / 3, srcImage.rows / 4 - srcImage.rows / 32, srcImage.cols / 4 - srcImage.cols / 16, srcImage.rows / 4);
		Rect ROIset_UU = Rect_<int>(srcImage.cols / 3 - srcImage.cols / 8, srcImage.rows / 4 - srcImage.rows / 8 - srcImage.rows / 16, srcImage.cols * 5 / 8, srcImage.rows / 4 /* - srcImage.rows / 8 + srcImage.rows / 16*/);
		Rect ROIset_RR = Rect_<int>(srcImage.cols * 2 / 3, srcImage.rows / 4 - srcImage.rows / 32, srcImage.cols / 4, srcImage.rows / 4);
		
		TStracker.SetImage(srcImage);
		TStracker.SetROI(ROIset_U, ROIset_R, ROIset_UU, ROIset_RR, imgROImask);
		TStracker.AdaptiveROI(imgshow);
		double t2 = (double)getTickCount();


		TSdetector.Clustering(TStracker.m_vecRectTracking, 10);
		t2 = (double)getTickCount() - t2;

		TStracker.kalmanMultiTarget(srcImage, TStracker.m_vecRectTracking, TStracker.m_multiTracker, TStracker.m_MultiKF, scaledist, cntCandidate, cntBefore, frameCandiate, cntframe, ROIset_U, vecValidRec, TStracker.m_imgROImask_T);
		TSclassifier.Tracking_Validation(imgshow, srcImage, TStracker.m_MserVec, TStracker.m_MultiKF, TStracker.m_multiTracker, cntframe, ROIset_U, fscale, vecValidRec, TStracker.m_imgROImask_T);
		
		TStracker.m_vecRectTracking.clear();
		vecValidRec.clear();
		t = (double)getTickCount() - t;
		printf("Traffic Sign detection : %f ms.\n", t*1000. / getTickFrequency());


		//*video << imgshow;


		imshow("imgshow", imgshow);
		//imshow("drawimg",imgdrawTest);

		char saveimg[50];

		//sprintf(saveimg, "imggg%d.jpg", cntframe);

		//imshow ("mask",imgROImask);
		int waitTime;
		(t*1000. / getTickFrequency() < 70) ? waitTime = 70 - t*1000. / getTickFrequency() : waitTime = 70;
		char c = waitKey(1);
		//if (c == 's')	imwrite(saveimg, imgshow);

		if (c == 'q')	break;
	}
	delete video;
	return 0;
}
