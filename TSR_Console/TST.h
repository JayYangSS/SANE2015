#pragma once
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cv.h> 
#include "opencv/highgui.h"
#include <io.h>
#include "kalmanFilter.h"
#include "structure.h"
#include "TSD.h"
using namespace std;
using namespace cv;
#define ToRadian(degree) ((degree)*(CV_PI/180.0f))
#define ToDegree(radian) ((radian)*(180.0f/CV_PI))


#ifndef Angle
#define Angle
float AngleTransform(const float &tempAngle, const int &scale);
float fAngle(int x, int y);


//float fAngle(int x, int y)
//{
//	return ToDegree(atan2f(y, x));
//}
#endif
class CTST
{
public:
	CTST();

	CTST(CTSD& adapDetect);

	~CTST();
	void SetVanishingPT(Point pt);
	Point GetVanishingPT();
	void kalmanMultiTarget(Mat& srcImage, vector<Rect>& vecRectTracking, SVecTracking& Set, vector<SKalman>& MultiKF, float scaledist, int cntCandidate, int cntBefore, int frameCandiate, int& cntframe, Rect& ROIset, vector<Rect>& vecValidRec, Mat& imgROImask);
	void AdaptiveROI(Mat& imgSrc);
	SVecTracking m_multiTracker;
	vector<SKalman> m_MultiKF;
	SVecMSERs m_MserVec;
	vector<Rect> m_vecRectTracking;
	
	void SetImage(Mat& imgSrc);
	void SetROI(Rect& droi_u, Rect& droi_r, Rect& troi_u, Rect& troi_r, Mat& roimask);
	Mat m_imgROImask;
	Mat m_imgROImask_T;
private:

	void kalmanTrackingStart(SKalman& temKalman, Rect& recStart);
	Point m_ptVanishing;
	CTSD m_adaptiveDetection;
	Rect m_roi_d_u;
	Rect m_roi_d_r;
	Rect m_roi_t_u;
	Rect m_roi_t_r;

	Mat m_imgSrc;
	
	
	

};

