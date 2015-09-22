#pragma once
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cv.h>       
#include <ml.h>		  // opencv machine learning include file
#include "opencv/highgui.h"
#include <io.h>
using namespace std;
using namespace cv;


struct SImgMask{
	Mat m_imgRedMask;
	Mat m_imgBlueMask;
	Mat m_imgGreenMask;
	Mat m_imgBlackMask;
	Mat m_imgGreenLight;
	Mat m_imgAdapThres;
};

struct SVecMSERs{
	vector<Rect> m_vecRectMser;
	vector<vector<Point> > m_vecPoints;
	vector<Mat> m_vecImgMser;
};

struct SImgAcromatic{
	Mat m_imgAcroMaskBlue;
	Mat m_imgAcroMaskRed;
	Mat m_imgAcroMaskMSER;
	Mat m_imgAcroMaskLight;
};

struct SRecSize{
	int m_minRectAreaRED;
	int m_maxRectAreaRED;
	int m_minRectAreaBLUE;
	int m_maxRectAreaBLUE;
	int m_minRectAreaBLACK;
	int m_maxRectAreaBLACK;
};

typedef struct SVecTracking{
	vector<Rect> vecBefore;
	vector<Rect> vecCandidate;
	vector<int> vecCount;
	vector<Point> vecCountPush;
	vector<Point2f> vecRtheta;
	vector<Point2f> vecAngle;
};

static const Vec3b bcolors[] =
{
	Vec3b(0, 0, 255),
	Vec3b(0, 128, 255),
	Vec3b(0, 255, 255),
	Vec3b(0, 255, 0),
	Vec3b(255, 128, 0),
	Vec3b(255, 255, 0),
	Vec3b(255, 0, 0),
	Vec3b(255, 0, 255),
	Vec3b(255, 255, 255)
};