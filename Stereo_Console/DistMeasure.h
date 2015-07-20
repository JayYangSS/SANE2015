/*
*  Class DistMeasure.h
*  measure the distace of detected object
*
*  Created by T.K.Woo on July/17/2015.
*  Copyright 2015 CVLAB at Inha. All rights reserved.
*
*/
#pragma once

#include<iostream>
#include<opencv2\opencv.hpp>
#include <string.h>
#include <fstream>
#include <math.h>

using namespace std;
using namespace cv;

#define PI 3.141592

class CDistMeasure{
	
private:
	//input
	Mat m_imgLeftInput; //retified image
	Mat m_imgRightInput;//retified image
	int m_nVanishingY;
	double m_dPitchDeg;
	int m_nStereoAlg;
	vector<Rect_<int> > m_vecrectRoi;
	//member image, temp image
	Mat m_matDisp;		//16bit disparity
	Mat m_imgDisp8;		//8bit disparity image
	Mat m_imgHist;		//histogram
	Mat m_imgRoiLeftTemp;	//debuging
	Mat m_imgRoiRightTemp;
	Mat m_imgGT;

public:
	//parameters
	enum { STEREO_BM = 0, STEREO_SGBM = 1 };
	enum { FVLM, MONO, STEREOBM, STEREOSGBM };

	double m_dBaseLine;
	double m_dFocalLength; //re
	double m_dBoundDist;
	double m_dMaxDist;
	int m_nNumberOfDisp;	//must be multiple of 16
	int m_nWindowSize;		//odd number
	int m_nDistAlg;// Flag seq : FVLM, MONO, STEREOBM, STEREOSGBM
	Size m_sizeSrc;
	bool m_flgVideo;		//dispaly flag
	bool m_flgDisplay;
	//output
	vector<double> m_vecdDistance;


	//funtion
	CDistMeasure();
	void help();
	void SetImage(Mat& imgLeft, Mat& imgRight, vector<Rect_<int> >& vecrectRoi);
	void SetParam(Mat& imgLeft, Mat& imgRight, vector<Rect_<int> >& vecrectRoi, double dBaseLine, double dFocalLength,
		int nVanishingY = 0, double dPitchDeg = 0,
		int nNumOfDisp = 48, int nWindowSize = 13, int nStereoAlg = 0, int nDistAlg = 0,	//number of disparity=48, stereo alg=SAD block matching, dist alg=FVLM
		double dBoundDist = 20, double dMaxDist = 50 //unit : meters
		);
	void SetParam(double dBaseLine, double dFocalLength,
		int nVanishingY = 0, double dPitchDeg = 0,
		int nNumOfDisp = 48, int nWindowSize = 13, int nStereoAlg = 0, int nDistAlg = 0,	//number of disparity=48, stereo alg=SAD block matching, dist alg=FVLM
		double dBoundDist = 20, double dMaxDist = 50 //unit : meters
		);
	int CalcDistImg(int flag = 0); // FVLM, mono, stereoBM, stereoSGBM
	int CalcDistRoi_mono(Rect_<int>& rectROI, double& dDistance);
	int CalcDistRoi_stereo(Rect_<int>& rectRoi, double& dDistance, int nflag = STEREOBM);
	int CalcDistRoi_FVLM(Rect_<int>& rectRoi, double& dDistance, int nflag = STEREOBM);//Fast Visual Localization Method : 2015 tkwoo
	int DispToHist(double& dDistance);
	void Display();

};