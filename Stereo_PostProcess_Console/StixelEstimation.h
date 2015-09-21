/*
*  Class StixelEstimation.h
*  This class creates STIXEL using disparity image.
*
*  Created by T.K.Woo on Aug/29/2015.
*  Copyright 2015 CVLAB at Inha. All rights reserved.
*
*/
#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <stdio.h>

using namespace std;
using namespace cv;

struct stixel_t{
	int nGround;
	int nHeight;
	uchar chDistance;
	stixel_t(){
		nGround = -1;
		nHeight = -1;
		chDistance = 0;
	}
};
struct CameraParam_t
{
	double m_dPitchDeg;
	double m_dFocalLength;
	double m_dCameraHeight;
	double m_dFOVvDeg;
	double m_dFOVhDeg;
	CameraParam_t(){
		m_dPitchDeg = 0.;
		m_dFocalLength = 0.;
		m_dCameraHeight = 0.;
		m_dFOVvDeg = 0.;
		m_dFOVhDeg = 0.;
	}
};
struct StereoCamParam_t
{
	int m_nNumberOfDisp;
	int m_nWindowSize;
	double m_dBaseLine;
	double m_dMaxDist;
	CameraParam_t objCamParam;
	StereoCamParam_t(){
		m_nNumberOfDisp = 80;
		m_nWindowSize = 9;
		m_dBaseLine = 0.;
		m_dMaxDist = 50.0;
	}
};

class CStixelEstimation{
	
private:
	//input
	Mat m_imgLeftInput; //retified image
	Mat m_imgRightInput;//retified image
	int m_nVanishingY;	//Vanishing row coor.
	double m_dPitchDeg;	//camera pitch degree
	int m_nStereoAlg;	//STEREO_BM, STEREO_SGBM
	double m_dBaseLine;	//unit : meter
	int m_nFocalLength; //unit : pixel
	
	//param
	int m_nStixelWidth;
	float m_fScaleFactor;	//It is a inverse num of nStixelWidth
	double m_dMaxDist;	//Maximum distance in program
	int m_nNumberOfDisp;//must be multiple of 16
	int m_nWindowSize;	//odd number
	double m_dGroundVdispSlope;	//The ground line slope in v-disparity
	double m_dGroundVdispOrig;	//The ground line orien in v-disparity
	Size m_sizeSrc;		//source image size
	
	Vector<Point2f> m_vecLinePoint; //v-disparity point : ground point
	Vec4f m_vec4fLine;				//ground line

	StereoCamParam_t objStereoParam;//Camera intrinsic, extrinsic parameter
	StereoBM bm;					//opencv stereoBM class
	StereoSGBM sgbm;				//opencv stereoSGBM class

	//member image
	Mat m_matDisp16;		//16bit disparity
	Mat m_imgGrayDisp8;		//8bit disparity image
	Mat m_imgColorDisp8;	//8bit 3ch disparity image
	Mat m_imgVDisp;			//v-disparity image
	Mat m_imgOriLeft;		//original left image
	Mat m_imgOriRight;		//original right image

	//LUT
	unsigned char m_pseudoColorLUT[256][3]; // RGB pseudo color

public:
	enum { STEREO_BM = 0, STEREO_SGBM = 1 };
	enum { GRAY, COLOR };
	enum { Daimler, KITTI };

	bool m_flgVideo;	//video play flag
	bool m_flgDisplay;	//dispaly flag
	bool m_flgColor;	//color flag

	//output
	stixel_t* m_ptobjStixels;
	vector<stixel_t> vecobjStixels;

	//functions
	CStixelEstimation();
	CStixelEstimation(bool flgPrintHelp);
	~CStixelEstimation();
	void help();
	
	//Set parameters
	int GetVanishingPointY(int nVanishingPointY);				//vanishing point:"y" coor.
	void SetImage(Mat& imgLeftInput, Mat& imgRightInput);		//image setting function of class
	void SetParam();
	void SetParamStereo(int nNumOfDisp = 48 , int nWindowSize = 9, int nStereoAlg = STEREO_BM);
	void SetParam(StereoCamParam_t& objStereoCamParam);
	void SetParam(int nDataSetName);	//open data set parameter set up
	void SetParamOCVStereo();			//Do not use it. It will be private.

	int SetStixelWidth(int nStixelWidth);
	
	void MakePseudoColorLUT();			//pseudo color LUT
	void cvtPseudoColorImage(Mat& srcGray, Mat& dstColor); // input : gray image, output : color image
	
	//Image display
	void Display();
	
	int CreateDisparity();				//make disparity(16bit, 8bit)
	int CreateDisparity(bool flgColor, bool flgDense=0);	//0:gray, 1:pseudo color
	int ImproveDisparity();
	int ComputeVDisparity();
	int RmVDisparityNoise();
	int StoreGroundPoint();				//store line point in v-disparity image : Vector<Point2f> type data
	int FitLineRansac();
	int FilterRansac();
	int GroundEstimation();				//v-disp, line fitting, ransac warping
	int HeightEstimation();
	int StixelDistanceEstimation();
	int StixelDistanceEstimation_col(int col, stixel_t& objStixel);
	int StixelSegmentation();

	int DrawStixelsColor();
	int DrawStixelsGray();

	int CreateStixels(Mat& imgLeftInput, Mat& imgRightInput, bool flgDense = true);
	int StixelEstimation(Mat& imgLeftInput, Mat& imgRightInput, bool flgColor=0);
};