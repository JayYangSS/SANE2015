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
	int m_nVanishingY;
	double m_dPitchDeg;
	int m_nStereoAlg;
	double m_dBaseLine;	//unit : meter
	int m_nFocalLength; //unit : pixel
	
	//param
	int m_nStixelWidth;
	float m_fScaleFactor;	//It is a inverse num of nStixelWidth
	double m_dMaxDist;	//Maximum distance in program
	int m_nNumberOfDisp;//must be multiple of 16
	int m_nWindowSize;	//odd number
	Size m_sizeSrc;		//source image size
	
	Vector<Point2f> m_vecLinePoint; //v-disparity point : ground point
	Vec4f m_vec4fLine;	//ground line

	StereoCamParam_t objStereoParam;
	StereoBM bm;
	StereoSGBM sgbm;

	//member image
	Mat m_matDisp16;		//16bit disparity
	Mat m_imgGrayDisp8;		//8bit disparity image
	Mat m_imgColorDisp8;	//8bit 3ch disparity image
	Mat m_imgVDisp;			//v-disparity image

	//LUT
	unsigned char m_pseudoColorLUT[256][3]; // RGB

public:
	enum { STEREO_BM = 0, STEREO_SGBM = 1 };
	enum { GRAY, COLOR };
	enum { Daimler, KITTI };

	bool m_flgVideo;	//dispaly flag
	bool m_flgDisplay;	//dispaly flag
	bool m_flgColor;

	//output
	stixel_t* m_ptobjStixels;

	//functions
	CStixelEstimation();
	~CStixelEstimation();
	void help();
	
	//Set parameters
	int GetVanishingPointY(int nVanishingPointY);//vanishing point:"y" coor.
	void SetImage(Mat& imgLeftInput, Mat& imgRightInput);
	void SetParam();
	void SetParamStereo(int nNumOfDisp = 48 , int nWindowSize = 9, int nStereoAlg = STEREO_BM);
	void SetParam(StereoCamParam_t& objStereoCamParam);
	void SetParam(int nDataSetName);	//open data set parameter set up
	void SetParamOCVStereo();			//Do not use it. It will be private.

	void SetStixelWidth(int nStixelWidth);
	
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
	int DrawStixelsColor();
	int DrawStixelsGray();

	int CreateStixels(Mat& imgLeftInput, Mat& imgRightInput, bool flgDense = true);
	int StixelEstimation(Mat& imgLeftInput, Mat& imgRightInput, bool flgColor=0);
};