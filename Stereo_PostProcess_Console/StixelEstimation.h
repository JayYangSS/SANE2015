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
	double m_dMaxDist;	//Maximum distance in program
	int m_nNumberOfDisp;//must be multiple of 16
	int m_nWindowSize;	//odd number
	Size m_sizeSrc;		//source image size
	bool m_flgVideo;	//dispaly flag
	bool m_flgDisplay;	//dispaly flag
	Vector<Point2f> m_vecLinePoint; //v-disparity point : ground point
	Vec4f m_vec4fLine;	//ground line

	//member image
	Mat m_matDisp16;		//16bit disparity
	Mat m_imgGrayDisp8;		//8bit disparity image
	Mat m_imgColorDisp8;	//8bit 3ch disparity image

	//LUT
	unsigned char m_pseudoColorLUT[256][3]; // RGB

public:
	enum { STEREO_BM = 0, STEREO_SGBM = 1 };

	//output
	stixel_t* m_ptobjStixels;

	//functions
	CStixelEstimation();
	void help();
	void SetImage();
	void SetParam();
	
	void MakePseudoColorLUT(); // pseudo color LUT
	void cvtPseudoColorImage(Mat& srcGray, Mat& dstColor); // input : gray image, output : color image
	void Display();

	int CreateDisparity();//make disparity(16bit, 8bit)
	int ComputeVDisparity();
	int RmVDisparityNoise();
	int StoreGroundPoint();//Vector<Point2f>
	int FitLineRansac();
	int filterRansac();
	int GroundEstimation();//v-disp, line fitting, ransac warping
	int HeightEstimation();
	int StixelDistanceEstimation();

	int CreateStixels();
};