/*
*  Class StixelEstimation.cpp
*  This class creates STIXEL using disparity image.
*
*  Created by T.K.Woo on Aug/29/2015.
*  Copyright 2015 CVLAB at Inha. All rights reserved.
*
*/
#include"StixelEstimation.h"

CStixelEstimation::CStixelEstimation()
{
	m_dBaseLine = 0.;
	m_nFocalLength = 0;
	m_nVanishingY = 0;
	m_dPitchDeg = 0.;
	m_dMaxDist = 30.;
	m_nNumberOfDisp = 160;
	m_nStereoAlg = STEREO_SGBM;
	m_flgColor = false;
	m_flgDisplay = true;
	m_flgVideo = false;

	help();
	MakePseudoColorLUT();
}
CStixelEstimation::~CStixelEstimation(){

	delete m_ptobjStixels;
}

void CStixelEstimation::help()
{
	
}

//
void CStixelEstimation::MakePseudoColorLUT()
{
	int b = 125;
	int g = 0;
	int r = 0;

	int idx = 0;

	int mode = 0;
	// mode = 0 : increasing 'b'
	// mode = 1 : increasing 'g'
	// mode = 2 : decreasing 'b'
	// mode = 3 : increasing 'r'
	// mode = 4 : decreasing 'g'
	// mode = 5 : decreasing 'r'

	while (1)
	{
		m_pseudoColorLUT[idx][0] = b;
		m_pseudoColorLUT[idx][1] = g;
		m_pseudoColorLUT[idx][2] = r;

		if (b == 255 && g == 0 && r == 0)
			mode = 1;
		else if (b == 255 && g == 255 && r == 0)
			mode = 2;
		else if (b == 0 && g == 255 && r == 0)
			mode = 3;
		else if (b == 0 && g == 255 && r == 255)
			mode = 4;
		else if (b == 0 && g == 0 && r == 255)
			mode = 5;

		switch (mode)
		{
		case 0: b += 5; break;
		case 1: g += 5; break;
		case 2: b -= 5; break;
		case 3: r += 5; break;
		case 4: g -= 5; break;
		case 5: r -= 5; break;
		default: break;
		}

		if (idx == 255)
			break;

		idx++;
	}
}
void CStixelEstimation::cvtPseudoColorImage(Mat& srcGray, Mat& dstColor)
{
	for (int i = 0; i<srcGray.rows; i++)
	{
		for (int j = 0; j<srcGray.cols; j++)
		{
			unsigned char val = srcGray.data[i*srcGray.cols + j];
			if (val == 0) continue;
			dstColor.data[(i*srcGray.cols + j) * 3 + 0] = m_pseudoColorLUT[val][0];
			dstColor.data[(i*srcGray.cols + j) * 3 + 1] = m_pseudoColorLUT[val][1];
			dstColor.data[(i*srcGray.cols + j) * 3 + 2] = m_pseudoColorLUT[val][2];
		}
	}
}

int CStixelEstimation::GetVanishingPointY(int nVanishingPointY){

	m_nVanishingY = nVanishingPointY;
	m_dPitchDeg = fastAtan2(nVanishingPointY - m_sizeSrc.height / 2, m_nFocalLength);
}
void CStixelEstimation::SetParam()
{
	m_dBaseLine = 0.;
	m_nFocalLength = 0;
	m_dPitchDeg = 0.;
	m_dMaxDist = 30.;
	m_nNumberOfDisp = 160;
	m_nStereoAlg = STEREO_SGBM;
	m_nStixelWidth = 1;
}
void CStixelEstimation::SetParamStereo(int nNumOfDisp, int nWindowSize, int nStereoAlg)
{
	m_nNumberOfDisp = nNumOfDisp;
	m_nWindowSize = nWindowSize;
	m_nStereoAlg = nStereoAlg;

	SetParamOCVStereo();
}
void CStixelEstimation::SetParam(int nDataSetName)
{
	if (nDataSetName == Daimler){
		m_dBaseLine = 0.25;
		m_nFocalLength = 1200;
		m_dPitchDeg = -1.89;
		m_dMaxDist = 60.;
		m_nNumberOfDisp = 48;
		m_nStereoAlg = STEREO_BM;
		m_nWindowSize = 11;
		m_sizeSrc = Size(640, 480);
	}
	else if (nDataSetName == KITTI){
		printf("Not available\n");
	}
	else
		printf("Just Daimler avaiilable\n");

	SetParamOCVStereo();
	m_ptobjStixels = new stixel_t[640];
}
void CStixelEstimation::SetParamOCVStereo()
{
	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = m_nWindowSize > 0 ? m_nWindowSize : 11;
	bm.state->minDisparity = 1;
	bm.state->numberOfDisparities = m_nNumberOfDisp;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 25;//9;
	bm.state->speckleRange = 32;//4;
	bm.state->disp12MaxDiff = 1;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = m_nWindowSize > 0 ? m_nWindowSize : 7;

	int cn = 1;//imgLeftInput.channels();

	sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 1;
	sgbm.numberOfDisparities = m_nNumberOfDisp;//numberOfDisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = bm.state->speckleWindowSize;
	sgbm.speckleRange = bm.state->speckleRange;
	sgbm.disp12MaxDiff = 1;
}

void CStixelEstimation::SetImage(Mat& imgLeftInput, Mat& imgRightInput)
{
	m_imgLeftInput = imgLeftInput;
	m_imgRightInput = imgRightInput;
}

void CStixelEstimation::Display()
{
	imshow("Left Input", m_imgLeftInput);
	if(!m_imgGrayDisp8.empty()) imshow("Disparity", m_imgGrayDisp8);
	if(!m_imgColorDisp8.empty()) imshow("Disparity Color", m_imgColorDisp8);
	if(!m_imgVDisp.empty()) imshow("VDisparity", m_imgVDisp);

	if (m_flgVideo){
		if (waitKey(1) == 27){
			m_flgVideo = false;
			return;
		}
	}
	else waitKey(0);
}

int CStixelEstimation::CreateDisparity()
{
	if (m_nStereoAlg == STEREO_BM){
		bm(m_imgLeftInput, m_imgRightInput, m_matDisp16, CV_16S);
	}
	else if (m_nStereoAlg == STEREO_SGBM){
		sgbm(m_imgLeftInput, m_imgRightInput, m_matDisp16);
	}
	m_matDisp16.convertTo(m_imgGrayDisp8, CV_8U, 255 / (m_nNumberOfDisp*16.));

	return 0;
}
int CStixelEstimation::CreateDisparity(bool flgColor, bool flgDense)
{
	m_flgColor = flgColor;
	if (m_flgColor != 1 && m_flgColor != 0){
		printf("Flag is wrong\n");
		return -1;
	}
	if (m_nStereoAlg == STEREO_BM){
		bm(m_imgLeftInput, m_imgRightInput, m_matDisp16, CV_16S);
	}
	else if (m_nStereoAlg == STEREO_SGBM){
		sgbm(m_imgLeftInput, m_imgRightInput, m_matDisp16);
	}
	m_matDisp16.convertTo(m_imgGrayDisp8, CV_8U, 255 / (m_nNumberOfDisp*16.));

	if (flgDense) ImproveDisparity();	
	if (m_flgColor == 1){
		Mat imgTemp;
		cvtColor(m_imgLeftInput, imgTemp, CV_GRAY2BGR);
		cvtColor(m_imgGrayDisp8, m_imgColorDisp8, CV_GRAY2BGR);
		cvtPseudoColorImage(m_imgGrayDisp8, m_imgColorDisp8);
		addWeighted(m_imgColorDisp8, 0.5, imgTemp, 0.5, 0.0, m_imgColorDisp8);
	}

	return 0;
}
int CStixelEstimation::ImproveDisparity(){
	uchar chTempCur = 0;
	uchar chTempPrev = 0;
	for (int v = 0; v < m_imgGrayDisp8.rows; v++){
		for (int u = m_nNumberOfDisp; u < m_imgGrayDisp8.cols; u++){
			chTempCur = m_imgGrayDisp8.at<uchar>(v, u);
			if (chTempCur == 0) m_imgGrayDisp8.at<uchar>(v, u) = chTempPrev;
			else chTempPrev = chTempCur;
		}
	}
	return 0;
}
int CStixelEstimation::ComputeVDisparity()
{
	int maxDisp = 255;
	m_imgVDisp = Mat(m_imgGrayDisp8.rows, 255, CV_8U, Scalar(0));
	for (int u = 0; u<m_imgGrayDisp8.rows; u++){
		if (u < 200) continue; // we are finding ground. therefore we check pixels below vanishing point 
		for (int v = 0; v<m_imgGrayDisp8.cols; v++){
			int disp = (m_imgGrayDisp8.at<uchar>(u, v));// / 8;
			//if(disp>0 && disp < maxDisp){
			if (disp>6 && disp < maxDisp - 2){ //We remove pixels of sky and car to compute the roadline
				m_imgVDisp.at<unsigned char>(u, disp) += 1;
			}
		}
	}
	return 0;

}
int CStixelEstimation::RmVDisparityNoise()
{
	int nThresh = 50;
	threshold(m_imgVDisp, m_imgVDisp, nThresh, 255, 3);
	return 0;
}
int CStixelEstimation::StoreGroundPoint()
{
	m_vecLinePoint.clear();
	for (int u = 200; u<m_imgVDisp.rows; u++){//200 is the vanishing row in image : It will be fixed 150901
		for (int v = 0; v<m_imgVDisp.cols; v++){
			int value = m_imgVDisp.at<unsigned char>(u, v);
			if (value > 0){
				m_vecLinePoint.push_back(Point2f(u, v));
			}
		}
	}
	return 0;
}
int CStixelEstimation::FitLineRansac()
{
	int iterations = 100;
	double sigma = 1.;
	double a_max = 7.;

	int n = m_vecLinePoint.size();
	//cout <<"point size : "<< n << endl;
	if (n<2)
	{
		printf("Points must be more than 2 EA\n");
		return -1;
	}

	RNG rng;
	double bestScore = -1.;
	for (int k = 0; k<iterations; k++)
	{
		int i1 = 0, i2 = 0;
		double dx = 0;
		while (i1 == i2)
		{
			i1 = rng(n);
			i2 = rng(n);
		}
		Point2f p1 = m_vecLinePoint[i1];
		Point2f p2 = m_vecLinePoint[i2];

		Point2f dp = p2 - p1;
		dp *= 1. / norm(dp);
		double score = 0;

		if (fabs(dp.x / 1.e-5f) && fabs(dp.y / dp.x) <= a_max)
		{
			for (int i = 0; i<n; i++)
			{
				Point2f v = m_vecLinePoint[i] - p1;
				double d = v.y*dp.x - v.x*dp.y;
				score += exp(-0.5*d*d / (sigma*sigma));
			}
		}
		if (score > bestScore)
		{
			m_vec4fLine = Vec4f(dp.x, dp.y, p1.x, p1.y);
			bestScore = score;
		}
	}

	return 0;
}
int CStixelEstimation::FilterRansac()
{
	double slope = m_vec4fLine[0] / m_vec4fLine[1];
	double orig = m_vec4fLine[2] - slope*m_vec4fLine[3];
	//printf("v=%lf * d + %lf\n", slope, orig); // print line eq.
	//slope = -0.7531;
	//orig = 200.;
	for (int u = 200; u<m_imgGrayDisp8.rows; u++){//200 is the vanishing row in image : It will be fixed 150901
		for (int v = 0; v<m_imgGrayDisp8.cols; v++){
			int value = m_imgGrayDisp8.at<unsigned char>(u, v);
			double test = orig + slope*value - u;
			if (test > 15){
				m_imgGrayDisp8.at<unsigned char>(u, v) = value;
				//res.at<unsigned char>(u, v) = value;
			}
			else{
				m_imgGrayDisp8.at<unsigned char>(u, v) = 0;
				//res.at<unsigned char>(u, v) = 0;
			}
		}
	}
	return 0;
}
int CStixelEstimation::GroundEstimation()
{
	if (m_imgGrayDisp8.empty()){
		printf("Disparity is empty\n");
		return -1;
	}
	ComputeVDisparity();
	RmVDisparityNoise();
	StoreGroundPoint();
	FitLineRansac();
	FilterRansac();

	return 0;
}
int CStixelEstimation::HeightEstimation()
{
	double slope = -0.5016;
	double orig = 191.696210;

	for (int u = 0; u<m_imgGrayDisp8.rows; u++){
		for (int v = 0; v<m_imgGrayDisp8.cols; v++){
			int value = m_imgGrayDisp8.at<unsigned char>(u, v);
			//double test = orig + slope*value - u;
			if (u < (orig + slope*value)){
				//img.at<unsigned char>(u, v) = value;
				m_imgGrayDisp8.at<unsigned char>(u, v) = 0;
				//res.at<unsigned char>(u, v) = value;
			}
			//else{
			//	img.at<unsigned char>(u, v) = 0;
			//	//res.at<unsigned char>(u, v) = 0;
			//}
		}
	}
	return 0;
}
int CStixelEstimation::StixelDistanceEstimation()
{
	for (int u = 0; u < m_imgGrayDisp8.cols; u++){
		if (u < 30) { //Removal manually left 30 cols because of paralax
			m_ptobjStixels[u].chDistance = 0;
			m_ptobjStixels[u].nGround = 0;
			m_ptobjStixels[u].nHeight = 0;
		}
		else StixelDistanceEstimation_col(u, m_ptobjStixels[u]);
	}
	return 0;
}
int CStixelEstimation::StixelDistanceEstimation_col(int col, stixel_t& objStixel)
{
	int nIter = m_imgGrayDisp8.rows / 2;
	uchar chDisp;

	for (int v = 1; v < nIter; v++){
		chDisp = m_imgGrayDisp8.at<uchar>(m_imgGrayDisp8.rows - v, col);

		if (m_imgGrayDisp8.at<uchar>(v, col)>0 && objStixel.nHeight == -1){ objStixel.nHeight = v; nIter = m_imgGrayDisp8.rows - v; }
		if (chDisp > 0 && objStixel.nGround == -1){
			objStixel.nGround = m_imgGrayDisp8.rows - v + 10; //10 is manually
			objStixel.chDistance = chDisp; // 2015.08.11 have to fix
			nIter = m_imgGrayDisp8.rows - v;
		}
	}
	//cout << col << " : " << objStixel.nGround << ", " << objStixel.nHeight << endl;
	
	return 0;
}
int CStixelEstimation::DrawStixelsColor()
{
	DrawStixelsGray();
	if (m_flgColor == 1){
		Mat imgTemp;
		cvtColor(m_imgLeftInput, imgTemp, CV_GRAY2BGR);
		cvtColor(m_imgGrayDisp8, m_imgColorDisp8, CV_GRAY2BGR);
		cvtPseudoColorImage(m_imgGrayDisp8, m_imgColorDisp8);
		addWeighted(m_imgColorDisp8, 0.5, imgTemp, 0.5, 0.0, m_imgColorDisp8);
	}

	/*for (int u = 0; u < m_imgColorDisp8.cols; u++){
		line(m_imgColorDisp8,
			Point(u, m_ptobjStixels[u].nGround),
			Point(u, m_ptobjStixels[u].nHeight),
			Scalar(0, 255 - m_ptobjStixels[u].chDistance, m_ptobjStixels[u].chDistance));
	}*/
	return 0;
}
int CStixelEstimation::DrawStixelsGray()
{
	for (int u = 0; u < m_imgGrayDisp8.cols; u++){
		line(m_imgGrayDisp8,
			Point(u, m_ptobjStixels[u].nGround),
			Point(u, m_ptobjStixels[u].nHeight),
			Scalar(m_ptobjStixels[u].chDistance));
	}
	return 0;
}

int CStixelEstimation::CreateStixels(Mat& imgLeftInput, Mat& imgRightInput)
{
	SetImage(imgLeftInput, imgRightInput);

}