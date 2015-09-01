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

	help();
	MakePseudoColorLUT();
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
void CStixelEstimation::SetParam(){
	m_dBaseLine = 0.;
	m_nFocalLength = 0;
	m_dPitchDeg = 0.;
	m_dMaxDist = 30.;
	m_nNumberOfDisp = 160;
	m_nStereoAlg = STEREO_SGBM;
	m_nStixelWidth = 1;
}
void CStixelEstimation::SetParamStereo(int nNumOfDisp, int nWindowSize, int nStereoAlg){
	m_nNumberOfDisp = nNumOfDisp;
	m_nWindowSize = nWindowSize;
	m_nStereoAlg = nStereoAlg;

	SetParamOCVStereo();
}
void CStixelEstimation::SetParam(int nDataSetName){
	if (nDataSetName == Daimler){
		m_dBaseLine = 0.25;
		m_nFocalLength = 1200;
		m_dPitchDeg = -1.89;
		m_dMaxDist = 60.;
		m_nNumberOfDisp = 48;
		m_nStereoAlg = STEREO_SGBM;
		m_nWindowSize = 7;
		m_sizeSrc = Size(640, 480);
	}
	else if (nDataSetName == KITTI){
		printf("Not available\n");
	}
	else
		printf("Just Daimler avaiilable\n");

	SetParamOCVStereo();
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

void CStixelEstimation::SetImage(Mat& imgLeftInput, Mat& imgRightInput){
	m_imgLeftInput = imgLeftInput;
	m_imgRightInput = imgRightInput;
}

int CStixelEstimation::CreateDisparity(){
	if (m_nStereoAlg == STEREO_BM){
		bm(m_imgLeftInput, m_imgRightInput, m_matDisp16, CV_16S);
	}
	else if (m_nStereoAlg == STEREO_SGBM){
		sgbm(m_imgLeftInput, m_imgRightInput, m_matDisp16);
	}
	m_matDisp16.convertTo(m_imgGrayDisp8, CV_8U, 255 / (m_nNumberOfDisp*16.));

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
