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



int CreateDisparity(){

	return 0;
}