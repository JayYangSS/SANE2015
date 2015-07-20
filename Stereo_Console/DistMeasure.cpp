/*
*  Class DistMeasure.cpp
*  measure the distace of detected object
*
*  Created by T.K.Woo on July/17/2015.
*  Copyright 2015 CVLAB at Inha. All rights reserved.
*
*/
#include"DistMeasure.h"

//default constructor
CDistMeasure::CDistMeasure()
{
	m_dBaseLine = 0.;
	m_dFocalLength = 0.;
	m_nVanishingY = 0;
	m_dPitchDeg = 0.;
	m_dBoundDist = 20.;
	m_dMaxDist = 50.;
	m_nNumberOfDisp = 48;
	m_nStereoAlg = STEREO_BM;
	m_nDistAlg = FVLM;

	help();
}
//how to use this class
void CDistMeasure::help()
{
	cout << "This class measures the distance of detected object." << endl;
	cout << "Please Follow the steps." << endl;
	cout << "1. SetParam(...)\n2. CalcDist()" << endl;
	cout << "The parameters that are imgLeft, imgRight, ROI seq, baseline and focal length is absolutely necessary" << endl;
	cout << "And you can control the each other parameters. Good Luck" << endl;
}
//set up parameters
void CDistMeasure::SetImage(Mat& imgLeft, Mat& imgRight, vector<Rect_<int> >& vecrectRoi)
{
	m_imgLeftInput = imgLeft.clone();
	m_imgRightInput = imgRight.clone();
	cvtColor(m_imgLeftInput, m_imgGT, CV_GRAY2BGR);
	m_sizeSrc = m_imgLeftInput.size();

	m_vecrectRoi = vecrectRoi;
}
void CDistMeasure::SetParam(Mat& imgLeft, Mat& imgRight, vector<Rect_<int> >& vecrectRoi, double dBaseLine, double dFocalLength,
	int nVanishingY, double dPitchDeg,
	int nNumOfDisp, int nWindowSize, int nStereoAlg, int nDistAlg,	//number of disparity=48, stereo alg=SAD block matching, dist alg=FVLM
	double dBoundDist, double dMaxDist //unit : meters
	)
{
	m_imgLeftInput = imgLeft.clone();
	m_imgRightInput = imgRight.clone();
	cvtColor(m_imgLeftInput, m_imgGT, CV_GRAY2BGR);
	m_sizeSrc = m_imgLeftInput.size();

	m_vecrectRoi = vecrectRoi;
	m_dBaseLine = dBaseLine;
	m_dFocalLength = dFocalLength;
	m_nVanishingY = nVanishingY;
	m_dPitchDeg = dPitchDeg;

	m_nNumberOfDisp = nNumOfDisp;
	m_nWindowSize = nWindowSize;
	m_nStereoAlg = nStereoAlg;
	m_nDistAlg = nDistAlg;
	m_dBoundDist = dBoundDist;
	m_dMaxDist = dMaxDist;
	m_flgVideo = false;
	m_flgDisplay = true;
}
void CDistMeasure::SetParam(double dBaseLine, double dFocalLength,
	int nVanishingY, double dPitchDeg,
	int nNumOfDisp, int nWindowSize, int nStereoAlg, int nDistAlg,	//number of disparity=48, stereo alg=SAD block matching, dist alg=FVLM
	double dBoundDist, double dMaxDist //unit : meters
	)
{
	m_dBaseLine = dBaseLine;
	m_dFocalLength = dFocalLength;
	m_nVanishingY = nVanishingY;
	m_dPitchDeg = dPitchDeg;

	m_nNumberOfDisp = nNumOfDisp;
	m_nWindowSize = nWindowSize;
	m_nStereoAlg = nStereoAlg;
	m_nDistAlg = nDistAlg;
	m_dBoundDist = dBoundDist;
	m_dMaxDist = dMaxDist;
	m_flgVideo = false;
	m_flgDisplay = true;
}
//Calculate distance function
int CDistMeasure::CalcDistImg(int nflag) // Flag seq : FVLM, MONO, STEREOBM, STEREOSGBM
{
	for (int i = 0; i < m_vecrectRoi.size(); i++){
		double dDistTemp = 0;
		if (nflag == FVLM)
			CalcDistRoi_FVLM(m_vecrectRoi[i], dDistTemp, nflag);
		else if (nflag == MONO)
			CalcDistRoi_mono(m_vecrectRoi[i], dDistTemp);
		else if (nflag == STEREOBM)
			CalcDistRoi_stereo(m_vecrectRoi[i], dDistTemp, nflag);
		else if (nflag == STEREOSGBM)
			CalcDistRoi_stereo(m_vecrectRoi[i], dDistTemp, nflag);
		else{ cout << "distance measurement method error" << endl; return 1; }

		m_vecdDistance.push_back(dDistTemp);

		cout << "#" << i << " Distance : " << dDistTemp << "m" << endl;

		if (m_flgDisplay) {
			rectangle(m_imgGT, m_vecrectRoi[i], CV_RGB(0, 255, 0), 2);
			Display();
		}
	}
	return 0;
}
int CDistMeasure::CalcDistRoi_mono(Rect_<int>& rectRoi, double& dDistance)
{
	dDistance = 1.17*tan((78.69 + m_dPitchDeg + 0.047125*(m_imgLeftInput.rows - (rectRoi.y + rectRoi.height)))*PI / 180);
	return 0;
}
int CDistMeasure::CalcDistRoi_stereo(Rect_<int>& rectRoi, double& dDistance, int nflag)
{
	if (nflag == STEREOBM) m_nStereoAlg = STEREO_BM;
	else if (nflag == STEREOSGBM) m_nStereoAlg = STEREO_SGBM;
	else if (nflag == FVLM) m_nStereoAlg = STEREO_BM;
	else { cout << "Stereo method error" << endl; return 1; }

	Rect_<int> rectRoiTemp(rectRoi);//?

	StereoBM bm;
	StereoSGBM sgbm;

	if (m_nStereoAlg == STEREO_BM){
		if (rectRoi.x < m_nNumberOfDisp){
			rectRoiTemp.x = 0;
			rectRoiTemp.width = rectRoi.width + m_nNumberOfDisp;
		}
		else{
			rectRoiTemp.x = rectRoi.x - m_nNumberOfDisp;
			rectRoiTemp.width = rectRoi.width + m_nNumberOfDisp;
		}
		//bm.state->roi1 = roi1;
		//bm.state->roi2 = roi2;
		bm.state->preFilterCap = 31;
		bm.state->SADWindowSize = m_nWindowSize > 0 ? m_nWindowSize : 13;
		bm.state->minDisparity = 1;
		bm.state->numberOfDisparities = m_nNumberOfDisp;
		bm.state->textureThreshold = 10;
		bm.state->uniquenessRatio = 15;
		bm.state->speckleWindowSize = 25;//9;
		bm.state->speckleRange = 32;//4;
		bm.state->disp12MaxDiff = 1;
	}
	else if (m_nStereoAlg == STEREO_SGBM){
		if (rectRoi.x < m_nNumberOfDisp){
			rectRoiTemp.x = 0;
			rectRoiTemp.width = rectRoi.width + m_nNumberOfDisp;
		}
		else{
			rectRoiTemp.x = rectRoi.x - m_nNumberOfDisp;
			rectRoiTemp.width = rectRoi.width + m_nNumberOfDisp;
		}

		sgbm.preFilterCap = 63;
		sgbm.SADWindowSize = m_nWindowSize > 0 ? m_nWindowSize : 7;

		int cn = m_imgLeftInput.channels();

		sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
		sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
		sgbm.minDisparity = 1;
		sgbm.numberOfDisparities = 48;//numberOfDisparities;
		sgbm.uniquenessRatio = 10;
		sgbm.speckleWindowSize = bm.state->speckleWindowSize;
		sgbm.speckleRange = bm.state->speckleRange;
		sgbm.disp12MaxDiff = 1;
	}
	else{
		cout << "Stereo algorithm error" << endl;
		return 1;
	}

	m_imgRoiLeftTemp = m_imgLeftInput(rectRoiTemp);
	m_imgRoiRightTemp = m_imgRightInput(rectRoiTemp);
	Mat imgDisp8Temp;

	if (m_nStereoAlg == STEREO_BM)
		bm(m_imgRoiLeftTemp, m_imgRoiRightTemp, m_matDisp, CV_16S);
	else if (m_nStereoAlg == STEREO_SGBM)
		sgbm(m_imgRoiLeftTemp, m_imgRoiRightTemp, m_matDisp);

	m_matDisp.convertTo(imgDisp8Temp, CV_8U, 255 / (m_nNumberOfDisp*16.));
	m_imgDisp8 = imgDisp8Temp(Rect(m_nNumberOfDisp, 0, rectRoi.width, rectRoi.height));

	DispToHist(dDistance);

	return 0;
}
int CDistMeasure::CalcDistRoi_FVLM(Rect_<int>& rectRoi, double& dDistance, int nflag)
{
	CalcDistRoi_mono(rectRoi, dDistance);
	cout << dDistance << endl;
	if (dDistance > m_dBoundDist)
		CalcDistRoi_stereo(rectRoi, dDistance, STEREOBM);
	if (dDistance > m_dMaxDist)
		dDistance = m_dMaxDist;

	return 0;
}
//disparity histogram in disparity ROI
int CDistMeasure::DispToHist(double& dDistance)
{
	// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;
	float mean = 0;
	float sum = 0;

	/// Compute the histograms:
	calcHist(&m_imgDisp8, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	hist.at<float>(0, 1) = 0;
	double maxVal = 0, minVal = 0;
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	Mat histImg(256, 256, CV_8U, Scalar(255));

	int hpt = static_cast<int>(0.9 * 256);//(5*256);//(10*256);
	float max = 0;
	int disp_of_object = 0;
	float nhist_aver = 0;

	for (int h = 0; h<208; h++)// 208 : 5m
	{
		if (h<4) hist.at<float>(h) = 0;//|| h>numberOfDisparities
		float binVal = hist.at<float>(h);
		nhist_aver += hist.at<float>(h) / 208;
		if (h>2 && binVal >= max) { max = binVal; disp_of_object = h; }
		int intensity = static_cast<int>(binVal*hpt / maxVal);
		line(histImg, Point(h, 255), Point(h, 255 - intensity), Scalar::all(0));
	}
	//cout << "max : " << max << ", aver : " << nhist_aver << endl;

	if ((double)max / ((double)nhist_aver - (double)max / 153) > 4.)
	{
		//cout << "h : " << disp_of_object << endl;
		//cout << "disparity : " << (double)disp_of_object*(double)nDisp/255+5 << endl;
		dDistance = (double)(m_dBaseLine*m_dFocalLength / ((double)(disp_of_object)*(double)m_nNumberOfDisp / 255));
		//cout << "distance of object : " << *dDistance << "m" << endl;
		//fprintf(f_distance,"%lf\n",(double)(BASELINE*FOCAL/((double)disp_of_object*(double)numberOfDisparities/255+7)));
	}
	else
	{
		cout << "can't find a object " << endl;
		dDistance = 0;
		//fprintf(f_distance,"%d\n", 0);
	}
	m_imgHist = histImg;
	return 0;
}
//dispaly mid processing images
void CDistMeasure::Display()
{
	imshow("imgLeft", m_imgLeftInput);
	imshow("imgRight", m_imgRightInput);
	imshow("imgGT", m_imgGT);
	//imshow("imgRoiLeft", m_imgRoiLeftTemp);
	//imshow("imgRoiRight", m_imgRoiRightTemp);
	//imshow("imgDisp8", m_imgDisp8);
	//imshow("imgHist", m_imgHist);
	if (m_flgVideo == false) {
		if (waitKey(0) == 't') m_flgVideo = !m_flgVideo;
	}
	else{
		if (waitKey(1) == 't') m_flgVideo = !m_flgVideo;
	}
}
