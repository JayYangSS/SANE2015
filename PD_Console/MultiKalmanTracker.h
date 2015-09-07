#pragma once

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <memory>

using namespace std;
using namespace cv;

class CMultiKalmanTracker
{
public:
	CMultiKalmanTracker();
	~CMultiKalmanTracker();

	vector<shared_ptr<KalmanFilter> > vecKF;

	void Initialize(int nDimState, int nDimMeas, int nWidthImg, int nHeightImg);
	bool SetTransitionMat_A(float* data, int nRow, int nCol);
	bool SetProcessNoiseCov_Q(float* data, int nRow, int nCol);
	bool SetMeasurementMat_H(float* data, int nRow, int nCol);
	bool SetMeasurementNoiseCov_R(float* data, int nRow, int nCol);

	void Track(vector<Rect_<int> > &vecrectBB);
	float AssociateWithKF(Rect_<int> &detectedBB, int idx);
	void AddNewObject(Rect_<int> &detectedBB);
	void ConvertStateToRect(Mat &matState, Rect_<int> &rect);
	void ConvertRectToMeasure(Rect_<int> &rect, Mat &matMeasure);

	int m_nDimState = 0;
	int m_nDimMeas = 0;
	float m_fAspectRatio = 20.f/50.f;

	Size2i m_sizeImg;

private:
	bool CheckInitialize();
	Mat m_matTransition;
	Mat m_matProcessNoise;
	Mat m_matMeasurement;
	Mat m_matMeasureNoise;
};

