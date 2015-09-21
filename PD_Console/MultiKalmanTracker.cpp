#include "MultiKalmanTracker.h"

CMultiKalmanTracker::CMultiKalmanTracker()
{
}

CMultiKalmanTracker::~CMultiKalmanTracker()
{
}

void CMultiKalmanTracker::Initialize(int nDimState, int nDimMeas, int nWidthImg, int nHeightImg)
{
	m_nDimState = nDimState;
	m_nDimMeas = nDimMeas;

	m_sizeImg.width = nWidthImg;
	m_sizeImg.height = nHeightImg;
}

bool CMultiKalmanTracker::CheckInitialize()
{
	if (m_nDimMeas == 0 || m_nDimState == 0)
	{
		printf("error : Initialize first !\n");
		return false;
	}
	return true;
}

bool CMultiKalmanTracker::SetTransitionMat_A(float* data, int nRow, int nCol)
{
	if (!CheckInitialize()) return false;

	if (nRow != m_nDimState || nCol != m_nDimState)
	{
		printf("error : Transition matrix must be [%dx%d]\n", m_nDimState,m_nDimState);
		return false;
	}

	m_matTransition = Mat::Mat(nRow, nCol, CV_32FC1, data);

	return true;
}

bool CMultiKalmanTracker::SetProcessNoiseCov_Q(float* data, int nRow, int nCol)
{
	if (!CheckInitialize()) return false;

	if (nRow != m_nDimState || nCol != m_nDimState)
	{
		printf("error : Process Noise Covariance matrix must be [%dx%d]\n", m_nDimState, m_nDimState);
		return false;
	}

	m_matProcessNoise = Mat::Mat(nRow, nCol, CV_32FC1, data);

	return true;
}

bool CMultiKalmanTracker::SetMeasurementMat_H(float* data, int nRow, int nCol)
{
	if (!CheckInitialize()) return false;

	if (nRow != m_nDimMeas || nCol != m_nDimState)
	{
		printf("error : Measurement matrix must be [%dx%d]\n", m_nDimMeas, m_nDimState);
		return false;
	}

	m_matMeasurement = Mat::Mat(nRow, nCol, CV_32FC1, data);

	return true;
}

bool CMultiKalmanTracker::SetMeasurementNoiseCov_R(float* data, int nRow, int nCol)
{
	if (!CheckInitialize()) return false;

	if (nRow != m_nDimMeas || nCol != m_nDimMeas)
	{
		printf("error : Measurement Noise Covariance matrix must be [%dx%d]\n", m_nDimMeas, m_nDimMeas);
		return false;
	}

	m_matMeasureNoise = Mat::Mat(nRow, nCol, CV_32FC1, data);

	return true;
}

void CMultiKalmanTracker::Track(vector<Rect_<int> > &vecrectBB)
{
	for (int i = 0; i < (int)vecKF.size(); i++)
	{
		vecKF[i]->predict();

		float fMaxOverlap = 0;
		int idxMax = -1;
		for (int j = 0; j < (int)vecrectBB.size(); j++)
		{
			float fOverlap = AssociateWithKF(vecrectBB[j], i);
			if (fOverlap > fMaxOverlap)
			{
				fMaxOverlap = fOverlap;
				idxMax = j;
			}
		}

		if (idxMax != -1)
		{
			Mat meas = Mat::zeros(m_nDimMeas, 1, CV_32FC1);
			ConvertRectToMeasure(vecrectBB[idxMax], meas);
			vecKF[i]->correct(meas);

			vecrectBB.erase(vecrectBB.begin() + idxMax);
		}
	}

	for (int i = 0; i < (int)vecrectBB.size(); i++)
	{
		AddNewObject(vecrectBB[i]);
	}

}

float CMultiKalmanTracker::AssociateWithKF(Rect_<int> &detectedBB, int idx)
{
	Rect_<int> rectKF;
	ConvertStateToRect(vecKF[idx]->statePre, rectKF);

	float fOverlap = (float)((detectedBB & rectKF).area()) / (float)((detectedBB | rectKF).area());
	
	return fOverlap;
}

void CMultiKalmanTracker::AddNewObject(Rect_<int> &detectedBB)
{
	shared_ptr<KalmanFilter> pKF(new KalmanFilter);
	
	pKF->init(m_nDimState, m_nDimMeas);
	
	m_matTransition.copyTo(pKF->transitionMatrix);       // A
	m_matMeasurement.copyTo(pKF->measurementMatrix);     // H
	m_matProcessNoise.copyTo(pKF->processNoiseCov);      // Q
	m_matMeasureNoise.copyTo(pKF->measurementNoiseCov);  // R

	Point2f ptCenter(detectedBB.br() + detectedBB.tl());
	ptCenter.x /= 2;
	ptCenter.y /= 2;

	int tx = ptCenter.x - m_sizeImg.width / 2;
	
	vector<float> vecfProportion;
	vecfProportion.push_back(0.01f); //////////////////////////////////////////////////////////////////////////
	vecfProportion.push_back(0.01f); //////////////////////////////////////////////////////////////////////////
	vecfProportion.push_back(0.01f); //////////////////////////////////////////////////////////////////////////

	pKF->statePost.at<float>(0) = ptCenter.x;
	pKF->statePost.at<float>(1) = ptCenter.y;
	pKF->statePost.at<float>(2) = detectedBB.height;
	pKF->statePost.at<float>(3) = tx*vecfProportion[0];
	pKF->statePost.at<float>(4) = tx*vecfProportion[1];
	pKF->statePost.at<float>(5) = tx*vecfProportion[2];

	vecKF.push_back(pKF);
}

void CMultiKalmanTracker::ConvertStateToRect(Mat &matState, Rect_<int> &rect)
{
	rect.height = matState.at<float>(2);
	rect.width = (int)((float)(rect.height) * m_fAspectRatio);
	rect.y = matState.at<float>(1) - rect.height / 2;
	rect.x = matState.at<float>(0) - rect.width / 2;
}

void CMultiKalmanTracker::ConvertRectToMeasure(Rect_<int> &rect, Mat &matMeasure)
{
	Point2i ptCenter(rect.br() + rect.tl());
	ptCenter.x /= 2;
	ptCenter.y /= 2;

	matMeasure.at<float>(0) = ptCenter.x;
	matMeasure.at<float>(1) = ptCenter.y;
	matMeasure.at<float>(2) = rect.height;
}
