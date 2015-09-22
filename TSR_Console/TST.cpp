#include "TST.h"
#include "TSD.h"
//
//#ifndef Angle
//#define Angle
//float AngleTransform(const float &tempAngle, const int &scale){
//	return (float)CV_PI / 180 * (tempAngle*scale);
//}
//float fAngle(int x, int y)
//{
//	return ToDegree(atan2f(y, x));
//}
//#endif
float AngleTransform(const float &tempAngle, const int &scale){
	return (float)CV_PI / 180 * (tempAngle*scale);
}
float fAngle(int x, int y)
{
	return ToDegree(atan2f(y, x));
}
CTST::CTST()
{
}


CTST::CTST(CTSD& adapDetect)
:m_adaptiveDetection(adapDetect)
{

}

CTST::~CTST()
{
}

void CTST::SetVanishingPT(Point pt)
{
	m_ptVanishing = pt;
}

Point CTST::GetVanishingPT()
{
	return m_ptVanishing;
}

void CTST::kalmanMultiTarget(Mat& srcImage, vector<Rect>& vecRectTracking, SVecTracking& Set, vector<SKalman>& MultiKF, float scaledist, int cntCandidate, int cntBefore, int frameCandiate, int& cntframe, Rect& ROIset, vector<Rect>& vecValidRec, Mat& imgROImask)
{
	Point pCurrent;

	double time6 = (double)getTickCount();

	for (int i = 0; i < Set.vecBefore.size(); i++)
	{
		pCurrent.x = srcImage.cols;
		pCurrent.y = srcImage.rows;
		int num = -1;

		float fDistXbefore = m_ptVanishing.x - (Set.vecBefore[i].x + Set.vecBefore[i].width / 2);
		float fDistYbefore = m_ptVanishing.y - (Set.vecBefore[i].y + Set.vecBefore[i].height / 2);
		float fDistBefore = fDistXbefore*fDistXbefore + fDistYbefore*fDistYbefore;
		fDistBefore = sqrt(fDistBefore);
		float fAngleBefore = fAngle(fDistXbefore, fDistYbefore);


		for (int j = 0; j < vecRectTracking.size(); j++)
		{
			float fDistXCandi = m_ptVanishing.x - (vecRectTracking[j].x + vecRectTracking[j].width / 2);
			float fDistYCandi = m_ptVanishing.y - (vecRectTracking[j].y + vecRectTracking[j].height / 2);
			float fDistCandi = fDistXCandi*fDistXCandi + fDistYCandi*fDistYCandi;
			fDistCandi = sqrt(fDistCandi);
			float fAngleCandi = fAngle(fDistXCandi, fDistYCandi);

			//int dist =abs(vecRectTracking[j].x+(vecRectTracking[j].width)/2-MultiKF[i].ptEstimate.x)+abs(vecRectTracking[j].y+(vecRectTracking[j].height)/2-MultiKF[i].ptEstimate.y);
			float dist = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x)*abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x)
				+ abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y)*abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y);
			dist = sqrt(dist);
			float pCurrnetXY = pCurrent.x*pCurrent.x + pCurrent.y*pCurrent.y;
			pCurrnetXY = sqrt(pCurrnetXY);
			if (((dist < pCurrnetXY && dist < scaledist *Set.vecBefore[i].width) || (abs(fAngleBefore - fAngleCandi) < 2)) && vecRectTracking[j].area() >= Set.vecBefore[i].area()*0.8 && vecRectTracking[j].y + vecRectTracking[j].height / 2 - (Set.vecBefore[i].y + (Set.vecBefore[i].height) / 2) <= 1)
			{
				pCurrent.x = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x);
				pCurrent.y = abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y);
				num = j;
			}
		}

		if (num == -1 && MultiKF.size() > 0)
		{
			MultiKF[i].bOK = false;
			MultiKF[i].ptCenter.x = MultiKF[i].matPrediction.at<float>(0);
			MultiKF[i].ptCenter.y = MultiKF[i].matPrediction.at<float>(1);
			MultiKF[i].speedX = MultiKF[i].matPrediction.at<float>(2);
			MultiKF[i].speedY = MultiKF[i].matPrediction.at<float>(3);
			MultiKF[i].width = MultiKF[i].matPrediction.at<float>(4);
			MultiKF[i].height = MultiKF[i].matPrediction.at<float>(5);

			Set.vecBefore[i].x = MultiKF[i].ptPredict.x - (Set.vecBefore[i].width) / 2;
			Set.vecBefore[i].y = MultiKF[i].ptPredict.y - (Set.vecBefore[i].height) / 2;

		}
		else if (Set.vecBefore.size() != 0)
		{
			MultiKF[i].bOK = true;
			MultiKF[i].speedX = vecRectTracking[num].x + (vecRectTracking[num].width) / 2 - (Set.vecBefore[i].x + (Set.vecBefore[i].width) / 2);
			MultiKF[i].speedY = vecRectTracking[num].y + (vecRectTracking[num].height) / 2 - (Set.vecBefore[i].y + (Set.vecBefore[i].height) / 2);

			Set.vecBefore[i] = vecRectTracking[num];
			vecRectTracking.erase(vecRectTracking.begin() + num);
			MultiKF[i].ptCenter.x = Set.vecBefore[i].x + (Set.vecBefore[i].width) / 2;
			MultiKF[i].ptCenter.y = Set.vecBefore[i].y + (Set.vecBefore[i].height) / 2;
			MultiKF[i].width = Set.vecBefore[i].width;
			MultiKF[i].height = Set.vecBefore[i].height;
			Set.vecCount[i] = cntframe; //vecCount


		}
		//cout << "Set.vecBefore[i].y + Set.vecBefore[i].height : " << (Set.vecBefore[i].y + Set.vecBefore[i].height) << endl;
		//cout << "Set.vecBefore[i].x + Set.vecBefore[i].width : " << Set.vecBefore[i].x + Set.vecBefore[i].width << endl;
		//cout << "Set.vecBefore[i] : " << Set.vecBefore[i] << endl;
		//
		bool bcheck = false;

		if (Set.vecBefore[i].y < 0 || Set.vecBefore[i].x < 0 || Set.vecBefore[i].y + Set.vecBefore[i].height >= srcImage.rows || Set.vecBefore[i].x + Set.vecBefore[i].width >= srcImage.cols)
		{
			Set.vecBefore.erase(Set.vecBefore.begin() + i);
			MultiKF.erase(MultiKF.begin() + i);
			Set.vecCount.erase(Set.vecCount.begin() + i);
			i--;
			bcheck = true;
		}

		if (bcheck == false && (imgROImask.at<uchar>((Set.vecBefore[i].y < 0 ? 0 : Set.vecBefore[i].y), (Set.vecBefore[i].x < 0 ? 0 : Set.vecBefore[i].x)) == 0 || imgROImask.at<uchar>((Set.vecBefore[i].y + Set.vecBefore[i].height < srcImage.rows ? Set.vecBefore[i].y + Set.vecBefore[i].height : srcImage.rows - 1), (Set.vecBefore[i].x + Set.vecBefore[i].width < srcImage.cols ? Set.vecBefore[i].x + Set.vecBefore[i].width : srcImage.cols - 1)) == 0 || (cntframe - Set.vecCount[i])>cntBefore) && Set.vecBefore.size() != 0 && MultiKF.size() != 0){
			Set.vecBefore.erase(Set.vecBefore.begin() + i);
			MultiKF.erase(MultiKF.begin() + i);
			Set.vecCount.erase(Set.vecCount.begin() + i);
			i--;
		}
		//cout << "test" << endl;
	}


	for (int i = 0; i < Set.vecCandidate.size(); i++)
	{
		pCurrent.x = srcImage.cols;
		pCurrent.y = srcImage.rows;
		///////////
		int num = -1;
		for (int j = 0; j < vecRectTracking.size(); j++)
		{

			float dist = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2))*abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2))
				+ abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2))*abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2));
			dist = sqrt(dist);
			float pCurrnetXY = pCurrent.x*pCurrent.x + pCurrent.y*pCurrent.y;
			pCurrnetXY = sqrt(pCurrnetXY);
			if (dist < pCurrnetXY && dist < scaledist * 2 * Set.vecCandidate[i].width && vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2) <= 0)
			{

				float fDistXCandi = m_ptVanishing.x - (vecRectTracking[j].x + vecRectTracking[j].width / 2);
				float fDistYCandi = m_ptVanishing.y - (vecRectTracking[j].y + vecRectTracking[j].height / 2);
				float fDistCandi = fDistXCandi*fDistXCandi + fDistYCandi*fDistYCandi;
				fDistCandi = sqrt(fDistCandi);
				float fAngleCandi = fAngle(fDistXCandi, fDistYCandi);
				//cout << "angle : " << fAngleCandi << endl;
				if (Set.vecRtheta[i].x<fDistCandi && abs(Set.vecRtheta[i].y - fAngleCandi)<2)
				{
					pCurrent.x = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2));
					pCurrent.y = abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2));
					num = j;
				}
			}
		}
		int speedtempX = 0;
		int speedtempY = 0;
		if (num > -1 && vecRectTracking.size()>0)
		{
			Set.vecCountPush[i].x++;
			Set.vecCountPush[i].y = cntframe;
			speedtempX = vecRectTracking[num].x - Set.vecCandidate[i].x;
			speedtempY = vecRectTracking[num].y - Set.vecCandidate[i].y;
			Set.vecCandidate[i] = vecRectTracking[num];
			//////////
			float fDistXCandi = m_ptVanishing.x - (vecRectTracking[num].x + vecRectTracking[num].width / 2);
			float fDistYCandi = m_ptVanishing.y - (vecRectTracking[num].y + vecRectTracking[num].height / 2);
			float fDistCandi = fDistXCandi*fDistXCandi + fDistYCandi*fDistYCandi;
			fDistCandi = sqrt(fDistCandi);
			float fAngleCandi = fAngle(fDistXCandi, fDistYCandi);
			Set.vecRtheta[i] = Point2f(fDistCandi, fAngleCandi);
			//////
			//	cout << i<<" R thetha : " << Set.vecRtheta[i]<<endl;
			vecRectTracking.erase(vecRectTracking.begin() + num);
		}

		if (Set.vecCountPush[i].x >= cntCandidate)
		{
			////
			Set.vecBefore.push_back(Set.vecCandidate[i]);
			SKalman temKalman;
			temKalman.speedX = speedtempX;
			temKalman.speedY = speedtempY;
			kalmanTrackingStart(temKalman, Set.vecCandidate[i]);
			MultiKF.push_back(temKalman);
			Set.vecCount.push_back(cntframe);

			Set.vecCandidate.erase(Set.vecCandidate.begin() + i);
			Set.vecCountPush.erase(Set.vecCountPush.begin() + i);
			Set.vecRtheta.erase(Set.vecRtheta.begin() + i);
			i--;
		}
	}
	for (int i = 0; i < vecRectTracking.size(); i++)
	{
		//Set.vecRtheta.push_back(

		float fDistXCandi = m_ptVanishing.x - (vecRectTracking[i].x + vecRectTracking[i].width / 2);
		float fDistYCandi = m_ptVanishing.y - (vecRectTracking[i].y + vecRectTracking[i].height / 2);
		float fDistCandi = fDistXCandi*fDistXCandi + fDistYCandi*fDistYCandi;
		fDistCandi = sqrt(fDistCandi);

		float fAngleCandi = fAngle(fDistXCandi, fDistYCandi);
		Set.vecRtheta.push_back(Point2f(fDistCandi, fAngleCandi));
		Set.vecCandidate.push_back(vecRectTracking[i]);
		Set.vecCountPush.push_back(Point(0, cntframe));
	}
	for (int i = 0; i < Set.vecCountPush.size(); i++)
	{
		if (cntframe - Set.vecCountPush[i].y >= frameCandiate)
		{
			Set.vecCandidate.erase(Set.vecCandidate.begin() + i);
			Set.vecCountPush.erase(Set.vecCountPush.begin() + i);
			Set.vecRtheta.erase(Set.vecRtheta.begin() + i);
			i--;
		}
	}

	time6 = (double)getTickCount() - time6;
	printf("tracking part : %f ms.\n", time6*1000. / getTickFrequency());
	////////////////////////////////
	double time7 = (double)getTickCount();


	///// Kalman filtering  //////////////
	for (int i = 0; i < Set.vecBefore.size(); i++)
	{
		//	cout << "vecBefore size : "<<Set.vecBefore.size();
		bool bfinish = kalmanfilter(srcImage, MultiKF[i], Set.vecBefore[i], ROIset, vecValidRec, imgROImask);
		if (bfinish == false)
		{
			Set.vecBefore.erase(Set.vecBefore.begin() + i);
			MultiKF.erase(MultiKF.begin() + i);
			Set.vecCount.erase(Set.vecCount.begin() + i);
			i--;
			bfinish = true;
		}
	}

	time7 = (double)getTickCount() - time7;
	//printf("kalman filter : %f ms.\n", time7*1000. / getTickFrequency());
}

void CTST::AdaptiveROI(Mat&imgSrc)
{

	m_imgROImask_T = Mat::zeros(m_imgROImask.size(), CV_8UC1);

	bool bChoice = false;
	for (int i = 0; i < m_multiTracker.vecBefore.size(); i++)
	{
		if (m_multiTracker.vecBefore[i].y + m_multiTracker.vecBefore[i].height < m_roi_t_u.y + m_roi_t_u.height)
			bChoice = true;

	}
	Rect roi_d_u;
	if (m_multiTracker.vecBefore.size() == 0 || bChoice == false)
		roi_d_u = Rect_<int>(m_imgROImask.cols / 3 - m_imgROImask.cols / 8, m_imgROImask.rows / 4 - m_imgROImask.rows / 8 + m_imgROImask.rows / 32, m_imgROImask.cols * 5 / 8, m_imgROImask.rows / 4 - m_imgROImask.rows / 8 + m_imgROImask.rows / 32);

	for (int i = 0; i < m_multiTracker.vecBefore.size(); i++)
	{
		int roix = (m_MultiKF[i].matPrediction.at<float>(0) - 150 > 0) ? m_MultiKF[i].matPrediction.at<float>(0) - 150 : 0;
		int roiy = (m_MultiKF[i].matPrediction.at<float>(1) - 60 > 0) ? m_MultiKF[i].matPrediction.at<float>(1) - 60 : 0;
		int roiw = (roix + m_MultiKF[i].matPrediction.at<float>(4) + 250 < m_imgROImask.cols) ? m_MultiKF[i].matPrediction.at<float>(4) + 250 : m_imgROImask.cols - 1 - roix;
		int roih = (roiy + m_MultiKF[i].matPrediction.at<float>(5) + 60 < m_imgROImask.rows) ? m_MultiKF[i].matPrediction.at<float>(5) + 60 : m_imgROImask.rows - 1 - roiy;

		Rect ROIset_adapD = Rect_<int>(roix, roiy, roiw, roih);
		m_imgROImask(ROIset_adapD).setTo(Scalar::all(255));

	}


	m_imgROImask(roi_d_u).setTo(Scalar::all(255));
	m_imgROImask(m_roi_d_r).setTo(Scalar::all(255));
	m_imgROImask_T(m_roi_t_u).setTo(Scalar::all(255));
	m_imgROImask_T(m_roi_t_r).setTo(Scalar::all(255));

	if (m_multiTracker.vecBefore.size() == 0){
		m_adaptiveDetection.SetROI(roi_d_u);
		m_adaptiveDetection.DetectSigns(m_imgSrc, m_MserVec, m_vecRectTracking);
	}
	m_adaptiveDetection.SetROI(m_roi_d_r);
	m_adaptiveDetection.DetectSigns(m_imgSrc, m_MserVec, m_vecRectTracking);


	if (m_multiTracker.vecBefore.size() != 0 && bChoice == true)
	{
		int dMaxX = 0;
		int dMaxY = 0;
		int dMinX = m_imgSrc.cols;
		int dMinY = m_imgSrc.rows;
		for (int i = 0; i < m_multiTracker.vecBefore.size(); i++)
		{
			if (m_MultiKF[i].matPrediction.at<float>(0) < dMinX)
				dMinX = m_MultiKF[i].matPrediction.at<float>(0);
			if (m_MultiKF[i].matPrediction.at<float>(1) < dMinY)
				dMinY = m_MultiKF[i].matPrediction.at<float>(1);
			if (m_MultiKF[i].matPrediction.at<float>(0) + m_MultiKF[i].matPrediction.at<float>(4) > dMaxX)
				dMaxX = m_MultiKF[i].matPrediction.at<float>(0) + m_MultiKF[i].matPrediction.at<float>(4);
			if (m_MultiKF[i].matPrediction.at<float>(1) + m_MultiKF[i].matPrediction.at<float>(5) > dMaxY)
				dMaxY = m_MultiKF[i].matPrediction.at<float>(1) + m_MultiKF[i].matPrediction.at<float>(5);
		}

		int roix = (dMinX - 150 > 0) ? dMinX - 150 : 0;
		int roiy = (dMinY - 60 > 0) ? dMinY - 60 : 0;
		int roiw = (dMaxX + 150 < m_imgSrc.cols) ? dMaxX - roix + 150 : m_imgSrc.cols - 1 - roix;
		int roih = (dMaxY + 60 < m_imgSrc.rows) ? dMaxY - roiy + 10 : m_imgSrc.rows - 1 - roiy;

		Rect ROIset_Track = Rect_<int>(roix, roiy, roiw, roih);

		m_imgROImask(ROIset_Track).setTo(Scalar::all(255));
		m_adaptiveDetection.SetROI(ROIset_Track);
		m_adaptiveDetection.DetectSigns(m_imgSrc, m_MserVec, m_vecRectTracking);


	}
		vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	findContours(m_imgROImask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 255, 255);
		drawContours(imgSrc, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	findContours(m_imgROImask_T, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255, 0, 255);
		drawContours(imgSrc, contours, i, color, 2, 8, hierarchy, 0, Point());
	}



}

void CTST::SetImage(Mat& imgSrc)
{
	m_imgSrc = imgSrc;
}

void CTST::SetROI(Rect& droi_u, Rect& droi_r, Rect& troi_u, Rect& troi_r, Mat& roimask)
{
	m_roi_d_u = droi_u;
	m_roi_d_r = droi_r;
	m_roi_t_u = troi_u;
	m_roi_t_r = troi_r;
	m_imgROImask = roimask.clone();

}

void CTST::kalmanTrackingStart(SKalman& temKalman, Rect& recStart)
{
	temKalman.KF.statePost.at<float>(0) = recStart.x + (recStart.width) / 2;
	temKalman.KF.statePost.at<float>(1) = recStart.y + (recStart.height) / 2;
	temKalman.KF.statePost.at<float>(2) = temKalman.speedX;
	temKalman.KF.statePost.at<float>(3) = temKalman.speedY;
	temKalman.KF.statePost.at<float>(4) = recStart.width;
	temKalman.KF.statePost.at<float>(5) = recStart.height;

	temKalman.KF.statePre.at<float>(0) = recStart.x + (recStart.width) / 2;
	temKalman.KF.statePre.at<float>(1) = recStart.y + (recStart.height) / 2;
	temKalman.KF.statePre.at<float>(2) = temKalman.speedX;
	temKalman.KF.statePre.at<float>(3) = temKalman.speedY;
	temKalman.KF.statePre.at<float>(4) = recStart.width;
	temKalman.KF.statePre.at<float>(5) = recStart.height;

	kalmansetting(temKalman.KF, temKalman.smeasurement);

	temKalman.smeasurement.at<float>(0) = recStart.x + (recStart.width) / 2;
	temKalman.smeasurement.at<float>(1) = recStart.y + (recStart.height) / 2;
	temKalman.width = recStart.width;
	temKalman.height = recStart.height;
	temKalman.ptCenter.x = recStart.x + (recStart.width) / 2;
	temKalman.ptCenter.y = recStart.y + (recStart.height) / 2;
	temKalman.ptEstimate.x = recStart.x + (recStart.width) / 2;
	temKalman.ptEstimate.y = recStart.y + (recStart.height) / 2;
	temKalman.ptPredict.x = temKalman.ptEstimate.x + temKalman.speedX;
	temKalman.ptPredict.y = temKalman.ptEstimate.y + temKalman.speedY;
	temKalman.rgb = Scalar(CV_RGB(rand() % 255, rand() % 255, rand() % 255));
}
