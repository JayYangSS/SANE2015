#include "PedestrianDetection.h"
//#include "KalmanTracker.h"

#define Debug_Kalman 1

#define Debug_OF_Tracking 0

#define __K_Next_Version 0
#define __K_Debug_1 0

#if Debug_OF_Tracking

float Median_Vector(vector<float> v)
{

	int n = floor(v.size() / 2.0);

	nth_element(v.begin(), v.begin() + n, v.end());
	return v[n];
}

/**
@brief Median Flow 방식의 Tracking입니다 \n
OpenCV의 Parse방식으로 영상 전체에서 특정 위치를 입력으로 받고 이미지를 피라미드화 하여 \n
NCC(Normalized Cross Correlation)의 최대치인 부분이 결과로 나오게 됩니다. \n
NCC의 Median 및 FB(Foward-Backward) Error의 Median Scale\n
3가지 정보를 가지고 Median을 이용하여 아웃라이어 제거 및 Tracking의 정확도를 향상에 기여합니다. \n
@param CurrentImage : 현재 전체 입력 이미지
@param InputROI : 트래킹하고자 하는 ROI
@param NewTrackingROI : 트래킹 결과 ROI
@return void
@remark 인식과 관련하여 Detection ROI와 Tracking ROI관련 \n
우선순위를 어디에 두어야 하는 상관관계에 관한것은 아직 포함되어 있지 않습니다.\n
@author 강근호(keunhokang@hanyang.co.kr)
@date LastUpdate 2015-08-21
*/
int cnt = 0;
bool bTrackingReady = false;
void OF_Tracking(Mat &Current_Image, cv::Rect &InputROI, cv::Rect &NewTrackingROI)
{
	static Mat DetectionPreImage;
	static Rect_<int> TrackingROI;

	if (!bTrackingReady)
	{
		DetectionPreImage = Current_Image.clone();
		TrackingROI = InputROI;
#if 0 // __K_Debug_1
		TrackingROI.x = 1170;
		TrackingROI.y = 540;
		TrackingROI.width = 100;
		TrackingROI.height = 50;
#endif
		bTrackingReady = true;
	}
	else
	{
		vector<Point2f> Grid_S;
		vector<Point2f> Grid_E;
		vector<Point2f> FB_Point;
		vector<unsigned char> status;
		vector<float> NCC;
		vector<unsigned char> FB_status;
		vector<float> FB_err;
		cv::Size Window_Size = Size(4, 4);
		int Pyramid_Lvl = 5;
		int GridPointStep = 5;
		bool Tracking_Flg = true;

#if __K_Next_Version
		//5 Point 단위 그리드 포인트
		for (int y = 0; y < TrackingROI.height; y += GridPointStep){
			for (int x = 0; x < TrackingROI.width; x += GridPointStep){
				Grid_S.push_back(Point2f(x + TrackingROI.x, y + TrackingROI.y));
				Grid_E.push_back(Point2f(x, y));
			}
		}
#endif

		Grid_S.push_back(Point2f(TrackingROI.x, TrackingROI.y));
		Grid_E.push_back(Point2f(TrackingROI.x, TrackingROI.y));
		int ROIWidth = TrackingROI.width;
		int ROIHeight = TrackingROI.height;
#if __K_Debug_0
		circle(DetectionPreImage, Grid_S[0], 5, Scalar(0, 255, 0), 2, 8, 0);
		imshow("DetectionPreImage", DetectionPreImage);
		waitKey(1);
#endif
		cv::calcOpticalFlowPyrLK(DetectionPreImage, Current_Image, Grid_S, Grid_E, status, NCC, Window_Size, Pyramid_Lvl, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.05), 0);
#if __K_Next_Version
		cv::calcOpticalFlowPyrLK(Current_Image, DetectionPreImage, Grid_E, FB_Point, FB_status, FB_err, Window_Size, Pyramid_Lvl, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.05), 0);
		for (int k = 0; k < FB_Point.size(); k++)
		{
			FB_err[k] = norm(FB_Point[k] - Grid_S[k]);
		}
		//Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
		cv::Mat rec0(10, 10, CV_8U);
		cv::Mat rec1(10, 10, CV_8U);
		cv::Mat res(1, 1, CV_32F);
		for (int k = 0; k < Grid_S.size(); k++){
			if (status[k])
			{
				getRectSubPix(Current_Image, Size(10, 10), Grid_S[k], rec0);
				getRectSubPix(DetectionPreImage, Size(10, 10), Grid_E[k], rec1);
				matchTemplate(rec0, rec1, res, CV_TM_CCOEFF_NORMED);
				NCC[k] = ((float*)(res.data))[0];
			}
			else
			{
				NCC[k] = 0.0;
			}
		}
#if __K_Debug_1
		//imshow("rec0",rec0);
		//imshow("rec1",rec1);
		//waitKey(0);
#endif

		//Get Medians
		float NCC_Median = Median_Vector(NCC);
		float FB_Median = Median_Vector(FB_err);
		if (FB_Median)
		{
			Tracking_Flg = false;
		}

		//Outlier : NCC && FBB
		int Tr_Pt_Cnt = 0;

		for (int k = 0; k < Grid_E.size(); k++)
		{
			if (status[k])
			{
				if (NCC[k] > NCC_Median || FB_err[k] <= FB_Median)
				{
					Grid_S[Tr_Pt_Cnt] = Grid_S[k];
					Grid_E[Tr_Pt_Cnt] = Grid_E[k];
					FB_err[Tr_Pt_Cnt] = FB_err[k];
					Tr_Pt_Cnt++;
				}
			}
		}
		if (Tr_Pt_Cnt == 0)
		{
			Tracking_Flg = false;
			cnt++;
		}
		Grid_S.resize(Tr_Pt_Cnt);
		Grid_E.resize(Tr_Pt_Cnt);
		FB_err.resize(Tr_Pt_Cnt);

		//////////////////////////////////////////////////////////////////////////
		//ROI Renewal
		vector<float> Diff_X(Tr_Pt_Cnt);
		vector<float> Diff_Y(Tr_Pt_Cnt);
		for (int k = 0; k < Grid_S.size(); k++)
		{
			Diff_X[k] = Grid_E[k].x - Grid_S[k].x;
			Diff_Y[k] = Grid_E[k].y - Grid_S[k].y;
		}
		float DX = Median_Vector(Diff_X);
		float DY = Median_Vector(Diff_Y);

		//Scale Chanege Ratio
		float s;
		if (Tr_Pt_Cnt > 1)
		{
			vector<float> d;
			d.reserve(Tr_Pt_Cnt * (Tr_Pt_Cnt - 1) / 2);
			for (int k = 0; k < Tr_Pt_Cnt; k++)
			{
				for (int l = k + 1; l < Tr_Pt_Cnt; l++)
				{
					//float Scale_Ratio = norm(Grid_E[k]-Grid_E[l])/norm(Grid_S[k]-Grid_S[l]);
					d.push_back(norm(Grid_E[k] - Grid_E[l]) / norm(Grid_S[k] - Grid_S[l]));
				}
			}
			s = Median_Vector(d);
		}
		else
		{
			s = 1.0;
		}
		//Renewal Tracking ROI
		float sW = 0.5*(s - 1)*TrackingROI.width;
		float sH = 0.5*(s - 1)*TrackingROI.height;
		float temp = TrackingROI.x + DX - sW;
		//Round Process
		NewTrackingROI.x = ((int(temp * 10)) % 10 < 5) ? floor(temp) : ceil(temp);
		temp = TrackingROI.y + DY - sH;
		NewTrackingROI.y = ((int(temp * 10)) % 10 < 5) ? floor(temp) : ceil(temp);
		temp = TrackingROI.width * s;
		NewTrackingROI.width = ((int(temp * 10)) % 10 < 5) ? floor(temp) : ceil(temp);
		temp = TrackingROI.height * s;
		NewTrackingROI.height = ((int(temp * 10)) % 10 < 5) ? floor(temp) : ceil(temp);

#endif

#if __K_Debug_1
		Mat ch3_Motion;
		ch3_Motion = Current_Image.clone();
		for (int p = 0; p < Tr_Pt_Cnt; p++)
		{
			line(ch3_Motion, Point((int)(Grid_S[p].x), (int)(Grid_S[p].y)), Point((int)(Grid_E[p].x), (int)(Grid_E[p].y)), Scalar(0, 255, 0), 1, 1, 0);
			circle(ch3_Motion, Point((int)(Grid_S[p].x), (int)(Grid_S[p].y)), 1, Scalar(0, 0, 255), 1, 1, 0);
		}
		cout << endl;
		cout << "	Ken : Start Position : " << Grid_S[0].x << " , " << Grid_S[0].y << endl;
		cout << "	Ken : End Position : " << Grid_E[0].x << " , " << Grid_E[0].y << endl;
		cout << "	Ken : Derivation : " << DX << " , " << DY << endl;
		cout << "	Ken : Cnt : " << cnt << endl;
		cout << endl;
		imshow("Motion", ch3_Motion);
		waitKey(1);
		DetectionPreImage = Current_Image.clone();
		TrackingROI = NewTrackingROI;
#endif
	}
	return;
}

#endif

int main()
{
	VideoCapture vcap("../CVLAB_dataset/data/2015-03-18-09h-59m-40s_F_normal.mp4"); // y=0.7513x+337.75
	//VideoCapture vcap("../CVLAB_dataset/data/2015-02-24-17h-52m-23s_F_event.mp4"); // y=0.9037x+359.41
	//VideoCapture vcap("../CVLAB_dataset/data/2015-04-29-14h-56m-42s_F_normal.mp4"); // y=0.6729x+319.02
	if (!vcap.isOpened())
		return -1;

	Rect rectROI = Rect(1280 / 4, 720 / 4, 1280 / 2, 720 / 2);
	//Rect rectROI = Rect(0, 0, 1280, 720);

	CPedestrianDetection objPD;
	objPD.LoadClassifier("acf_classifier.txt");

	Mat imgInput;
	Mat imgDisp;
	Mat imgDisp2;
	int nDelayms = 0;
	int cntFrames = 0;

#if Debug_Kalman

	float dt = 0.15;
	const int nDimState = 9;
	const int nDimMeas = 3;
	KalmanFilter kf;
	kf.init(nDimState, nDimMeas);
	kf.transitionMatrix = Mat::eye(nDimState, nDimState, CV_32FC1); // A
	kf.transitionMatrix.at<float>(0, 3) = 1;
	kf.transitionMatrix.at<float>(1, 4) = 1;
	kf.transitionMatrix.at<float>(2, 5) = 1;
	kf.transitionMatrix.at<float>(3, 6) = 1;
	kf.transitionMatrix.at<float>(4, 7) = 1;
	kf.transitionMatrix.at<float>(5, 8) = 1;
	
	kf.processNoiseCov = Mat::eye(nDimState, nDimState, CV_32FC1); // Q
	kf.processNoiseCov.at<float>(0, 0) = 0.1;
	kf.processNoiseCov.at<float>(1, 1) = 0.1;
	kf.processNoiseCov.at<float>(2, 2) = 0.1;
	kf.processNoiseCov.at<float>(3, 3) = 1;
	kf.processNoiseCov.at<float>(4, 4) = 1;
	kf.processNoiseCov.at<float>(5, 5) = 1;
	kf.processNoiseCov.at<float>(6, 6) = 1;
	kf.processNoiseCov.at<float>(7, 7) = 1;
	kf.processNoiseCov.at<float>(8, 8) = 1;

	kf.measurementMatrix = Mat::eye(nDimMeas, nDimState, CV_32FC1); // H
	kf.measurementNoiseCov = Mat::eye(nDimMeas, nDimMeas, CV_32FC1); // R
	kf.measurementNoiseCov.at<float>(0, 0) = 10;
	kf.measurementNoiseCov.at<float>(1, 1) = 10;
	kf.measurementNoiseCov.at<float>(2, 2) = 10;

	kf.errorCovPost = Mat::eye(nDimState, nDimState, CV_32FC1) * 1;

#endif

	while (1)
	{
		vcap >> imgInput;
		if (imgInput.empty()) break;

		Mat imgInputResz;
		imgInputResz = imgInput(rectROI).clone();
		imgDisp = imgInputResz.clone();
		imgDisp2 = imgInputResz.clone();

		double t = (double)getTickCount();
		
		objPD.Detect(imgInputResz);
		
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("%04d  %.2lf ms  %.1lf fps\n", cntFrames, t * 1000, 1 / t);

		vector<Rect_<int> > bb = objPD.m_vecrectDetectedBB;
		for (int i = 0; i < (int)bb.size(); i++)
		{
			bb[i].x += rectROI.x;
			bb[i].y += rectROI.y;
		}

		for (int i = 0; i < (int)bb.size(); i++)
		{
			double est = (double)bb[i].height*0.7513 + 337.75;
			if (abs((double)bb[i].br().y - est) > 10)
			{
				objPD.m_vecrectDetectedBB.erase(objPD.m_vecrectDetectedBB.begin() + i);
				bb.erase(bb.begin() + i);
				i--;
			}
		}

		objPD.DrawBoundingBox(imgDisp, CV_RGB(0, 0, 0));
		
#if Debug_Kalman
		Rect_<int> rectKF;
		
		if (cntFrames >= 70)
		{
			if (cntFrames == 70)
			{
				float* idxPost = (float*)kf.statePost.data;
				idxPost[0] = (float)objPD.m_vecrectDetectedBB[0].x;
				idxPost[1] = (float)objPD.m_vecrectDetectedBB[0].y;
				idxPost[2] = (float)objPD.m_vecrectDetectedBB[0].width;

				rectKF = objPD.m_vecrectDetectedBB[0];

				nDelayms = 0;
			}
			else
			{
				kf.predict();

				float* idxPre = (float*)kf.statePost.data;
				rectKF.x = (int)idxPre[0];
				rectKF.y = (int)idxPre[1];
				rectKF.width = (int)idxPre[2];
				rectKF.height = (int)(idxPre[2] * 2.5f);

				Mat measurement = Mat::zeros(3, 1, CV_32FC1);
				float* idxMeas = (float*)measurement.data;
				if (objPD.m_vecrectDetectedBB.size() > 0)
				{
					idxMeas[0] = (float)objPD.m_vecrectDetectedBB[0].x;
					idxMeas[1] = (float)objPD.m_vecrectDetectedBB[0].y;
					idxMeas[2] = (float)objPD.m_vecrectDetectedBB[0].width;
					kf.correct(measurement);
				}
			}

			rectangle(imgDisp2, rectKF, CV_RGB(0, 255, 255), 2);

			//cout << kf.errorCovPost << endl;
			cout << kf.statePost << endl;

		}
#endif

#if Debug_OF_Tracking
		if (cntFrames >= 70 && cntFrames <= 123)
		{
			nDelayms = 0;
			Rect_<int> bbd = (Rect_<int>)objPD.m_vecrectDetectedBB[0];
			Rect_<int> bbt;
			if (objPD.m_vecrectDetectedBB.size() > 0)
			{
				OF_Tracking(imgInputResz, bbd, bbt);
			}

			rectangle(imgDisp, bbt, CV_RGB(0, 255, 255), 2);

		}
#endif

		imshow("Display", imgDisp);
		imshow("Track", imgDisp2);
		cntFrames++;

		int nKey = waitKey(nDelayms);
		if (nKey == 27) break;
		else if (nKey == 32) nDelayms = 1 - nDelayms;
		else if (nKey == 'f') { nDelayms = 0; }
	}

	return 0;
}