#ifndef Kalman
#define Kalman

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
//#include <Windows.h>
using namespace std;
using namespace cv;

struct SKalman{


	KalmanFilter KF;

	Mat_<float> state; /* (x, y, Vx, Vy) */
	Mat processNoise;
	Mat_<float> smeasurement;
	Scalar rgb;
	int failcase;
	int succcase;
	Point ptEstimate;
	//int ptEstimate.x;
	//int ptEstimate.y;
	Mat matPrediction;
	Point ptCenter;
	//int ptCenter.x;
	//int ptCenter.y;
	bool bOK;

	int width;
	int height;
	int speedX;
	int speedY;
	Point ptPredict;
	SKalman()
	{
		KalmanFilter temKF(8, 6, 0);
		Mat_<float> temState(8, 1);
		Mat temProcessNoise(8, 1, CV_32F);
		Mat_<float> temMeasurement(6, 1);
		temKF.statePost.at<float>(2) = 0;
		temKF.statePost.at<float>(3) = 0;
		temKF.statePost.at<float>(4) = 0;
		temKF.statePost.at<float>(5) = 0;
		temKF.statePost.at<float>(6) = 0;
		temKF.statePost.at<float>(7) = 0;
		temKF.statePre.at<float>(2) = 0;
		temKF.statePre.at<float>(3) = 0;
		temKF.statePre.at<float>(4) = 0;
		temKF.statePre.at<float>(5) = 0;
		temKF.statePost.at<float>(6) = 0;
		temKF.statePost.at<float>(7) = 0;
		//kalmansetting(temKF,temMeasurement);
		temMeasurement.at<float>(2) = 0;
		temMeasurement.at<float>(3) = 0;
		KF = temKF;
		state = temState;
		processNoise = temProcessNoise;
		smeasurement = temMeasurement;

		speedX = 0;
		speedY = -3;
	}

};
void drawCross(Mat img, Point center, Scalar color, int d);
void kalmansetting(KalmanFilter& KF, Mat_<float>& measurement);
int kalmanfilter(Mat img, SKalman& MultiKF, Rect& rec, Rect& ROIset, vector<Rect>& vecValidRec, Mat& imgROImask);

#endif



//===============================================================================================================
//KalmanFilter
//
//    Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
//        - ������, ���� ���� ���°�, �ʱ�ȭ
//    Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
//        - ������, correct() �Լ��� ���� ����
//
//    Mat transitionMatrix;   //!< state transition matrix (A)
//        - ��ȯ ���, ��ȯ ��Ŀ���� ����, �߿�
//    Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
//        - ���� ���, �ʱ�ȭ ���� ����
//    Mat measurementMatrix;  //!< measurement matrix (H)
//        - ���� ���, correct() �Լ��� ������ �Է½� �ڵ� ��ȭ
//
//    Mat processNoiseCov;    //!< process noise covariance matrix (Q)
//        - ���μ��� ���� ���л�, Ŭ���� �������� ���� ����, 1e-4
//    Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
//        - ���� ���� ���л�, �������� �������� ���� ����, 1e-1
//        - 100ms ������ ���� �ð����� �Ѵ� 1e-3 ������ ������ ��
//
//    Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
//        - �ʱ�ȭ ����, ���� ���� ���л� 
//    Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
//        - �ʱ�ȭ ����
//    Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
//        - ���� ���� ���л�, �ʱ�ȭ ����
//
//predict();//���� : ���� �ð��ܰ迡 ���� ����, ����� statePre�� ����
//correct(Mat measurement);//���� : ���ο� ����ġ�� ����, ����� statePost�� ����
//
//
//
//
//=============================================================================================================
