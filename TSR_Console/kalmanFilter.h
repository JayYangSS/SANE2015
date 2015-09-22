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
//        - 예측값, 수정 이전 상태값, 초기화
//    Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
//        - 수정값, correct() 함수에 의해 계산됨
//
//    Mat transitionMatrix;   //!< state transition matrix (A)
//        - 변환 행렬, 변환 메커니즘 적용, 중요
//    Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
//        - 제어 행렬, 초기화 하지 않음
//    Mat measurementMatrix;  //!< measurement matrix (H)
//        - 측정 행렬, correct() 함수에 측정값 입력시 자동 변화
//
//    Mat processNoiseCov;    //!< process noise covariance matrix (Q)
//        - 프로세스 잡음 공분산, 클수록 수정값이 많이 변함, 1e-4
//    Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
//        - 측정 잡음 공분산, 작을수록 수정값이 많이 변함, 1e-1
//        - 100ms 정도의 지연 시간에서 둘다 1e-3 정도가 적당한 듯
//
//    Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
//        - 초기화 안함, 이전 오차 공분산 
//    Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
//        - 초기화 안함
//    Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
//        - 이후 오차 공분산, 초기화 안함
//
//predict();//예측 : 다음 시간단계에 대한 예측, 결과는 statePre에 저장
//correct(Mat measurement);//교정 : 새로운 측정치와 통합, 결과는 statePost에 저장
//
//
//
//
//=============================================================================================================
