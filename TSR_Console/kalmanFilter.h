#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <Windows.h>
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
void drawCross(Mat img, Point center, Scalar color, int d)
{
	line(img, Point(center.x - d, center.y - d), Point(center.x + d, center.y + d), color, 2, CV_AA, 0);
	line(img, Point(center.x + d, center.y - d), Point(center.x - d, center.y + d), color, 2, CV_AA, 0);
}

void kalmansetting(KalmanFilter& KF, Mat_<float>& measurement)
{
	// ------------------------------------------------------------ Kalman Filter setup

	//KalmanFilter KF(4, 2, 0);
	//Mat_<float> state(4, 1); /* (x, y, Vx, Vy) */
	//Mat processNoise(4, 1, CV_32F);
	//Mat_<float> measurement(2,1); 
	//int failcase=0;
	//int succcase=0;

	//int es1=0;
	//int es2=0;
	///////////////////////////////////////////////////////////////
	//// -------------------------------------------------------- Initialise Kalman parameters 

	//KF.statePre.at<float>(0) = 0;
	//KF.statePre.at<float>(1) = 0;
	//KF.statePre.at<float>(2) = 0;
	float dt = 1.2f;
	//KF.statePre.at<float>(3) = 0;
	KF.transitionMatrix = *(Mat_<float>(8, 8) <<
		1.0f, 0.0f, dt, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, dt, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);  //A
	//KF.measurementMatrix = *(Mat_<float>(2, 4) << 1,0,0,0,   0,1,0,0);

	setIdentity(KF.measurementMatrix);


	setIdentity(KF.processNoiseCov, Scalar::all(1e-3));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-3));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	/*measurement.setTo(Scalar(0));*/
	//	cout << KF.measurementMatrix<<endl;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


}

int kalmanfilter(Mat img, SKalman& MultiKF, Rect& rec, Rect& ROIset, vector<Rect>& vecValidRec, Mat& imgROImask)
{



	if (MultiKF.ptCenter.x == 0 && MultiKF.ptCenter.y == 0)
	{
		MultiKF.ptCenter.x = MultiKF.ptEstimate.x;
		MultiKF.ptCenter.y = MultiKF.ptEstimate.y;
	}


	//	cout << "prediction : "<<prediction.at<float>(1)<<endl;
	MultiKF.smeasurement(0) = MultiKF.ptCenter.x;
	MultiKF.smeasurement(1) = MultiKF.ptCenter.y;
	MultiKF.smeasurement(2) = MultiKF.speedX;
	MultiKF.smeasurement(3) = MultiKF.speedY;
	MultiKF.smeasurement(4) = rec.width;
	MultiKF.smeasurement(5) = rec.height;


	cout << "MultiKF.smeasurement : " << MultiKF.smeasurement << endl;
	//// generate measurement
	//     MultiKF.smeasurement += MultiKF.KF.measurementMatrix*MultiKF.state;

	//	cout << "speed : "<<MultiKF.smeasurement(2)<<", "<<MultiKF.smeasurement(3)<<endl;
	Point measPt(MultiKF.smeasurement(0), MultiKF.smeasurement(1));
	///////////


	Mat estimated = MultiKF.KF.correct(MultiKF.smeasurement);  //measuremnet�� predicted state�� update --------- P���ϴµ�..
	//!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
	Point statePt(estimated.at<float>(0), estimated.at<float>(1));//!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))---compute the estimate

	//Point statePt(0,0);
	if (imgROImask.at<uchar>((statePt.y - (rec.height / 2)<0 ? 0 : statePt.y - (rec.height / 2)), (statePt.x - (rec.width / 2) < 0 ? 0 : statePt.x - (rec.width / 2))) == 0)
		return false;
	else
	{
		Rect recKalman = Rect(Point2d(statePt.x - (rec.width / 2), statePt.y - (rec.height / 2)), Point2d(statePt.x + (rec.width / 2), statePt.y + (rec.height / 2)));
		vecValidRec.push_back(recKalman);


		////////////
//		drawCross(img, statePt, MultiKF.rgb, 5);   // tracking ���
//		cv::rectangle(img, recKalman, MultiKF.rgb, 2); //tracking ���
		//line(video_cap_, Point2d(statePt.x,0), Point2d(statePt.x,240), CV_RGB(255,0,0),3);

		/*line(img, Point2d(statePt.x-1,statePt.y-1), Point2d(statePt.x,statePt.y), MultiKF.rgb,3);
		line(img, Point2d(measPt.x-1,measPt.y-1), Point2d(measPt.x,measPt.y), MultiKF.rgb,3);*/

		//////////

		//	cout << "statePt : " <<statePt.x<<" "<<statePt.y <<endl;
		//	cout << "measPt : " <<measPt.x<<" "<<measPt.y <<endl;
		//resize(video_cap,video_cap,Size2i(480,360));
		//resize(video_cap_,video_cap_,Size2i(480,360));
		//	resize(imageMatches,imageMatches,Size2i(960,360));

		Mat prediction = MultiKF.KF.predict();
		cout << "prediction : " << prediction << endl;
		//putText(img,timeInfo,Point2i(10,30),FONT_HERSHEY_SIMPLEX,0.7,CV_RGB(0,255,0),2);
		MultiKF.ptPredict = Point(prediction.at<float>(0), prediction.at<float>(1));
		//MultiKF.speedX = prediction.at<float>(2);
		//MultiKF.speedY = prediction.at<float>(3);
		//MultiKF.width = prediction.at<float>(4);
		//MultiKF.height = prediction.at<float>(5);
		MultiKF.matPrediction = prediction;

		MultiKF.ptEstimate.x = estimated.at<float>(0);
		MultiKF.ptEstimate.y = estimated.at<float>(1);
		return true;
	}


}





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
