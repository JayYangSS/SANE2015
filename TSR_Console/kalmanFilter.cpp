#include "kalmanFilter.h"

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


	//	cout << "MultiKF.smeasurement : " << MultiKF.smeasurement << endl;
	//// generate measurement
	//     MultiKF.smeasurement += MultiKF.KF.measurementMatrix*MultiKF.state;

	//	cout << "speed : "<<MultiKF.smeasurement(2)<<", "<<MultiKF.smeasurement(3)<<endl;
	Point measPt(MultiKF.smeasurement(0), MultiKF.smeasurement(1));
	///////////


	Mat estimated = MultiKF.KF.correct(MultiKF.smeasurement);  //measuremnet를 predicted state로 update --------- P구하는듯..
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
		//		drawCross(img, statePt, MultiKF.rgb, 5);   // tracking 결과
		//		cv::rectangle(img, recKalman, MultiKF.rgb, 2); //tracking 결과
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
		//		cout << "prediction : " << prediction << endl;
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
