#include <opencv2/opencv.hpp>
#include <iostream>
//#include <math.h>
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
		KalmanFilter temKF(8, 6, 0);	//상태8, 측정6
		Mat_<float> temState(8, 1);	//상태값(??)
		Mat temProcessNoise(8, 1, CV_32F);	//노이즈 상태값4
		Mat_<float> temMeasurement(6, 1);	//측정6
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
	float dt = 0.1f;
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

	//transitionMatirx : 전이행렬

	//단위행렬로 초기화

	//측정행렬
	setIdentity(KF.measurementMatrix);
	//프로세스 잡음 공분산, 클수록 수정값이 많이 변함
	setIdentity(KF.processNoiseCov, Scalar::all(1e-3)); //1e-4, 따라오는 속도, 클수록 빠름
	//측정 잡음 공분산, 작을수록 수정값이 많이 변함
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-3));	//따라오는 속도, 작을수록 빠름
	//사후 에러 공분산
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	/*measurement.setTo(Scalar(0));*/
	//	cout << KF.measurementMatrix<<endl;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


}

//
//void clustering(vector<Rect>& vecCluster, int distance)
//{
//
//	for (int i = 0; i < vecCluster.size(); i++)
//	{
//		int cntSame = 0;
//		for (int j = i + 1; j < vecCluster.size(); j++)
//		{
//			if (i == j)
//				continue;
//			int recX = abs(vecCluster[i].x - vecCluster[j].x);
//			int recY = abs(vecCluster[i].y - vecCluster[j].y);
//			int recW = abs(vecCluster[i].x + vecCluster[i].width - vecCluster[j].x - vecCluster[j].width);
//			int recH = abs(vecCluster[i].y + vecCluster[i].height - vecCluster[j].y - vecCluster[j].height);
//
//			if ((recX < distance && recY < distance && recW < distance && recH < distance) || (recX + recY + recW + recH) < distance * 3)
//			{
//				cntSame++;
//
//				vecCluster[i].x = (int)(vecCluster[i].x*(cntSame + 1) + vecCluster[j].x) / (cntSame + 2);
//				vecCluster[i].y = (int)(vecCluster[i].y*(cntSame + 1) + vecCluster[j].y) / (cntSame + 2);
//				vecCluster[i].width = (int)(vecCluster[i].width*(cntSame + 1) + vecCluster[j].width) / (cntSame + 2);
//				vecCluster[i].height = (int)(vecCluster[i].height*(cntSame + 1) + vecCluster[j].height) / (cntSame + 2);
//				//cout <<"clustering : "<<i<<"-- "<<  vecRect[i].x<<" " <<vecRect[i].y <<" "<< vecRect[i].width <<" "<<vecRect[i].height<<endl;
//				vecCluster.erase(vecCluster.begin() + j);
//				j--;
//			}
//		}
//		cntSame = 0;
//	}
//}
//void Tracking_Validation(Mat &srcImage, vector<Rect> &vecRectTracking, vector<Rect>& vecRectDetect, vector<SKalman>& MultiKF, Track_t& High, int& cntframe, Rect& ROIset, float& fscale, Mat& imgROImask)
//{
//
//
//	//clustering//
//	clustering(vecRectTracking, 10);
//	char roiName[100] = { 0 };
//
//	////Tracking 시작
//
//	double time4 = (double)getTickCount();
//
//	vector<Rect> vecDetect;
//	vector<Rect> vecValidRec;
//	Point pCurrent;
//	float scaledist = 0.4;
//	int cntCandidate = 2; // 후보 ROI 검증 횟수, 검증되면 tracking 시작
//	int cntBefore = 4; // 못찾은 횟수가 연속으로 n프레임 이상이면 ROI 제거
//	int frameCandiate = 5; // 못찾은 횟수가 연속으로 n프레임 이상이면 후보 ROI 제거
//
//	kalmanMultiTarget(srcImage, vecRectTracking, High, MultiKF, scaledist, cntCandidate, cntBefore, frameCandiate, cntframe, ROIset, vecValidRec, imgROImask);
//
//
//	time4 = (double)getTickCount() - time4;
//	//printf( "tracking : %f ms.\n", time4*1000./getTickFrequency() );
//
//	vector<Rect> vecValid;
//
//	vecValid = High.vecBefore;
//	//vecValid.resize((int)High.vecBefore.size());
//	//copy(High.vecBefore.begin(),High.vecBefore.end(),vecValid);
//
//
//	/////validation/////
//	double time5 = (double)getTickCount();
//	int szLager = 10;
//	for (int i = 0; i<vecValid.size(); i++)
//	{
//		if (vecValid[i].x<szLager || vecValid[i].x>srcImage.cols - szLager || vecValid[i].y<szLager || vecValid[i].y> srcImage.rows - szLager) // 예외처리
//			continue;
//		int TempArea = vecValid[i].area();
//		Rect recOri = vecValid[i];
//
//		vecValid[i].x -= szLager;
//		vecValid[i].y -= szLager;
//		vecValid[i].width += szLager * 2;
//		vecValid[i].height += szLager * 2;
//		//cout << "vecValid[i].height: "<<vecValid[i].height <<" High.vecBefore[i].height: "<<High.vecBefore[i].height<<endl;
//		int margin = 3;
//		//if(vecValid[i].x-margin<=ROIset.x || vecValid[i].y-margin <= ROIset.y || vecValid[i].x + vecValid[i].width+margin >= ROIset.x+ROIset.width) // 예외처리
//		if (imgROImask.at<uchar>((vecValid[i].y - margin < 0 ? 0 : vecValid[i].y - margin), (vecValid[i].x - margin <0 ? 0 : vecValid[i].x - margin)) == 0 || imgROImask.at<uchar>((vecValid[i].y + vecValid[i].height - margin > srcImage.rows - 1 ? srcImage.rows : vecValid[i].y + vecValid[i].height - margin), (vecValid[i].x + vecValid[i].width - margin > srcImage.cols ? (srcImage.cols - 1) : (vecValid[i].x + vecValid[i].width - margin))) == 0) // 예외처리
//			continue;
//
//
//
//		Rect recValid;
//		Rect recDetect;
//		//Mat imgValid;
//		float overlap = 0;
//		for (int j = 0; j<MserVec.vecRectMser.size(); j++)
//		{
//			//MserVec.vecRectMser[j].x += ROIset.x;
//			//MserVec.vecRectMser[j].y += ROIset.y;
//			//rectangle(srcImage,vecRectMser[j],CV_RGB(0,255,0),2);
//			recValid = vecValid[i] & MserVec.vecRectMser[j];
//			float overlapRate = (float)recValid.area() / (float)vecValid[i].area();
//			//if(overlap < overlapRate && MserVec.vecRectMser[j].x > vecValid[i].x&& MserVec.vecRectMser[j].y > vecValid[i].y && MserVec.vecRectMser[j].x +MserVec.vecRectMser[j].width < vecValid[i].x + vecValid[i].width&& MserVec.vecRectMser[j].y +MserVec.vecRectMser[j].height < vecValid[i].y + vecValid[i].height &&(float)MserVec.vecRectMser[j].width/(float)MserVec.vecRectMser[j].height>0.8 && (float)MserVec.vecRectMser[j].width/(float)MserVec.vecRectMser[j].height<1.2)
//			//if(overlap < overlapRate && MserVec.vecRectMser[j].x > vecValid[i].x&& MserVec.vecRectMser[j].y > vecValid[i].y && MserVec.vecRectMser[j].x +MserVec.vecRectMser[j].width < vecValid[i].x + vecValid[i].width&& MserVec.vecRectMser[j].y +MserVec.vecRectMser[j].height < vecValid[i].y + vecValid[i].height /*&&(float)MserVec.vecRectMser[j].width/(float)MserVec.vecRectMser[j].height>0.8 && (float)MserVec.vecRectMser[j].width/(float)MserVec.vecRectMser[j].height<1.2*/)
//			if (overlap < overlapRate && MserVec.vecRectMser[j].x > vecValid[i].x&& MserVec.vecRectMser[j].y > vecValid[i].y && MserVec.vecRectMser[j].x + MserVec.vecRectMser[j].width < vecValid[i].x + vecValid[i].width&& MserVec.vecRectMser[j].y + MserVec.vecRectMser[j].height < vecValid[i].y + vecValid[i].height)
//			{
//				if (((float)MserVec.vecRectMser[j].width / (float)MserVec.vecRectMser[j].height>0.8 && (float)MserVec.vecRectMser[j].width / (float)MserVec.vecRectMser[j].height<1.2) || ((float)MserVec.vecRectMser[j].width / (float)MserVec.vecRectMser[j].height>1.5 && (float)MserVec.vecRectMser[j].width / (float)MserVec.vecRectMser[j].height<6))
//				{
//					overlap = overlapRate;
//					recDetect = MserVec.vecRectMser[j];
//				}
//				//imgValid = MserVec.vecImgMser[j];
//				/*imshow("imgMserSegment",MserVec.vecImgMser[j]);
//				waitKey(0);*/
//			}
//
//		}
//		if (recDetect.width != 0)
//		{
//
//
//			if ((float)recDetect.area()*1.2>TempArea && (float)recDetect.area()*0.8<TempArea)
//			{
//				//resize(imgValid,imgValid,Size(50,50));
//				//imshow("imgMserSegment",imgValid);
//				//waitKey(0);
//				//					sprintf_s(roiName,"3sunny\\roi2%d_%d.png",cntframe,i);
//				//					imwrite(roiName,srcImage(recDetect));
//				//rectangle(srcImage,recDetect,CV_RGB(0,255,255),2);
//				//rectangle(srcImage,recDetect,MultiKF[i].rgb,3);
//
//
//				rectangle(imgshow, recDetect, MultiKF[i].rgb, 2);
//				Point ptCenter = Point(recDetect.x + recDetect.width / 2, recDetect.y + recDetect.height / 2);
//				drawCross(imgshow, ptCenter, MultiKF[i].rgb, 5);
//				line(imgshow, ptVanishing, ptCenter, MultiKF[i].rgb, 2);
//				CNN(srcImage, recDetect);
//
//
//				float fDistXbefore = ptVanishing.x - ptCenter.x;
//				float fDistYbefore = ptVanishing.y - ptCenter.y;
//
//				float fAngleend = fAngle(fDistXbefore, fDistYbefore);
//
//				char szAngle[50];
//				sprintf_s(szAngle, "%.1f", fAngleend - 90);
//
//				putText(imgshow, szAngle, Point((ptVanishing.x *(i + 1) / (2 + i) + ptCenter.x / (2 + i)), (ptVanishing.y *(i + 1) / (2 + i) + ptCenter.y / (2 + i))), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(MultiKF[i].rgb.val[2] * 2 / 3, MultiKF[i].rgb.val[1] * 2 / 3, MultiKF[i].rgb.val[0] * 2 / 3), 2);
//
//
//
//
//			}
//			else
//			{
//				//					sprintf_s(roiName,"3sunny\\roi2%d_%d.png",cntframe,i);
//				//					imwrite(roiName,srcImage(recOri));
//				//rectangle(srcImage,recOri,CV_RGB(0,255,255),2);
//				//	rectangle(srcImage,recDetect,MultiKF[i].rgb,3);
//
//				rectangle(imgshow, recOri, MultiKF[i].rgb, 2);
//				Point ptCenter = Point(recOri.x + recOri.width / 2, recOri.y + recOri.height / 2);
//				drawCross(imgshow, ptCenter, MultiKF[i].rgb, 5);
//				line(imgshow, ptVanishing, ptCenter, MultiKF[i].rgb, 2);
//				CNN(srcImage, recOri);
//
//				float fDistXbefore = ptVanishing.x - ptCenter.x;
//				float fDistYbefore = ptVanishing.y - ptCenter.y;
//
//				float fAngleend = fAngle(fDistXbefore, fDistYbefore);
//
//				char szAngle[50];
//				sprintf_s(szAngle, "%.1f", fAngleend - 90);
//
//				putText(imgshow, szAngle, Point((ptVanishing.x *(i + 1) / (2 + i) + ptCenter.x / (2 + i)), (ptVanishing.y *(i + 1) / (2 + i) + ptCenter.y / (2 + i))), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(MultiKF[i].rgb.val[2] * 2 / 3, MultiKF[i].rgb.val[1] * 2 / 3, MultiKF[i].rgb.val[0] * 2 / 3), 2);
//
//				//rectangle(srcImage,vecValid[i],MultiKF[i].rgb,2);
//			}
//
//		}
//
//		overlap = 0;
//	}
//
//
//
//
//	time5 = (double)getTickCount() - time5;
//	//printf( "validation : %f ms.\n", time5*1000./getTickFrequency() );
//	MserVec.vecRectMser.clear();
//	MserVec.vecImgMser.clear();
//	vecRectTracking.clear();
//}
//


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


	//cout << "MultiKF.smeasurement : " << MultiKF.smeasurement << endl;
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
		//cout << "prediction : " << prediction << endl;
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
