#include <iostream>
#include <stdio.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <highgui.h>
//#include "structure.h"
#include "ConvNet.h"
#include "kalmanFilter.h"
#include "EM.h"
using namespace std;
using namespace cv;


struct SrecSize{
	float minRectAreaRED;
	float maxRectAreaRED;
	float minRectAreaBLUE;
	float maxRectAreaBLUE;

	float minRectAreaBLACK;
	float maxRectAreaBLACK;
};
typedef struct vecTracking{
	vector<Rect> vecBefore;
	vector<Rect> vecCandidate;
	vector<int> vecCount;
	vector<Point> vecCountPush;
	vector<Point2f> vecRtheta;
	vector<Point2f> vecAngle;
}Track_t;

typedef struct vecMSERs{
	vector<Rect> vecRectMser;
	vector<vector<Point> > vecPoints;
	vector<Mat> vecImgMser;
}Mser_t;

typedef struct imgColor{
	Mat imgRedMask;
	Mat imgBlueMask;
	Mat imgGreenMask;
	Mat imgBlackMask;
	Mat imgGreenLight;
}Color_t;

typedef struct imgAcromatic{
	Mat imgAcroMaskBlue;
	Mat imgAcroMaskRed;
	Mat imgAcroMaskMSER;
	Mat imgAcroMaskLight;
}Acro_t;



static const Vec3b bcolors[] =
{
	Vec3b(0, 0, 255),
	Vec3b(0, 128, 255),
	Vec3b(0, 255, 255),
	Vec3b(0, 255, 0),
	Vec3b(255, 128, 0),
	Vec3b(255, 255, 0),
	Vec3b(255, 0, 0),
	Vec3b(255, 0, 255),
	Vec3b(255, 255, 255)
};
float AngleTransform(const float &tempAngle, const int &scale){
	return (float)CV_PI / 180 * (tempAngle*scale);
}
#define ToRadian(degree) ((degree)*(CV_PI/180.0f))
#define ToDegree(radian) ((radian)*(180.0f/CV_PI))
float fAngle(int x, int y)
{
	return ToDegree(atan2f(y, x));
}


//void HistVec(Mat& , Mat& );
void ColorMask(Color_t& CM, Mat& imgHue, Mat& imgRGB);
void GrayImgFilter(Mat& imgV, Mat& imgThreshold);
//void MSERfilter(Mat&, Mat&, Mat&, Mat&, Mat&, Mat&, vector<Rect>&);
void MSERsegmentation(MSER& mser, Mat& imgInputMSER, Mat& imgOrigin, Color_t& CM, Mat& imgAdapThres, vector<Rect>& vecRect, vector<vector<Point> >& points, bool bRed, bool bBlue, bool bBlack, Mat& imgProb, Acro_t& AM, Mser_t& MserVec, Rect & ROIset, float& fscale);
void Acromatic(Mat& imgRGB, Mat& imgMask, float fvalue, uchar uThreshold);
//void ExtractDOTfeature(const Mat &, const SDotParameter &, vector<SDotValue> &);
//bool GetDOTfeature(Mat& , vector<SDotValue> &);
//void MSERring(MSER& , Mat& , Mat&);
void kalmanTrackingStart(SKalman& temKalman, Rect& recStart);
//void kalmanMultiTarget(Mat&, vector<Rect>&, vector<SKalman>&, float, int, int, int, int&, bool&, bool&);
void kalmanMultiTarget(Mat& srcImage, vector<Rect>& vecRectTracking, Track_t& Set, vector<SKalman>& MultiKF, float scaledist, int cntCandidate, int cntBefore, int frameCandiate, int& cntframe, Rect& ROIset, vector<Rect>& vecValidRec, Mat& imgROImask);
//void MSERroi(MSER& ,Mat& , Mat& ,Rect &, vector<vector<Point> >& , vector<vector<Point> >& );
void Grouping(Rect& ROIset, Mat& srcImage, int& cntframe, float& fscale, vector<Rect>& vecRect, Mser_t& MserVec, vector<Rect>& vecRectTracking);
void Tracking_Validation(Mat &srcImage, vector<Rect> &vecRectTracking, Mser_t& MserVec, vector<SKalman>& MultiKF, Track_t& High, int& cntframe, Rect& ROIset, float& fscale, Mat& imgROImask);
void CNN(Mat& imgsrc, Rect & rec);
void clustering(vector<Rect>& vecCluster, int distance);
void enhancement(Mat imgR, Mat imgG, Mat imgB, Mat& imgDst, Mat& imgDst2);

Mat invTri;
Mat RectangleM;
Mat imgdrawTest;
Mat imgshow;
Point ptVanishing;

//vector<Point> vecPoint;
SrecSize recSZ;
int cntmser;
CvNormalBayesClassifier *bayesian = new CvNormalBayesClassifier;
Mat testest;




double avg_detection = 0;
double avg_tracking = 0;
double avg_cnn = 0;
double avg_tot = 0;
int cntdetection = 0;
int cnttracking = 0;
int cntcnn = 0;

int main()
{

	// ------------------------------------------------------------ Kalman Filter setup
	vector<SKalman> MultiKF_U;
	vector<SKalman> MultiKF_R;


	///////////////////////////////////////////////////////////////
	char Xmlfilename[100] = "20classTS.xml";
	initRead(Xmlfilename);
	bayesian->load("trainDtBs.xml");


	//invTri = imread("invtri.png",0);
	//RectangleM = imread("rectangle.png",0);
	/* Load Video */
	//cv::VideoCapture m_videoCapture("sunny.wmv");
	//cv::VideoCapture m_videoCapture("blackbox2hd.wmv");
	//	string source = "2015-03-05-11h-52m-26s_F_normal.mp4"; //2015-03-18-17h-26m-33s_F_normal//
	string source = "sunny_03_urban.wmv";

	//	string source = "2015-03-18-10h-10m-20s_F_normal.mp4";
	//	string source = "2015-03-18-17h-20m-32s_F_normal.mp4";
	//	string source = "2015-03-18-17h-26m-33s_F_normal.mp4";
	//	string source = "2015-03-07-09h-18m-05s_F_event.mp4";
	//	string source = "2015-03-15-15h-08m-11s_F_event.mp4";
	//	string source = "2015-03-18-10h-08m-20s_F_normal.mp4";
	cv::VideoCapture m_videoCapture(source);
	if (!m_videoCapture.isOpened()){
		cout << "Could not load the video file." << endl;
		return -1;
	}
	cout << "Successfully loaded the video file." << endl;
	///////////////////////////////////////////////////

	unsigned int frameCount = m_videoCapture.get(CV_CAP_PROP_FRAME_COUNT);
	cout << frameCount << endl;
	Mat srcImage;
	Mat imgROImask;
	Mat imgDifference;
	Mat imgTempDiff;
	Mat imgDetect;
	int cntframe = 0;
	int numframe = 0;
	float fscale = 1.0f;

	recSZ.minRectAreaRED = 400 * fscale;
	recSZ.maxRectAreaRED = 4000 * fscale;
	recSZ.minRectAreaBLUE = 400 * fscale;
	recSZ.maxRectAreaBLUE = 4000 * fscale;
	recSZ.minRectAreaBLACK = 100 * fscale;
	recSZ.maxRectAreaBLACK = 3000 * fscale;



	//ptVanishing.x = 640-30;
	//ptVanishing.y = 360-20;


	ptVanishing.x = 640 - 20;
	ptVanishing.y = 360;


	// tracker //
	Track_t High;
	Mser_t MserVec;
	while (true)
	{


		cntframe++;

		cout << cntframe << endl;
		cntmser = cntframe;
		/* Capture Image Frame */
		m_videoCapture >> srcImage;
		if (srcImage.empty())	break;
		///////////////////
		//if (cntframe < 60)
		//	continue;

		imgshow = srcImage.clone();

		double t = (double)getTickCount();
		///////////////////
		circle(imgshow, ptVanishing, 1, CV_RGB(255, 0, 0), 2);
		line(imgshow, Point(ptVanishing.x, 0), Point(ptVanishing.x, ptVanishing.y), CV_RGB(0, 255, 0));
		//line(imgshow, Point((ptVanishing.x + ptVanishing.y*tan(ToRadian(60))>imgshow.cols ? imgshow.cols : ptVanishing.x + ptVanishing.y*tan(ToRadian(60))), 0), Point(ptVanishing.x, ptVanishing.y), CV_RGB(0, 255, 0),2);
		line(imgshow, Point(imgshow.cols, ptVanishing.y), Point(ptVanishing.x, ptVanishing.y), CV_RGB(255, 255, 255), 2);
		line(imgshow, Point((ptVanishing.x - (ptVanishing.y*tan(ToRadian(50)))>imgshow.cols ? imgshow.cols : ptVanishing.x - (ptVanishing.y*tan(ToRadian(50)))), 0), Point(ptVanishing.x, ptVanishing.y), CV_RGB(255, 255, 255), 2);





		srcImage.copyTo(imgROImask);
		srcImage.copyTo(imgDetect);
		imgROImask = Mat::zeros(srcImage.size(), CV_8UC1);

		vector<Rect> vecRect;
		vector<Rect> vecRectTracking;
		//Rect ROIset = Rect_<int>(srcImage.cols/3,srcImage.rows/4-srcImage.rows/16,srcImage.cols*3/8,srcImage.rows/4-srcImage.rows/8+srcImage.rows/16);
		Rect ROIset_U = Rect_<int>(srcImage.cols / 3 - srcImage.cols / 8, srcImage.rows / 4 - srcImage.rows / 8 - srcImage.rows / 16, srcImage.cols * 5 / 8, srcImage.rows / 4 /* - srcImage.rows / 8 + srcImage.rows / 16*/);
		//Rect ROIset_U = Rect_<int>(srcImage.cols / 3 - srcImage.cols / 8, srcImage.rows / 4 - srcImage.rows / 8 - srcImage.rows / 16, srcImage.cols * 5 / 8, srcImage.rows / 4 - srcImage.rows / 8 + srcImage.rows / 16);
		Rect ROIset_R = Rect_<int>(srcImage.cols * 2 / 3 - srcImage.cols / 16, srcImage.rows / 4 - srcImage.rows / 32, srcImage.cols / 4, srcImage.rows / 4);

		//Rect ROIset_T = Rect_<int>(0, 0, srcImage.cols , srcImage.rows / 2);
		//cout << "area : " << ROIset_U.width << " " << ROIset_U.height << endl;

		Rect ROIset_T;
		//ROIset_T.x = (ROIset_U.x < ROIset_R.x) ? ROIset_U.x : ROIset_R.x;
		//ROIset_T.y = (ROIset_U.y < ROIset_R.y) ? ROIset_U.y : ROIset_R.y;
		//ROIset_T.width = (ROIset_U.x + ROIset_U.width > ROIset_R.x + ROIset_R.width) ? ROIset_U.x + ROIset_U.width - ROIset_T.x : ROIset_R.x + ROIset_R.width - ROIset_T.x;
		//ROIset_T.height = (ROIset_U.y + ROIset_U.height > ROIset_R.y + ROIset_R.height) ? ROIset_U.y + ROIset_U.height - ROIset_T.y : ROIset_R.y + ROIset_R.height - ROIset_T.y;

		//imgROImask(ROIset_T).setTo(Scalar::all(255));

		imgROImask(ROIset_U).setTo(Scalar::all(255));
		imgROImask(ROIset_R).setTo(Scalar::all(255));

		//Grouping(ROIset_T, srcImage, cntframe, fscale, vecRect, MserVec, vecRectTracking);
		double t2 = (double)getTickCount();
		Grouping(ROIset_U, srcImage, cntframe, fscale, vecRect, MserVec, vecRectTracking);
		Grouping(ROIset_R, srcImage, cntframe, fscale, vecRect,MserVec, vecRectTracking);

		t2 = (double)getTickCount() - t2;
		//	printf("Traffic Sign detection : %f ms.\n", t2*1000. / getTickFrequency());

		double cudetection = t2*1000. / getTickFrequency();

		avg_detection += cudetection;

		cntdetection++;

		printf("Avg Traffic Sign detection : %f ms.\n", avg_detection / cntdetection);



		//cout << "vecRect.size() : " << vecRect.size()<< "vecRectTracking.size() : " << vecRectTracking.size()<<endl;
		for (int i = 0; i< vecRectTracking.size(); i++)
		{

			rectangle(imgDetect, vecRectTracking[i], CV_RGB(255, 0, 255), 2);
		}
		Tracking_Validation(srcImage, vecRectTracking, MserVec, MultiKF_U, High, cntframe, ROIset_U, fscale, imgROImask);
		t = (double)getTickCount() - t;
		printf("Traffic Sign detection : %f ms.\n", t*1000. / getTickFrequency());

		double cutot = t*1000. / getTickFrequency();
		avg_tot += cutot;
		printf("Avg Traffic Sign module : %f ms.\n", avg_tot / cntdetection);


		vector<Vec4i> hierarchy;
		vector<vector<Point> > contours;
		findContours(imgROImask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		//Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(0, 255, 255);
			drawContours(imgshow, contours, i, color, 2, 8, hierarchy, 0, Point());
		}






		//rectangle(srcImage,ROIset_U,CV_RGB(255,255,0),2);
		//rectangle(srcImage,ROIset_R,CV_RGB(255,255,0),2);
		//imshow("imgDetect", imgDetect);
		//imshow("original", srcImage);
		imshow("imgshow", imgshow);
		//imshow("drawimg",imgdrawTest);



		//imshow ("mask",imgROImask);
		int waitTime;
		(t*1000. / getTickFrequency()<70) ? waitTime = 70 - t*1000. / getTickFrequency() : waitTime = 70;
		char c = waitKey(1);
		if (c == 'q')	break;
	}
	return 0;
}


void Grouping(Rect& ROIset, Mat& srcImage, int& cntframe, float& fscale, vector<Rect>& vecRect, Mser_t& MserVec, vector<Rect>& vecRectTracking)
{
	double time1 = (double)getTickCount();

	int bBlack = 0;
	if (ROIset.x == srcImage.cols / 3)
		bBlack = 0;

	double time2 = (double)getTickCount();
	Mat imgInput = srcImage(ROIset);
	resize(imgInput, imgInput, Size(cvRound(fscale*imgInput.cols), cvRound(fscale*imgInput.rows)), 0, 0, INTER_CUBIC);

	/*imshow("original", cv::MSER::MSER(int, int, int, double, double, int, double, double, int)'
	srcImage);*/

	//imgInput(Rect_<int>(imgInput.cols/4,imgInput.rows/2,imgInput.cols/2,imgInput.rows/2)).setTo(0);
	//vector<Mat> vecRGB;
	//split(imgInput,vecRGB);
	/////////////  HSV  /////////////////
	Mat imgHSV;
	Mat imgYCrCb;
	Mat imgDst;
	Mat imgGRAY;
	cvtColor(imgInput, imgGRAY, CV_BGR2GRAY);
	cvtColor(imgInput, imgHSV, CV_BGR2HSV);
	vector<Mat> vecHSV;
	split(imgHSV, vecHSV);



	vector<Mat> vecRGB;
	split(imgInput, vecRGB);

	Mat imgEnhanceR(imgInput.size(), CV_8UC1);
	Mat imgEnhanceB(imgInput.size(), CV_8UC1);
	enhancement(vecRGB[2], vecRGB[1], vecRGB[0], imgEnhanceR, imgEnhanceB);

	imgEnhanceR += imgEnhanceB;

	//imshow("imgEnhanceB", imgEnhanceB);

	//imshow("imgEnhanceR", imgEnhanceR);
	//	equalizeHist(vecHSV[1],vecHSV[1]);
	//equalizeHist(vecHSV[2],vecHSV[2]);
	////imshow("H",vecHSV[0]);
	//vecHSV[1] = vecHSV[1]/2+vecYCrCb[1]/2;

	//		imshow("S",vecHSV[1]);
	//		imshow("V",vecHSV[2]);




	/////////////////////////////////////////////////////////////////////////////////

	Acro_t AM;

	AM.imgAcroMaskBlue = Mat::zeros(imgInput.rows, imgInput.cols, CV_8UC1);
	AM.imgAcroMaskRed = Mat::zeros(imgInput.rows, imgInput.cols, CV_8UC1) + 255;
	//AM.imgAcroMaskMSER = Mat::zeros(imgInput.rows, imgInput.cols,CV_8UC1);
	//AM.imgAcroMaskLight = Mat::zeros(imgInput.rows, imgInput.cols,CV_8UC1);

	//		Acromatic(imgInput, imgAcroMaskMSER, 10, 30); 
	Acromatic(imgInput, AM.imgAcroMaskBlue, 10, 30);
	Acromatic(imgInput, AM.imgAcroMaskRed, 10, 30);
	//	Acromatic(imgInput, AM.imgAcroMaskMSER, 10, 0);
	//	Acromatic(imgInput, AM.imgAcroMaskLight, 30, 0);


	//AM.imgAcroMaskMSER = 255 - AM.imgAcroMaskMSER;

	//		imshow("imgAcroMaskMSER",AM.imgAcroMaskMSER);
	//	imshow("imgAcroMaskLight",AM.imgAcroMaskLight);
	//imshow("imgAcroMaskBlue",imgAcroMaskBlue);
	//imshow("imgAcroMaskRed",imgAcroMaskRed);
	/////////////////////////////////////////////////////////////////////////////////
	//////////////////////////  ColorMask  //////////////////////////////////////////
	Color_t CM;
	CM.imgRedMask = Mat::zeros(vecHSV[0].rows, vecHSV[0].cols, CV_8UC1);
	CM.imgBlueMask = Mat::zeros(vecHSV[0].rows, vecHSV[0].cols, CV_8UC1);
	//CM.imgGreenMask=Mat::zeros(vecHSV[0].rows,vecHSV[0].cols,CV_8UC1);
	//CM.imgBlackMask = Mat::zeros(vecHSV[0].rows,vecHSV[0].cols,CV_8UC1);
	//CM.imgGreenLight = Mat::zeros(vecHSV[0].rows,vecHSV[0].cols,CV_8UC1);
	ColorMask(CM, vecHSV[0], imgInput);
	//Mat imgGrayMask;
	//GrayImgFilter(vecRGB[2],imgGrayMask);//v theshold
	//bitwise_and(imgBlueMask,imgGrayMask,imgBlueMask);
	//		imshow("CM.imgBlackMask1",CM.imgBlackMask);
	bitwise_and(CM.imgBlueMask, AM.imgAcroMaskBlue, CM.imgBlueMask);





	////////////////////////////////////////////////////////////////////////////////
	//////////////////////// Adaptive Threshold ////////////////////////////////////
	Mat imgAdapThres;
	adaptiveThreshold(vecHSV[2], imgAdapThres, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 31, 5);
	//adaptiveThreshold(vecHSV[2],imgAdapThres,255,ADAPTIVE_THRESH_GAUSSIAN_C,CV_THRESH_BINARY_INV,21,15);
	//	imshow("imgAdapThres",imgAdapThres);
	////////////////////////////////////////////////////////////////////////////////
	merge(vecHSV, imgHSV);
	cvtColor(imgHSV, imgDst, CV_HSV2BGR);
	//	imshow("dst", imgDst);

	time1 = (double)getTickCount() - time1;
	//printf( "imgprocessing : %f ms.\n", time1*1000./getTickFrequency() );
	////////////////////////////////////////////////////////////////////////////////
	////////////////////////// MSER ////////////////////////////////////////////////



	Mat imgSroi;
	//bitwise_and(imgSum, vecHSV[1], vecHSV[1]);
	//bitwise_and(imgSum, vecHSV[2], vecHSV[2]);
	//		imshow("imgSum2", vecHSV[1]);
	//		imshow("imgSum3", vecHSV[2]);
	vector<vector<Point> > points;
	vector<vector<Point> > graypoints;
	imgInput.copyTo(imgdrawTest);
	testest = imgInput.clone();

	Mat imgProb = Mat::zeros(vecHSV[1].rows, vecHSV[1].cols, CV_8UC1);
	//MSER mserSaturation(5,50, 800, 0.2, 0.1, 250, 1.01, 0.003, 5); //0.7
	MSER mserSaturation(7, 15, 700, 0.5, 0.1, 250, 1.01, 0.003, 5);
	MSERsegmentation(mserSaturation, vecHSV[1], imgInput, CM, imgAdapThres, vecRect, points, 1, 1, 0, imgProb, AM, MserVec, ROIset, fscale);
	//MSERsegmentation(mserSaturation, imgP1, imgInput, CM, imgAdapThres, vecRect, points, 1, 1, 0, imgProb, AM, MserVec, ROIset, fscale);
	MSER mserGray(7, 15, 700, 0.5, 0.1, 250, 1.01, 0.003, 5);
	MSERsegmentation(mserGray, vecHSV[2], imgInput, CM, imgAdapThres, vecRect, graypoints, 1, 1, 0, imgProb, AM, MserVec, ROIset, fscale);
	//MSERsegmentation(mserGray, imgP2, imgInput, CM, imgAdapThres, vecRect, graypoints, 1, 1, 0, imgProb, AM, MserVec, ROIset, fscale);
	//		imshow("drawimg",imgdrawTest);
	//imshow("imgProb", imgProb);
	//		imshow("testest", testest);

	//waitKey(0);
	time2 = (double)getTickCount() - time2;
	printf("MSER detection : %f ms.\n", time2*1000. / getTickFrequency());

	//// clustering /////
	//for(int i =0; i<MserVec.vecImgMser.size(); i++)
	//{
	//Mat imgValid;
	//imgValid = MserVec.vecImgMser[i];
	//resize(imgValid,imgValid,Size(50,50));
	//imshow("imgMserSegment",imgValid);
	//waitKey(0);
	//}
	double time3 = (double)getTickCount();
	clustering(vecRect, 10);
	//for(int i=0; i< vecRect.size(); i++)
	//{
	//	rectangle(imgInput,vecRect[i],CV_RGB(0,0,255),2);
	//}
	MserVec.vecPoints = points;
	MserVec.vecPoints.insert(MserVec.vecPoints.begin() + points.size(), graypoints.begin(), graypoints.end());
	//cout << "vecRectMser : "<<MserVec.vecRectMser.size()<< "vecPoints.size : "<<MserVec.vecPoints.size()<<"vecimgMser : "<<MserVec.vecImgMser.size()<<endl;
	points.clear();
	graypoints.clear();
	//imshow("imgInput", imgInput);
	time3 = (double)getTickCount() - time3;
	//printf( "time Detection : %f ms.\n", time3*1000./getTickFrequency() );



	for (int i = 0; i< vecRect.size(); i++)
	{
		Rect rec = Rect(Point(cvRound(vecRect[i].x / fscale + ROIset.x), cvRound(vecRect[i].y / fscale + ROIset.y)), Point(cvRound((vecRect[i].x + vecRect[i].width) / fscale + ROIset.x), cvRound((vecRect[i].y + vecRect[i].height) / fscale + ROIset.y)));
		vecRectTracking.push_back(rec);

	}

}

void Tracking_Validation(Mat &srcImage, vector<Rect> &vecRectTracking, Mser_t& MserVec, vector<SKalman>& MultiKF, Track_t& High, int& cntframe, Rect& ROIset, float& fscale, Mat& imgROImask)
{


	//clustering//
	clustering(vecRectTracking, 10);
	char roiName[100] = { 0 };

	////Tracking
	double time4 = (double)getTickCount();

	vector<Rect> vecDetect;
	vector<Rect> vecValidRec;
	Point pCurrent;
	float scaledist = 0.4;
	int cntCandidate = 2;
	int cntBefore = 4;
	int frameCandiate = 5;

	kalmanMultiTarget(srcImage, vecRectTracking, High, MultiKF, scaledist, cntCandidate, cntBefore, frameCandiate, cntframe, ROIset, vecValidRec, imgROImask);


	time4 = (double)getTickCount() - time4;
	//	printf( "tracking : %f ms.\n", time4*1000./getTickFrequency() );

	double cutracking = time4*1000. / getTickFrequency();
	avg_tracking += cutracking;

	cnttracking++;

	printf("Avg tracking : %f ms.\n", avg_tracking / cnttracking);

	vector<Rect> vecValid;

	vecValid = High.vecBefore;
	//vecValid.resize((int)High.vecBefore.size());
	//copy(High.vecBefore.begin(),High.vecBefore.end(),vecValid);


	/////validation/////
	double time5 = (double)getTickCount();
	int szLager = 10;
	for (int i = 0; i<vecValid.size(); i++)
	{
		if (vecValid[i].x<szLager || vecValid[i].x>srcImage.cols - szLager || vecValid[i].y<szLager || vecValid[i].y> srcImage.rows - szLager)
			continue;
		int TempArea = vecValid[i].area();
		Rect recOri = vecValid[i];

		vecValid[i].x -= szLager;
		vecValid[i].y -= szLager;
		vecValid[i].width += szLager * 2;
		vecValid[i].height += szLager * 2;
		//cout << "vecValid[i].height: "<<vecValid[i].height <<" High.vecBefore[i].height: "<<High.vecBefore[i].height<<endl;
		int margin = 3;
		//if(vecValid[i].x-margin<=ROIset.x || vecValid[i].y-margin <= ROIset.y || vecValid[i].x + vecValid[i].width+margin >= ROIset.x+ROIset.width) 
		if (imgROImask.at<uchar>((vecValid[i].y - margin < 0 ? 0 : vecValid[i].y - margin), (vecValid[i].x - margin <0 ? 0 : vecValid[i].x - margin)) == 0 || imgROImask.at<uchar>((vecValid[i].y + vecValid[i].height - margin > srcImage.rows - 1 ? srcImage.rows : vecValid[i].y + vecValid[i].height - margin), (vecValid[i].x + vecValid[i].width - margin > srcImage.cols ? (srcImage.cols - 1) : (vecValid[i].x + vecValid[i].width - margin))) == 0)
			continue;



		Rect recValid;
		Rect recDetect;
		//Mat imgValid;
		float overlap = 0;
		for (int j = 0; j<MserVec.vecRectMser.size(); j++)
		{
			//MserVec.vecRectMser[j].x += ROIset.x;
			//MserVec.vecRectMser[j].y += ROIset.y;
			//rectangle(srcImage,vecRectMser[j],CV_RGB(0,255,0),2);
			recValid = vecValid[i] & MserVec.vecRectMser[j];
			float overlapRate = (float)recValid.area() / (float)vecValid[i].area();
			//if(overlap < overlapRate && MserVec.vecRectMser[j].x > vecValid[i].x&& MserVec.vecRectMser[j].y > vecValid[i].y && MserVec.vecRectMser[j].x +MserVec.vecRectMser[j].width < vecValid[i].x + vecValid[i].width&& MserVec.vecRectMser[j].y +MserVec.vecRectMser[j].height < vecValid[i].y + vecValid[i].height &&(float)MserVec.vecRectMser[j].width/(float)MserVec.vecRectMser[j].height>0.8 && (float)MserVec.vecRectMser[j].width/(float)MserVec.vecRectMser[j].height<1.2)
			//if(overlap < overlapRate && MserVec.vecRectMser[j].x > vecValid[i].x&& MserVec.vecRectMser[j].y > vecValid[i].y && MserVec.vecRectMser[j].x +MserVec.vecRectMser[j].width < vecValid[i].x + vecValid[i].width&& MserVec.vecRectMser[j].y +MserVec.vecRectMser[j].height < vecValid[i].y + vecValid[i].height /*&&(float)MserVec.vecRectMser[j].width/(float)MserVec.vecRectMser[j].height>0.8 && (float)MserVec.vecRectMser[j].width/(float)MserVec.vecRectMser[j].height<1.2*/)
			if (overlap < overlapRate && MserVec.vecRectMser[j].x > vecValid[i].x&& MserVec.vecRectMser[j].y > vecValid[i].y && MserVec.vecRectMser[j].x + MserVec.vecRectMser[j].width < vecValid[i].x + vecValid[i].width&& MserVec.vecRectMser[j].y + MserVec.vecRectMser[j].height < vecValid[i].y + vecValid[i].height)
			{
				if (((float)MserVec.vecRectMser[j].width / (float)MserVec.vecRectMser[j].height>0.8 && (float)MserVec.vecRectMser[j].width / (float)MserVec.vecRectMser[j].height<1.2) || ((float)MserVec.vecRectMser[j].width / (float)MserVec.vecRectMser[j].height>1.5 && (float)MserVec.vecRectMser[j].width / (float)MserVec.vecRectMser[j].height<6))
				{
					overlap = overlapRate;
					recDetect = MserVec.vecRectMser[j];
				}
				//imgValid = MserVec.vecImgMser[j];
				/*imshow("imgMserSegment",MserVec.vecImgMser[j]);
				waitKey(0);*/
			}

		}






		if (recDetect.width != 0)
		{


			if ((float)recDetect.area()*1.2>TempArea && (float)recDetect.area()*0.8<TempArea)
			{
				//resize(imgValid,imgValid,Size(50,50));
				//imshow("imgMserSegment",imgValid);
				//waitKey(0);
				//					sprintf_s(roiName,"3sunny\\roi2%d_%d.png",cntframe,i);
				//					imwrite(roiName,srcImage(recDetect));
				//rectangle(srcImage,recDetect,CV_RGB(0,255,255),2);
				//rectangle(srcImage,recDetect,MultiKF[i].rgb,3);


				rectangle(imgshow, recDetect, MultiKF[i].rgb, 2);
				Point ptCenter = Point(recDetect.x + recDetect.width / 2, recDetect.y + recDetect.height / 2);
				drawCross(imgshow, ptCenter, MultiKF[i].rgb, 5);
				line(imgshow, ptVanishing, ptCenter, MultiKF[i].rgb, 2);
				CNN(srcImage, recDetect);


				float fDistXbefore = ptVanishing.x - ptCenter.x;
				float fDistYbefore = ptVanishing.y - ptCenter.y;

				float fAngleend = fAngle(fDistXbefore, fDistYbefore);

				char szAngle[50];
				sprintf(szAngle, "%.1f", fAngleend - 90);

				putText(imgshow, szAngle, Point((ptVanishing.x *(i + 1) / (2 + i) + ptCenter.x / (2 + i)), (ptVanishing.y *(i + 1) / (2 + i) + ptCenter.y / (2 + i))), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(MultiKF[i].rgb.val[2] * 2 / 3, MultiKF[i].rgb.val[1] * 2 / 3, MultiKF[i].rgb.val[0] * 2 / 3), 2);




			}
			else
			{
				//					sprintf_s(roiName,"3sunny\\roi2%d_%d.png",cntframe,i);
				//					imwrite(roiName,srcImage(recOri));
				//rectangle(srcImage,recOri,CV_RGB(0,255,255),2);
				//	rectangle(srcImage,recDetect,MultiKF[i].rgb,3);

				rectangle(imgshow, recOri, MultiKF[i].rgb, 2);
				Point ptCenter = Point(recOri.x + recOri.width / 2, recOri.y + recOri.height / 2);
				drawCross(imgshow, ptCenter, MultiKF[i].rgb, 5);
				line(imgshow, ptVanishing, ptCenter, MultiKF[i].rgb, 2);
				CNN(srcImage, recOri);

				float fDistXbefore = ptVanishing.x - ptCenter.x;
				float fDistYbefore = ptVanishing.y - ptCenter.y;

				float fAngleend = fAngle(fDistXbefore, fDistYbefore);

				char szAngle[50];
				sprintf(szAngle, "%.1f", fAngleend - 90);

				putText(imgshow, szAngle, Point((ptVanishing.x *(i + 1) / (2 + i) + ptCenter.x / (2 + i)), (ptVanishing.y *(i + 1) / (2 + i) + ptCenter.y / (2 + i))), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(MultiKF[i].rgb.val[2] * 2 / 3, MultiKF[i].rgb.val[1] * 2 / 3, MultiKF[i].rgb.val[0] * 2 / 3), 2);

				//rectangle(srcImage,vecValid[i],MultiKF[i].rgb,2);
			}

		}

		overlap = 0;
	}




	time5 = (double)getTickCount() - time5;
	printf("validation : %f ms.\n", time5*1000. / getTickFrequency());

	double cucnn = time5*1000. / getTickFrequency();
	avg_cnn += cucnn;

	cntcnn++;

	printf("Avg cnn: %f ms.\n", avg_cnn / cntcnn);

	MserVec.vecRectMser.clear();
	MserVec.vecImgMser.clear();
	vecRectTracking.clear();
}





void kalmanMultiTarget(Mat& srcImage, vector<Rect>& vecRectTracking, Track_t& Set, vector<SKalman>& MultiKF, float scaledist, int cntCandidate, int cntBefore, int frameCandiate, int& cntframe, Rect& ROIset, vector<Rect>& vecValidRec, Mat& imgROImask)
{
	Point pCurrent;

	double time6 = (double)getTickCount();

	for (int i = 0; i < Set.vecBefore.size(); i++)
	{
		pCurrent.x = srcImage.cols;
		pCurrent.y = srcImage.rows;
		int num = -1;

		float fDistXbefore = ptVanishing.x - (Set.vecBefore[i].x + Set.vecBefore[i].width / 2);
		float fDistYbefore = ptVanishing.y - (Set.vecBefore[i].y + Set.vecBefore[i].height / 2);
		float fDistBefore = fDistXbefore*fDistXbefore + fDistYbefore*fDistYbefore;
		fDistBefore = sqrt(fDistBefore);
		float fAngleBefore = fAngle(fDistXbefore, fDistYbefore);


		for (int j = 0; j < vecRectTracking.size(); j++)
		{
			float fDistXCandi = ptVanishing.x - (vecRectTracking[j].x + vecRectTracking[j].width / 2);
			float fDistYCandi = ptVanishing.y - (vecRectTracking[j].y + vecRectTracking[j].height / 2);
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

		if (Set.vecBefore[i].y < 0 || Set.vecBefore[i].x<0 || Set.vecBefore[i].y + Set.vecBefore[i].height >= srcImage.rows || Set.vecBefore[i].x + Set.vecBefore[i].width >= srcImage.cols)
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


	for (int i = 0; i<Set.vecCandidate.size(); i++)
	{
		pCurrent.x = srcImage.cols;
		pCurrent.y = srcImage.rows;
		///////////
		int num = -1;
		for (int j = 0; j< vecRectTracking.size(); j++)
		{

			float dist = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2))*abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2))
				+ abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2))*abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2));
			dist = sqrt(dist);
			float pCurrnetXY = pCurrent.x*pCurrent.x + pCurrent.y*pCurrent.y;
			pCurrnetXY = sqrt(pCurrnetXY);
			if (dist < pCurrnetXY && dist <scaledist * 2 * Set.vecCandidate[i].width && vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2) <= 0)
			{

				float fDistXCandi = ptVanishing.x - (vecRectTracking[j].x + vecRectTracking[j].width / 2);
				float fDistYCandi = ptVanishing.y - (vecRectTracking[j].y + vecRectTracking[j].height / 2);
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
			float fDistXCandi = ptVanishing.x - (vecRectTracking[num].x + vecRectTracking[num].width / 2);
			float fDistYCandi = ptVanishing.y - (vecRectTracking[num].y + vecRectTracking[num].height / 2);
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
	for (int i = 0; i<vecRectTracking.size(); i++)
	{
		//Set.vecRtheta.push_back(

		float fDistXCandi = ptVanishing.x - (vecRectTracking[i].x + vecRectTracking[i].width / 2);
		float fDistYCandi = ptVanishing.y - (vecRectTracking[i].y + vecRectTracking[i].height / 2);
		float fDistCandi = fDistXCandi*fDistXCandi + fDistYCandi*fDistYCandi;
		fDistCandi = sqrt(fDistCandi);

		float fAngleCandi = fAngle(fDistXCandi, fDistYCandi);
		Set.vecRtheta.push_back(Point2f(fDistCandi, fAngleCandi));
		Set.vecCandidate.push_back(vecRectTracking[i]);
		Set.vecCountPush.push_back(Point(0, cntframe));
	}
	for (int i = 0; i<Set.vecCountPush.size(); i++)
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


	/// Kalman filtering  //////////////
	for (int i = 0; i<Set.vecBefore.size(); i++)
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
	printf("kalman filter : %f ms.\n", time7*1000. / getTickFrequency());
}

void kalmanTrackingStart(SKalman& temKalman, Rect& recStart)
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

void CNN(Mat& imgsrc, Rect & rec)
{
	///////////////////////////////////////////////////////////////////////
	char roiName[100] = { 0 };
	Mat img = imgsrc(rec).clone();


	vector<Mat> vecCNN;
	Mat imgCNN;
	resize(img, img, Size(32, 32), 0, 0, INTER_CUBIC);
	cvtColor(img, imgCNN, CV_RGB2GRAY);
	imgCNN.convertTo(imgCNN, CV_64FC1);
	vecCNN.push_back((imgCNN - 128) / 255.0);

	Mat results;
	cnnMain(vecCNN, results);
	results.convertTo(results, CV_32SC1);
	cout << "result : " << results.at<int>(0) << endl;
	int resultnum = results.at<int>(0);
	if (resultnum != 0)
	{
		cout << "resultnum : " << resultnum << endl;
		sprintf(roiName, "popup/show%d.png", resultnum);
		cout << roiName << endl;

		Mat imgPopup = imread(roiName, 1);
		//Mat imgPopup = Mat::zeros(100,100,CV_8UC3);
		//imshow("pop", imgPopup);
		resize(imgPopup, imgPopup, rec.size());
		Rect poprec;
		poprec.x = rec.x;
		poprec.y = rec.y + rec.height + 1;
		poprec.width = rec.width;
		poprec.height = rec.height;
		//imgsrc(poprec).cop
		imgPopup.copyTo(imgshow(poprec));


	}


	vecCNN.clear();

}

void Acromatic(Mat& imgRGB, Mat& imgMask, float fvalue, uchar uThreshold)
{
	float facromatic = 0;
	vector<Mat> vecRGB;
	split(imgRGB, vecRGB);
	int cols = imgRGB.cols;
	for (int i = 0; i<imgRGB.rows; i++)
	{
		for (int j = 0; j<imgRGB.cols; j++)
		{
			//float sum =(abs((float)vecRGB[0].at<uchar>(i*cols+j) - (float)vecRGB[1].at<uchar>(i*cols+j)) + abs((float)vecRGB[0].at<uchar>(i*cols+j) - (float)vecRGB[2].at<uchar>(i*cols+j)-10)+abs((float)vecRGB[1].at<uchar>(i*cols+j) - (float)vecRGB[2].at<uchar>(i*cols+j)-10));
			float sum = (abs((float)vecRGB[0].at<uchar>(i*cols + j) - (float)vecRGB[1].at<uchar>(i*cols + j)) + abs((float)vecRGB[0].at<uchar>(i*cols + j) - (float)vecRGB[2].at<uchar>(i*cols + j)) + abs((float)vecRGB[1].at<uchar>(i*cols + j) - (float)vecRGB[2].at<uchar>(i*cols + j)));
			facromatic = sum / (3 * fvalue);
			//cout<<"facromatic : "<< sum<<endl;
			if (facromatic <1)
				imgMask.at<uchar>(i*cols + j) = 0;
			else if (facromatic >= 1 || (vecRGB[0].at<uchar>(i*cols + j)<uThreshold&&vecRGB[1].at<uchar>(i*cols + j)<uThreshold&&vecRGB[2].at<uchar>(i*cols + j)<uThreshold))
				imgMask.at<uchar>(i*cols + j) = 255;

		}
	}


}
void GrayImgFilter(Mat& imgV, Mat& imgThreshold)
{


	threshold(imgV, imgThreshold, 70, 255, THRESH_BINARY_INV);
	//cv::Mat const structure_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//morphologyEx(imgThreshold,imgThreshold,MORPH_DILATE,structure_elem);
	//medianBlur(imgThreshold,imgThreshold,5);
	imshow("threshold", imgThreshold);


}


void MSERsegmentation(MSER& mser, Mat& imgInputMSER, Mat& imgOrigin, Color_t& CM, Mat& imgAdapThres, vector<Rect>& vecRect, vector<vector<Point> >& points, bool bRed, bool bBlue, bool bBlack, Mat& imgProb, Acro_t& AM, Mser_t& MserVec, Rect & ROIset, float& fscale)
{
	mser(imgInputMSER, points);
	double t = (double)getTickCount();
	//Mat imgdrawTest;
	/*imgOrigin.copyTo(imgdrawTest);*/



	Rect rec;
	for (int i = 0; i < (int)points.size(); i++)
	{


		rec = boundingRect(points.at(i));
		Rect recPush = rec;
		recPush.x = recPush.x / fscale + ROIset.x;
		recPush.y = recPush.y / fscale + ROIset.y;
		recPush.width = (float)recPush.width / fscale;
		recPush.height = (float)recPush.height / fscale;
		MserVec.vecRectMser.push_back(recPush);

		//////////////////////////////////////////
		Mat imgTempMser;//(rec.size(),CV_8UC1);
		vector<Point>& r = points[i]; //
		for (int j = 0; j < (int)r.size(); j++)
		{
			Point pt = r[j];
			imgProb.at<uchar>(pt) = 255;

		}

		Mat imgTempProb;
		imgProb.copyTo(imgTempProb);
		imgTempMser = imgTempProb(rec);
		MserVec.vecImgMser.push_back(imgTempMser);
		imgProb.setTo(0);
		/*imshow("imgtempmser",imgTempMser);
		waitKey(0);*/
		//////////////////////////////////////
		///int cntgreen = 0;
		int cntadapt = 0;
		for (int j = 0; j<points[i].size(); j++){
			//if (CM.imgGreenMask.at<uchar>(points[i][j]) != 0)
			//	cntgreen++;
			if (imgAdapThres.at<uchar>(points[i][j]) != 0)
				cntadapt++;
		}
		//////////////////////////


		if (rec.x>2)
			rec.x -= 2;
		if (rec.y>2)
			rec.y -= 2;
		if (rec.x + rec.width + 4<imgInputMSER.cols)
			rec.width += 4;
		if (rec.y + rec.height + 4<imgInputMSER.rows)
			rec.height += 4;

		if (rec.x<2 || rec.y<3 || (rec.x + rec.width)>imgOrigin.cols - 2)
			continue;

		if (bRed == 1){
			/////////////////////////////
			int count = 0;
			int cntred = 0;
			int cntacro = 0;
			Mat imgBlock = CM.imgRedMask(rec);
			for (int j = 0; j<imgBlock.cols*imgBlock.rows; j++)
			if (imgBlock.at<uchar>(j)>0)
				count++;

			//Mat imgAcro = AM.imgAcroMaskMSER(rec);
			//for (int j = 0; j<imgAcro.cols*imgAcro.rows; j++)
			//	if (imgAcro.at<uchar>(j)>0)
			//		cntacro++;

			for (int j = 0; j<points[i].size(); j++)
			if (CM.imgRedMask.at<uchar>(points[i][j]) != 0)
				cntred++;
			////////////////////////////////
			//for ( int j = 0; j < (int)r.size(); j++ )
			//{
			//	Point pt = r[j];
			//	imgdrawTest.at<Vec3b>(pt) = bcolors[i%9];

			//}
			//if((float)rec.width/(float)rec.height<1.3 && (float)rec.width/(float)rec.height>0.8&&rec.area()<recSZ.maxRectAreaRED &&rec.area()>recSZ.minRectAreaRED  &&(float)count/(float)rec.area()<0.7 && (float)count>1 &&(float)cntred/points[i].size()>0.05 && (float)cntgreen/(float)points[i].size()<0.2)
			if ((float)rec.width / (float)rec.height<1.3 && (float)rec.width / (float)rec.height>0.8&&rec.area()<recSZ.maxRectAreaRED &&rec.area()>recSZ.minRectAreaRED && (float)count / (float)rec.area()<0.9 && (float)count>1 && (float)cntred / points[i].size()>0.05 /*&& (float)cntgreen / (float)points[i].size()<0.2*/ && (float)cntadapt / (float)points[i].size()>0.6)
				//if((float)rec.width/(float)rec.height<1.3 && (float)rec.width/(float)rec.height>0.8&&rec.area()<2000  &&(float)count/(float)rec.area()<0.7 && (float)cntred/points[i].size()>0.3 && (float)cntgreen/points[i].size()<0.1)
			{
				//	imshow("imgtempmser",imgTempMser);
				//waitKey(0);

				double timeR_s = (double)getTickCount();

				resize(imgTempMser, imgTempMser, Size(30, 30));
				double timeR1_e = (double)getTickCount();
				int bayes = BayesianClassifier(bayesian, imgTempMser);
				double timeR2_e = (double)getTickCount();
				//printf( "resize : %f ms.\n", (timeR1_e-timeR_s)*1000./getTickFrequency() );
				//printf( "classify : %f ms.\n", (timeR2_e-timeR_s)*1000./getTickFrequency() );
				//imshow("imgMserSegment",imgTempMser);
				//waitKey(1);

				if (bayes == 1 || bayes == 2)
				{
					/*cout << "rec pos X:" << rec.x << " rec pos Y:" << rec.y<<endl;*/
					rectangle(imgdrawTest, Point(rec.x, rec.y), Point(rec.x + rec.width, rec.y + rec.height), CV_RGB(255, 0, 0), 2);
					vecRect.push_back(rec);

				}

				//char fileName[50] ={0};
				//char num[50] ={0};
				//string inpath = "training\\mser";
				//strcpy(fileName,inpath.c_str());
				//sprintf(num,"R%d_%d.png",cntmser,i);

				//strncat(fileName,num,strlen(num));

				//imwrite(fileName,imgTempMser);
				//
				//				Mat imghist;
				//HistVec(imgTempMser, imghist);




				//for ( int j = 0; j < (int)r.size(); j++ )
				//{
				//	Point pt = r[j];
				//	imgdrawTest.at<Vec3b>(pt) = bcolors[i%9];

				//}

			}
		}
		////////////////////////////////
		if (bBlue == 1){
			//rectangle(imgdrawTest,Point(rec.x,rec.y), Point(rec.x+rec.width, rec.y+rec.height),CV_RGB(155,0,255),1);
			int count2 = 0;
			for (int j = 0; j<points[i].size(); j++)
			if (CM.imgBlueMask.at<uchar>(points[i][j]) != 0)
				count2++;
			//if((float)rec.width/(float)rec.height<1.4 && (float)rec.width/(float)rec.height>0.6 && rec.area()<3000&& (float)count2/(float)rec.area()>0.3 &&(float)count2/(float)rec.area()<0.8&& (float)cntgreen/points[i].size()<0.1)
			if ((float)rec.width / (float)rec.height<1.2 && (float)rec.width / (float)rec.height>0.5 && rec.area()<recSZ.maxRectAreaBLUE && rec.area()>recSZ.minRectAreaBLUE && (float)count2 / points[i].size()>0.7 && (float)count2 / points[i].size()<1 /*&& (float)cntgreen / points[i].size()<0.1*/ && (float)cntadapt / (float)points[i].size()>0.6)
			{
				double timeB_s = (double)getTickCount();
				resize(imgTempMser, imgTempMser, Size(30, 30), 0, 0, INTER_NEAREST);
				double timeB1_e = (double)getTickCount();
				int bayes = BayesianClassifier(bayesian, imgTempMser);
				double timeB2_e = (double)getTickCount();
				//imshow("imgMserSegment",imgTempMser);
				//waitKey(1);

				//printf( "resize : %f ms.\n", (timeB1_e-timeB_s)*1000./getTickFrequency() );
				//printf( "classify : %f ms.\n", (timeB2_e-timeB_s)*1000./getTickFrequency() );
				//cout << "resize : "<<timeB1_e-timeB_s << " classify : "<< timeB2_e-timeB_s<<endl;
				rectangle(imgdrawTest, Point(rec.x, rec.y), Point(rec.x + rec.width, rec.y + rec.height), CV_RGB(255, 0, 255), 1);
				if (bayes<3)
				{
					/*cout << "rec pos X:" << rec.x << " rec pos Y:" << rec.y<<endl;*/
					//rectangle(imgdrawTest, Point(rec.x, rec.y), Point(rec.x + rec.width, rec.y + rec.height), CV_RGB(255, 0, 255), 1);
					vecRect.push_back(rec);

				}





				//resize(imgTempMser,imgTempMser,Size(50,50));
				//imshow("imgMserSegment",imgTempMser);
				//waitKey(0);
				//Mat imghist;

				//resize(imgTempMser,imgTempMser,Size(30,30));
				//char fileName[50] ={0};
				//char num[50] ={0};
				//string inpath = "training\\mser";
				//strcpy(fileName,inpath.c_str());
				//sprintf(num,"B%d_%d.png",cntmser,i);

				//strncat(fileName,num,strlen(num));

				//imwrite(fileName,imgTempMser);
				////

				//HistVec(imgTempMser, imghist);


				//imshow("imgMserSegment",imgTempMser);
				//waitKey(1);
				//for ( int j = 0; j < (int)r.size(); j++ )
				//{
				//	Point pt = r[j];
				//	imgdrawTest.at<Vec3b>(pt) = bcolors[i%9];

				//}
			}
		}
		//if((float)rec.width/(float)rec.height<5 && (float)rec.width/(float)rec.height>2.5 && rec.area()<recSZ.maxRectAreaBLACK && rec.area()>recSZ.minRectAreaBLACK  )
		//if((float)rec.width/(float)rec.height<7 && (float)rec.width/(float)rec.height>2 )

		//if (bBlack == 1){
		//	int cntblack = 0;
		//	Mat imgBlack = CM.imgBlackMask(rec);
		//	for (int j = 0; j<imgBlack.cols*imgBlack.rows; j++)
		//		if (imgBlack.at<uchar>(j)>0)
		//			cntblack++;

		//	int cntlight = 0;
		//	Mat imgCromatic = AM.imgAcroMaskLight(rec);
		//	for (int j = 0; j<imgCromatic.cols*imgCromatic.rows; j++)
		//		if (imgCromatic.at<uchar>(j)>0)
		//			cntlight++;
		//	int cntGlight = 0;
		//	Mat imglight = CM.imgGreenLight(rec);
		//	for (int j = 0; j<imglight.cols*imglight.rows; j++)
		//		if (imglight.at<uchar>(j)>0)
		//			cntGlight++;

		//	if ((float)rec.width / (float)rec.height<5 && (float)rec.width / (float)rec.height>1.5 && rec.area()<recSZ.maxRectAreaBLACK && rec.area()>recSZ.minRectAreaBLACK && (float)cntblack / (float)rec.area()>0.2 && (float)cntadapt / (float)points[i].size()>0.8 && (float)cntlight / (float)rec.area()>0.001 && (float)cntlight / (float)rec.area()<0.5 && (float)cntGlight / (float)rec.area()>0.001)
		//	{
		//		rectangle(imgdrawTest, Point(rec.x, rec.y), Point(rec.x + rec.width, rec.y + rec.height), CV_RGB(255, 0, 255), 1);
		//		vecRect.push_back(rec);

		//		for (int j = 0; j < (int)r.size(); j++)
		//		{
		//			Point pt = r[j];
		//			imgdrawTest.at<Vec3b>(pt) = bcolors[i % 9];

		//		}
		//	}
		//}

	}
	//cout<<"rect num : "<<(int)points.size() <<"MserVec.vecImgMser num : "<<MserVec.vecImgMser.size()<<endl;

	t = (double)getTickCount() - t;
	//printf( "MSER extracted %d contours in %g ms.\n", (int)points.size(), t*1000./getTickFrequency() );
	//imshow("drawimg",imgdrawTest);
	//imshow("imgProb",imgProb);


}





void ColorMask(Color_t& CM, Mat& imgHue, Mat& imgRGB)
{
	vector<Mat> vecRGB;
	split(imgRGB, vecRGB);

	//cout << vecHSV[0]<<endl;
	for (int i = 0; i < imgHue.rows*imgHue.cols; i++)
	{
		if ((imgHue.at<uchar>(i)>0 && imgHue.at<uchar>(i)<20) || (imgHue.at<uchar>(i)>130 && imgHue.at<uchar>(i)<180) || (abs(vecRGB[0].at<uchar>(i)-vecRGB[1].at<uchar>(i))<5 && abs(vecRGB[1].at<uchar>(i)-vecRGB[2].at<uchar>(i))<5 && abs(vecRGB[2].at<uchar>(i)-vecRGB[0].at<uchar>(i))<5))
			CM.imgRedMask.at<uchar>(i) = 255;
		else
			CM.imgRedMask.at<uchar>(i) = 0;

		if (imgHue.at<uchar>(i)>100 && imgHue.at<uchar>(i)<125)
			CM.imgBlueMask.at<uchar>(i) = 255;
		else
			CM.imgBlueMask.at<uchar>(i) = 0;

		//if(imgHue.at<uchar>(i)>20 && imgHue.at<uchar>(i)<90 && (vecRGB[0].at<uchar>(i)<220 || vecRGB[1].at<uchar>(i)<220 || vecRGB[2].at<uchar>(i)<220))
		//if(imgHue.at<uchar>(i)>20 && imgHue.at<uchar>(i)<90 && (vecRGB[0].at<uchar>(i)<220 && vecRGB[1].at<uchar>(i)<230 && vecRGB[2].at<uchar>(i)<220))
		//	CM.imgGreenMask.at<uchar>(i)=255;
		//else
		//	CM.imgGreenMask.at<uchar>(i)=0;



		//int absVal = abs(vecRGB[0].at<uchar>(i)-vecRGB[1].at<uchar>(i));

		//if(imgHue.at<uchar>(i)>20 && imgHue.at<uchar>(i)<90 && (vecRGB[0].at<uchar>(i)<255 && vecRGB[1].at<uchar>(i)>100 && vecRGB[2].at<uchar>(i)<100) &&absVal<30)
		//	CM.imgGreenLight.at<uchar>(i)=255;
		//else
		//	CM.imgGreenLight.at<uchar>(i)=0;


		//if(vecRGB[0].at<uchar>(i)<120 && vecRGB[1].at<uchar>(i)<110 && vecRGB[2].at<uchar>(i)<90 && absVal <20)
		//	CM.imgBlackMask.at<uchar>(i)=255;
		//else
		//	CM.imgBlackMask.at<uchar>(i)=0;




	}

}
void clustering(vector<Rect>& vecCluster, int distance)
{

	for (int i = 0; i< vecCluster.size(); i++)
	{
		int cntSame = 0;
		for (int j = i + 1; j<vecCluster.size(); j++)
		{
			if (i == j)
				continue;
			int recX = abs(vecCluster[i].x - vecCluster[j].x);
			int recY = abs(vecCluster[i].y - vecCluster[j].y);
			int recW = abs(vecCluster[i].x + vecCluster[i].width - vecCluster[j].x - vecCluster[j].width);
			int recH = abs(vecCluster[i].y + vecCluster[i].height - vecCluster[j].y - vecCluster[j].height);


			if ((recX<distance && recY< distance && recW <distance && recH <distance) || (recX + recY + recW + recH)<distance * 3)
			{
				cntSame++;

				vecCluster[i].x = (int)(vecCluster[i].x*(cntSame + 1) + vecCluster[j].x) / (cntSame + 2);
				vecCluster[i].y = (int)(vecCluster[i].y*(cntSame + 1) + vecCluster[j].y) / (cntSame + 2);
				vecCluster[i].width = (int)(vecCluster[i].width*(cntSame + 1) + vecCluster[j].width) / (cntSame + 2);
				vecCluster[i].height = (int)(vecCluster[i].height*(cntSame + 1) + vecCluster[j].height) / (cntSame + 2);
				//cout <<"clustering : "<<i<<"-- "<<  vecRect[i].x<<" " <<vecRect[i].y <<" "<< vecRect[i].width <<" "<<vecRect[i].height<<endl;
				vecCluster.erase(vecCluster.begin() + j);
				j--;
			}
		}
		cntSame = 0;
	}
}


void enhancement(Mat imgR, Mat imgG, Mat imgB, Mat& imgDst, Mat& imgDst2)
{
	imgR.convertTo(imgR, CV_32FC1);
	imgG.convertTo(imgG, CV_32FC1);
	imgB.convertTo(imgB, CV_32FC1);
	Mat imgTempR(imgDst.size(), CV_32FC1);
	Mat imgTempB(imgDst.size(), CV_32FC1);

	for (int i = 0; i < imgDst.cols; i++)
	{
		for (int j = 0; j < imgDst.rows; j++)
		{
			imgTempR.at<float>(j, i) = (float)(255.0 * 2.0 * max(0.0f, (min(imgR.at<float>(j, i) - imgG.at<float>(j, i), imgR.at<float>(j, i) - imgB.at<float>(j, i)) / (imgR.at<float>(j, i) + imgG.at<float>(j, i) + imgB.at<float>(j, i)))));
			imgTempB.at<float>(j, i) = (float)(255.0 * 2.0 * max(0.0f, (max(imgB.at<float>(j, i) - imgG.at<float>(j, i), imgB.at<float>(j, i) - imgR.at<float>(j, i)) / (imgR.at<float>(j, i) + imgG.at<float>(j, i) + imgB.at<float>(j, i)))));

			//cout << imgTemp.at<float>(j, i)*255 << " ";
		}

	}

	imgTempR.convertTo(imgDst, CV_8UC1);
	imgTempB.convertTo(imgDst2, CV_8UC1);

	//waitKey(0);

}


