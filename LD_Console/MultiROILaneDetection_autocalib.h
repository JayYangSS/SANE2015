//master header
#pragma once
#include "highgui.h"
#include "cv.h"
#include "opencv2/opencv.hpp"
#include <vector>

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <iomanip>
#include <ctime>
#include <math.h>
#include "ranker.h"
using namespace cv;
using namespace std;
#define EMPTY -1
#define COMPARE_STANDARD 0.4
#define MULTIROINUMBER 12
#define MAXCOMP -99999
#define MINCOMP 99999
#define DEBUG_LINE 1
#define IPM_WIDTH_SCALE 1
#define IPM_HEIGHT_SCALE 1.5
#define DIST_TRACKING_WIDTH 200
#define MOVING_AVERAGE_NUM 7
#define TRACKING_FLAG_NUM 4
#define TRACKING_ERASE_LEVEL 1
#define TRACKINGERASE 5
#define MIN_WORLD_WIDTH 2.0		//Left,Right lane minimum interval
enum EROINUMBER{
	CENTER_ROI = 0,
	LEFT_ROI0,
	LEFT_ROI1,
	LEFT_ROI2,
	LEFT_ROI3,
	RIGHT_ROI0,
	RIGHT_ROI1,
	RIGHT_ROI2,
	RIGHT_ROI3,
	RIGHT_ROI4,
	AUTOCALIB,
	GROUND,
	KALMAN_LEFT,
	KALMAN_RIGHT
};
typedef enum LineType_ {
	LINE_HORIZONTAL = 0,
	LINE_VERTICAL = 1
} LineType;

typedef struct SFileInformation{
	char szDataDir[200];
	char szDataName[200];
	//int nTotalFrame;

}SFileInformation;

typedef struct SCameraInformation{
	Size_<double> sizeFocalLength;
	Point_<double> ptOpticalCenter;
	Size sizeCameraImage;

	double fHeight;
	double fPitch;
	double fYaw;
	Point_<float> ptVanishingPoint;
	float fGroundTop;
	float fGroundBottom;

}SCameraInformation;

typedef struct SConfiguration{
	float fLineWidth;
	float fLineHeight;
	int nRansacIteration;
	float nRansacThreshold;
	float fVanishPortion;
	float fLowerQuantile;
	int nLocalMaxIgnore;

}SConfiguration;

typedef struct SRoiInformation{
	Size sizeRoi;
	Point ptRoi;
	Point ptRoiEnd;
	int nLeft;
	int nRight;
	int nTop;
	int nBottom;
	Size sizeIPM;

	double dXLimit[2];
	double dYLimit[2];
	double dXScale;
	double dYScale;
	int nIpm2WorldWidth;
	int nIpm2WorldHeight;

	

	//detection information
	int nDetectionThreshold;
	int nGetEndPoint;
	int nGroupThreshold;
	float fOverlapThreshold;
	//RansacLine
	int nRansacNumSamples;
	int nRansacNumIterations;
	int nRansacNumGoodFit;
	float fRansacThreshold;
	int nRansacScoreThreshold;
	int nRansacLineWindow;


}SRoiInformation;

typedef struct SLine{
	//start pt world
	Point_<double> ptStartLine;
	//end pt world
	Point_<double> ptEndLine;
	//start pt uv
	Point_<double> ptUvStartLine;
	//end pt world
	Point_<double> ptUvEndLine;
	////score
	//float fScore;
	float fXcenter;
	float fXderiv;
	float fGroundHeight;
	//r & theta

	float fR;
	float fTheta;
}SLine;

typedef struct SWorldLane{
	float fXcenter;
	float fXderiv;
	float fYtop;
	float fYBottom;
	Point_<double> ptStartLane;
	Point_<double> ptEndLane;
	//start pt uv
	Point_<double> ptUvStartLine;
	//end pt world
	Point_<double> ptUvEndLine;
}SWorldLane;
typedef struct SKalman
{
	KalmanFilter KF;
	Mat_<float> matState;
	Mat matProcessNoise;
	Mat_<float> matMeasurement;
	SLine SKalmanTrackingLine;
	SLine SKalmanTrackingLineBefore;
	int cntNum;
	int cntErase;
	SKalman()
	{
		KF = KalmanFilter(6, 4, 0);
		matState = Mat_<float>(6, 1);
		matMeasurement = Mat_<float>(4, 1);
		cntNum = 0;
		cntErase = 0;
	}
}SKalman;

typedef struct SEvaluation
{
	int LeftTP;
	int LeftFP;
	int LeftFN;
	int LeftTN;

	int RightTP;
	int RightFP;
	int RightFN;
	int RightTN;

	int nLeftGroundTruth;
	int nRightGroundTruth;
	int nTotalFrame;
	SEvaluation()
	{
		LeftTP = 0;
		LeftFP = 0;
		LeftFN = 0;
		LeftTN = 0;

		RightTP = 0;
		RightFP = 0;
		RightFN = 0;
		RightTN = 0;

		nLeftGroundTruth = 0;
		nRightGroundTruth = 0;
		nTotalFrame = 0;
	}
}SEvaluation;

class CMultiROILaneDetection{
public:

	CMultiROILaneDetection();
	Mat m_imgOrigin;


	Mat m_imgResizeOrigin;

	Mat m_imgOriginScale;
	Mat m_imgResizeScaleGray;


	SFileInformation m_sPreScanDB;
	unsigned int m_nFrameNum;

	SCameraInformation m_sCameraInfo;
	SConfiguration m_sConfig;


	SRoiInformation m_sRoiInfo[MULTIROINUMBER];
	Mat m_matXYGrid[MULTIROINUMBER];
	Mat m_matUVGrid[MULTIROINUMBER];
	Mat m_imgIPM[MULTIROINUMBER];
	Mat m_ipmFiltered[MULTIROINUMBER];
	Mat m_filteredThreshold[MULTIROINUMBER];
	
	vector<SLine> m_lanes[MULTIROINUMBER];
	vector<float> m_laneScore[MULTIROINUMBER];
	vector<SLine> m_lanesResult[MULTIROINUMBER];
	vector<SLine> m_lanesGroundResult[MULTIROINUMBER];

	bool m_bTracking[MULTIROINUMBER];
	vector<SLine> m_leftTracking;
	vector<SLine>::iterator m_iterLeft;
	vector<SLine> m_rightTracking;
	vector<SLine>::iterator m_iterRight;

	vector<SLine> m_leftGroundTracking;
	vector<SLine>::iterator m_iterGroundLeft;
	vector<SLine> m_rightGroundTracking;
	vector<SLine>::iterator m_iterGroundRight;

	SLine m_sImgCenter;
	SWorldLane m_sWorldCenterInit;
	SWorldLane m_sLeftTrakingLane;
	SWorldLane m_sRightTrakingLane;

	SKalman m_SKalmanLeftLane;
	SKalman m_SKalmanRightLane;
	//KalmanFilter m_kalmanRightLane;

	int nLeftCnt;
	int nRightCnt;

	bool m_bLeftDraw;
	bool m_bRightDraw;
private:
	SVD m_SvdCalc;
	Mat m_MatFx;
	Mat m_MatFy;


public:
	void SetRoiIpmCofig(EROINUMBER nFlag);

	void StartLanedetection(EROINUMBER nFlag){
		GetIPM(nFlag);
		FilterLinesIPM(nFlag); //input = m_imgIPM, Output1= m_ipmFiltered, Output2= m_ipmFilteredThreshold
		GetLinesIPM(nFlag);
		LineFitting(nFlag);
		IPM2ImLines(nFlag);
	}
	
	void InitialResizeFunction(Size sizeResize);
	void GetIPM(EROINUMBER nFlag);
	void FilterLinesIPM(EROINUMBER nFlag);
	void GetLinesIPM(EROINUMBER nFlag);
	void LineFitting(EROINUMBER nFlag);
	void IPM2ImLines(EROINUMBER nFlag);

	//////////////////////////////////////////////////////////////////////////
	//Auto Calibration
	void PushBackResult(EROINUMBER nFlag,Vector<Mat> &vecMat){
		vecMat.push_back(m_ipmFiltered[nFlag].clone());
	}
	void GetCameraPose(EROINUMBER nFlag, Vector<Mat> &vecMat);
	void TransformImage2Ground(const Mat &matInPoints, Mat &matOutPoints);
	void TransformGround2Image(const Mat &matInPoints, Mat &matOutPoints);
	void ClearResultVector(EROINUMBER nFlag);
	Point TransformPointImage2Ground(Point ptIn);
	Point TransformPointGround2Image(Point ptIn);
	void KalmanTrackingStage(EROINUMBER nflag);
	void KalmanSetting(SKalman &SKalmanInput, EROINUMBER nflag);
	void TrackingStageGround(EROINUMBER nflag);
	void ClearDetectionResult();
private:
	void SetVanishingPoint();


	//void TransformImage2Ground(const Mat &matInPoints,Mat &matOutPoints);
	//void TransformGround2Image(const Mat &matInPoints,Mat &matOutPoints);
	void GetVectorMax(const Mat &matInVector, double &dMax, int &nMaxLoc, int nIgnore);
	double GetLocalMaxSubPixel(double dVal1, double dVal2, double dVal3);
	//mFunc
	void GetMaxLineScore(EROINUMBER nFlag);
	void GetTrackingLineCandidate(EROINUMBER nFlag);
	
	void GetMaxLineScoreTwo(EROINUMBER nFlag);
	//end mFunc
	/*void PointImIPM2World(EROINUMBER nFlag){
		for(unsigned int i=0; i<m_lanes[nFlag].size(); i++){
			m_lanesResult[nFlag].at(i).ptStartLine.x = 
				m_lanesResult[nFlag].at(i).ptStartLine.x
				/m_sRoiInfo[nFlag].dXScale
				+m_sRoiInfo[nFlag].dXLimit[0];

		}
	}*/
	void Lines2Mat(const vector<SLine> &lines, Mat &mat);
	void Mat2Lines(const Mat &mat, vector<SLine> &lines);
	void GroupLines(vector<SLine> &lines, vector<float> &lineScores,
		float groupThreshold, Size_<float> bbox);
	void LineXY2RTheta(const SLine &line, float &r, float &theta);
	void IntersectLineRThetaWithBB(float r, float theta, const Size_<float> bbox, SLine *outLine);
	bool IsPointInside(Point2d point, Size_<int> bbox);
	bool IsPointInside(Point2d point, Size_<float> bbox);
	void GetLinesBoundingBoxes(const vector<SLine> &lines, LineType type,
		Size_<int> size, vector<Rect> &boxes);
	void GroupBoundingBoxes(vector<Rect> &boxes, LineType type,
		float groupThreshold);
	void  SetMat(Mat& imgInMat,Rect_<int> RectMask, double val);
	void FitRansacLine(const Mat& matImage, int numSamples, int numIterations,
		float threshold, float scoreThreshold, int numGoodFit,
		bool getEndPoints, LineType lineType,
		SLine *lineXY, float *lineRTheta, float *lineScore,EROINUMBER nFlag);
	bool GetNonZeroPoints(const Mat& matInMat, Mat& matOutMat,bool floatMat);

	void CumSum(const Mat &inMat, Mat &outMat);

	void SampleWeighted(const Mat &cumSum, int numSamples, Mat &randInd, RNG &rng);

	void FitRobustLine(const Mat &matPoints, float *lineRTheta, float *lineAbc);
	
};
//other custom function
void SetFrameName(char* szDataName, char* szDataDir,int nFrameNum);
void ScaleMat(const Mat &inMat, Mat &outMat);
void ShowImageNormalize( const char str[],const Mat &pmat);
void ShowResults(CMultiROILaneDetection &obj, EROINUMBER nflag);
void SetFrameNameBMP(char* szDataName, char* szDataDir,int nFrameNum);


CMultiROILaneDetection::CMultiROILaneDetection(){
	cout << "contructor start" << endl;
	/*Mat matFx;
	Mat matFy;*/
	//create the convoultion kernel
	//Filtering LUT
	int derivLen = 33; //23; 13; 33;
	int smoothLen = 9; //9; 17;

	//this is for 5-pixels wide
	float derivp[] = {
		1.000000e-16, 1.280000e-14, 7.696000e-13, 2.886400e-11, 7.562360e-10,
		1.468714e-08, 2.189405e-07, 2.558828e-06, 2.374101e-05, 1.759328e-04,
		1.042202e-03, 4.915650e-03,
		1.829620e-02, 5.297748e-02,
		1.169560e-01, 1.918578e-01,
		2.275044e-01,
		1.918578e-01, 1.169560e-01,
		5.297748e-02, 1.829620e-02,
		4.915650e-03, 1.042202e-03,
		1.759328e-04, 2.374101e-05, 2.558828e-06, 2.189405e-07, 1.468714e-08,
		7.562360e-10, 2.886400e-11, 7.696000e-13, 1.280000e-14, 1.000000e-16
	};
	
	float smoothp[] = {
		-1.000000e-03,
		-2.200000e-02,
		-1.480000e-01,
		-1.940000e-01,
		7.300000e-01,
		-1.940000e-01,
		-1.480000e-01,
		-2.200000e-02,
		-1.000000e-03
	};
	m_MatFx = Mat(derivLen, 1, CV_32FC1, derivp).clone();
	m_MatFy = Mat(1, smoothLen, CV_32FC1, smoothp).clone();
}

void CMultiROILaneDetection::SetVanishingPoint(){
	//get vanishing point in world coordinates
	float fArrVp[] = {
		sin(m_sCameraInfo.fYaw)/cos(m_sCameraInfo.fPitch),
		cos(m_sCameraInfo.fYaw)/cos(m_sCameraInfo.fPitch),
		0
	};
	Mat matVp = Mat(3,1,CV_32FC1,fArrVp);
	//cout<<"mat vp "<<matVp<<endl;
	//Yaw rotation matrix
	float fArrTyaw[] = {
		cos(m_sCameraInfo.fYaw), -sin(m_sCameraInfo.fYaw), 0,
		sin(m_sCameraInfo.fYaw), cos(m_sCameraInfo.fYaw), 0,
		0,						0,					1
	};
	Mat matTyaw = Mat(3,3, CV_32FC1, fArrTyaw);

	//Pitch rotation matrix
	float fArrPitchp[] = {
		1,			0,			0,
		0,-sin(m_sCameraInfo.fPitch),-cos(m_sCameraInfo.fPitch),
		0,cos(m_sCameraInfo.fPitch), -sin(m_sCameraInfo.fPitch)
	};
	Mat matTpitch = Mat(3,3,CV_32FC1,fArrPitchp);
	//combine transform matrix
	Mat matTransform = matTpitch * matTyaw;
	//cout<<matTransform<<endl;
	//
	//transformation from (xc, yc) in camera coordinates
	// to (u,v) in image frame
	//
	//matrix to shift optical center and focal length
	float fArrCamerap[]={
		m_sCameraInfo.sizeFocalLength.width,0,m_sCameraInfo.ptOpticalCenter.x,
		0,m_sCameraInfo.sizeFocalLength.height,m_sCameraInfo.ptOpticalCenter.y,
		0,0,1
	};
	Mat matCameraTransform = Mat(3,3,CV_32FC1, fArrCamerap);
	//combine transform
	matTransform = matCameraTransform * matTransform;
	matVp = matTransform * matVp;
	float *pVP = (float*)&matVp.data[0];
	m_sCameraInfo.ptVanishingPoint.x = pVP[0];
	m_sCameraInfo.ptVanishingPoint.y = pVP[1];
	//cout<<"vanishing point "<<m_sCameraInfo.ptVanishingPoint<<endl;


}
//pointer need

/*LYW_0724*/
// (u,v) --> (x,y) 나오게끔. h는 사전에 미리 셋팅.
// Input : row의 개수는 1 이상만 넣어주면 됨. col : 반드시 2개.
// Input/Output matrix의 dim은 같아야함.
// 
void CMultiROILaneDetection::TransformImage2Ground(const Mat &matInPoints,Mat &matOutPoints){
	//	cout<<*matInPoints<<endl;
	//	cout<<"확인1\n"<<endl;

	//add two rows to the input points

	Mat matInPoints4;
	matInPoints4.create(matInPoints.rows+2,matInPoints.cols,matInPoints.type());

	//copy inPoints to first two rows

	//call by reference
	Mat matInPoints2=matInPoints4.rowRange(0,2);
	Mat matInPoints3=matInPoints4.rowRange(0,3);	
	Mat matInPointsr3=matInPoints4.row(2);
	Mat matInPointsr4=matInPoints4.row(3);


	matInPointsr3.setTo(1);

	matInPoints.copyTo(matInPoints2);
	//	cout<<matInPoints4<<endl;
	//	cout<<"확인2\n"<<endl;

	//create the transformation matrix
	float fC1 = cos(m_sCameraInfo.fPitch);
	float fS1 = sin(m_sCameraInfo.fPitch);
	float fC2 = cos(m_sCameraInfo.fYaw);
	float fS2 = sin(m_sCameraInfo.fYaw);
	float fArrT[] = {
		-m_sCameraInfo.fHeight*fC2/m_sCameraInfo.sizeFocalLength.width,
		m_sCameraInfo.fHeight*fS1*fS2/m_sCameraInfo.sizeFocalLength.height,
		(m_sCameraInfo.fHeight*fC2*m_sCameraInfo.ptOpticalCenter.x/m_sCameraInfo.sizeFocalLength.width)-
		(m_sCameraInfo.fHeight *fS1*fS2* m_sCameraInfo.ptOpticalCenter.y/
		m_sCameraInfo.sizeFocalLength.height) - m_sCameraInfo.fHeight *fC1*fS2,

		m_sCameraInfo.fHeight *fS2 /m_sCameraInfo.sizeFocalLength.width,
		m_sCameraInfo.fHeight *fS1*fC2 /m_sCameraInfo.sizeFocalLength.height,
		(-m_sCameraInfo.fHeight *fS2* m_sCameraInfo.ptOpticalCenter.x
		/m_sCameraInfo.sizeFocalLength.width)-(m_sCameraInfo.fHeight *fS1*fC2*
		m_sCameraInfo.ptOpticalCenter.y /m_sCameraInfo.sizeFocalLength.height) -
		m_sCameraInfo.fHeight *fC1*fC2,

		0,
		m_sCameraInfo.fHeight *fC1 /m_sCameraInfo.sizeFocalLength.height,
		(-m_sCameraInfo.fHeight *fC1* m_sCameraInfo.ptOpticalCenter.y /m_sCameraInfo.sizeFocalLength.height) + m_sCameraInfo.fHeight *fS1,

		0,
		-fC1 /m_sCameraInfo.sizeFocalLength.height,
		(fC1* m_sCameraInfo.ptOpticalCenter.y /m_sCameraInfo.sizeFocalLength.height) - fS1,
	};// constant


	Mat matMat=Mat(4,3,CV_32FC1,fArrT);
	matInPoints4=matMat*matInPoints3;


	float *pMatInPoints4 = (float*)(&matInPoints4.data[0]);
	float *pMatInPointsr4 = (float*)(&matInPointsr4.data[0]);

	for (int i=0; i<matInPoints.cols; i++)
	{
		double div = pMatInPointsr4[matInPointsr4.cols*0 +i];		
		pMatInPoints4[matInPoints4.cols*0 +i]/=div;
		pMatInPoints4[matInPoints4.cols*1 +i]/=div;
	}


	//put back the result into outPoints
	matInPoints2.copyTo(matOutPoints);
	//cout<<matOutPoints<<endl<<endl;

}
void CMultiROILaneDetection::TransformGround2Image(const Mat &matInPoints,Mat &matOutPoints){
	//add two rows to the input points
	Mat matInPoints3(matInPoints.rows+1,matInPoints.cols,matInPoints.type()); //(X,Y,-H)

	Mat matInPoints2=matInPoints3.rowRange(0,2);
	Mat matInPointsr3=matInPoints3.row(2);

	matInPointsr3.setTo(-m_sCameraInfo.fHeight);
	matInPoints.copyTo(matInPoints2);

	//create the transformation matrix
	float c1 = cos(m_sCameraInfo.fPitch);
	float s1 = sin(m_sCameraInfo.fPitch);
	float c2 = cos(m_sCameraInfo.fYaw);
	float s2 = sin(m_sCameraInfo.fYaw);
	float matp[] = {
		m_sCameraInfo.sizeFocalLength.width * c2 + c1*s2* m_sCameraInfo.ptOpticalCenter.x,
		-m_sCameraInfo.sizeFocalLength.width * s2 + c1*c2* m_sCameraInfo.ptOpticalCenter.x,
		- s1 * m_sCameraInfo.ptOpticalCenter.x,

		s2 * (-m_sCameraInfo.sizeFocalLength.height * s1 + c1* m_sCameraInfo.ptOpticalCenter.y),
		c2 * (-m_sCameraInfo.sizeFocalLength.height * s1 + c1* m_sCameraInfo.ptOpticalCenter.y),
		-m_sCameraInfo.sizeFocalLength.height * c1 - s1* m_sCameraInfo.ptOpticalCenter.y,

		c1*s2,
		c1*c2,
		-s1
	};
	Mat matMat(3,3,CV_32FC1,matp);
	matInPoints3=matMat*matInPoints3;
	for (int i=0; i<matInPoints.cols; i++)
	{
		float div = matInPointsr3.at<float>(0,i);	
		matInPoints3.at<float>(0,i)=matInPoints3.at<float>(0,i)/div;
		matInPoints3.at<float>(1,i)=matInPoints3.at<float>(1,i)/div;
	}
	matInPoints2.copyTo(matOutPoints);
}

//[LYW_0724] : LUT를 만드는 함수
void CMultiROILaneDetection::SetRoiIpmCofig(EROINUMBER nFlag ){ 
	m_bTracking[nFlag] = false;
	SetVanishingPoint();
	Point_<float> ptVp = m_sCameraInfo.ptVanishingPoint;
	ptVp.y = MAX(0,ptVp.y);
	float fWidth = m_imgResizeOrigin.cols;
	float fHeight = m_imgResizeOrigin.rows;
	float fEps = m_sConfig.fVanishPortion * fHeight;


	//vanishing point validation
	m_sRoiInfo[nFlag].nLeft = MAX(0,m_sRoiInfo[nFlag].nLeft);
	m_sRoiInfo[nFlag].nRight = MIN(fWidth,m_sRoiInfo[nFlag].nRight);
	m_sRoiInfo[nFlag].nTop = MAX(ptVp.y+fEps, m_sRoiInfo[nFlag].nTop);
	m_sRoiInfo[nFlag].nBottom = MIN(fHeight-1,m_sRoiInfo[nFlag].nBottom);

	//ROI boundary limits
	float fArrLimits[] = {
		ptVp.x,					m_sRoiInfo[nFlag].nRight, m_sRoiInfo[nFlag].nLeft, ptVp.x,
		m_sRoiInfo[nFlag].nTop, m_sRoiInfo[nFlag].nTop,   m_sRoiInfo[nFlag].nTop,  m_sRoiInfo[nFlag].nBottom
	};

	Mat matUvLimits(2,4,CV_32FC1,fArrLimits);
	Mat matXyLimits(2,4,CV_32FC1);

	TransformImage2Ground(matUvLimits,matXyLimits);
	//	cout<<"matXyLimits \n"<<matXyLimits<<endl;
	double xfMax, xfMin, yfMax, yfMin;

	Mat matRow1=matXyLimits.row(0);
	Mat matRow2=matXyLimits.row(1);

	minMaxLoc(matRow1,(double*)&xfMin,(double*)&xfMax);
	minMaxLoc(matRow2,(double*)&yfMin,(double*)&yfMax);

	int outRow = m_sRoiInfo[nFlag].sizeIPM.height;
	int outCol = m_sRoiInfo[nFlag].sizeIPM.width;
	float stepRow = (yfMax-yfMin)/outRow;
	float stepCol = (xfMax-xfMin)/outCol;

	//construct the grid to sample

	Mat matXyGrid(2,outRow*outCol,CV_32FC1);
	//Grid is LUT
	float *pMatXyGrid= (float*)&matXyGrid.data[0];
	int i,j;
	float x,y;
	for ( i=0,  y=yfMax-.5*stepRow; i<outRow; i++, y-=stepRow)//delete .at() complete
	{
		for ( j=0,  x=xfMin+.5*stepCol; j<outCol; j++, x+=stepCol)
		{
			pMatXyGrid[matXyGrid.cols*0+i*outCol+j]=x;
			pMatXyGrid[matXyGrid.cols*1+i*outCol+j]=y;
		}
	}
	matXyGrid.copyTo(m_matXYGrid[nFlag]);
	Mat matUvGrid(2, outRow*outCol, CV_32FC1);

	float* pMatUvGrid = (float*)&matUvGrid.data[0];

	TransformGround2Image(matXyGrid, matUvGrid); 
	matUvGrid.copyTo(m_matUVGrid[nFlag]);

	m_imgIPM[nFlag].create(m_sRoiInfo[nFlag].sizeIPM,CV_32FC1);

	m_sRoiInfo[nFlag].dXLimit[0] = matXyGrid.at<float>(0,0);
	m_sRoiInfo[nFlag].dXLimit[1] = matXyGrid.at<float>(0, (outRow-1)*outCol+outCol-1);
	m_sRoiInfo[nFlag].dYLimit[1] = matXyGrid.at<float>(1,0);
	m_sRoiInfo[nFlag].dYLimit[0] = matXyGrid.at<float>(1, (outRow-1)*outCol+outCol-1);
	m_sRoiInfo[nFlag].dXScale = 1/stepCol;
	m_sRoiInfo[nFlag].dYScale = 1/stepRow;
	m_sRoiInfo[nFlag].nIpm2WorldHeight=m_sRoiInfo[nFlag].sizeIPM.height; //not used
	m_sRoiInfo[nFlag].nIpm2WorldWidth=m_sRoiInfo[nFlag].sizeIPM.width; //not used


	

}
void CMultiROILaneDetection::GetIPM( EROINUMBER nFlag){


	Scalar sMean=mean(m_imgResizeScaleGray);
	double dmean=sMean.val[0];
	//	cout<<"dmean\n";
	//	cout<<dmean<<endl;
	int i,j;
	float ui, vi;	

	float* ppMatOutImage= (float*)&m_imgIPM[nFlag].data[0];
	float* ppMatInImage = (float*)&m_imgResizeScaleGray.data[0];
	float* pMatUvGrid = (float*) &m_matUVGrid[nFlag].data[0];
	int nResizeImgWidth = m_imgResizeScaleGray.cols;
	int nResizeImgHeight = m_imgResizeScaleGray.rows;
	int nIpmWidth = m_sRoiInfo[nFlag].sizeIPM.width;
	int nIpmHeight = m_sRoiInfo[nFlag].sizeIPM.height;
	int nUvGridWidth = m_matUVGrid[nFlag].cols;
	int nUvGridHeight = m_matUVGrid[nFlag].rows;
	//IPM image make process
	for (i=0; i<nIpmHeight; i++)
		for (j=0; j<nIpmWidth; j++){ 
			/*get pixel coordiantes*/ 

			ui = pMatUvGrid[nUvGridWidth*0+i*nIpmWidth+j];
			vi = pMatUvGrid[nUvGridWidth*1+i*nIpmWidth+j];
			/*check if out-of-bounds*/ 
			if (ui<m_sRoiInfo[nFlag].nLeft || ui>m_sRoiInfo[nFlag].nRight || 
				vi<m_sRoiInfo[nFlag].nTop || vi>m_sRoiInfo[nFlag].nBottom) { 
					ppMatOutImage[nIpmWidth*i+j]=(float)dmean;
					//if()
			} 
			/*not out of bounds, then get nearest neighbor*/ 
			else 
			{ 
				/*Bilinear interpolation*/ 
				{ 
					int x1 = int(ui), x2 = int(ui+1); 
					int y1 = int(vi), y2 = int(vi+1); 
					float x = ui - x1, y = vi - y1;   

					float val = 
						ppMatInImage[x1+nResizeImgWidth*y1]*(1-x)*(1-y)+
						ppMatInImage[x2+nResizeImgWidth*y1]*(x)*(1-y) +
						ppMatInImage[x1+nResizeImgWidth*y2]*(1-x)*(y) +
						ppMatInImage[x2+nResizeImgWidth*y2]*(x)*(y);
					//		cout<<"x "<<x<<"y "<<y<<endl;
					//		cout<<val<<endl;
					ppMatOutImage[j+i*nIpmWidth] = val;
					//	cout<<val<<endl;
					//pMatOutImage->at<float>(i,j) = (float)val;
				} 
				/*nearest-neighbor interpolation*/ 
				/*else 
				{

				pMatOutImage->at<float>(i,j)=pMatInImage->at<float>(int(vi+.5),int(ui+.5));
				}*/
			} 
			/*if (outPoints && 
			(ui<ipmInfo->ipmLeft+10 || ui>ipmInfo->ipmRight-10 || 
			vi<ipmInfo->ipmTop || vi>ipmInfo->ipmBottom-2) )	{
			outPoints->push_back(cvPoint(j, i)); 
			}*/
		}

}

void CMultiROILaneDetection::InitialResizeFunction(Size sizeResize){
	resize(m_imgOrigin,m_imgResizeOrigin,sizeResize);
	m_imgResizeOrigin.convertTo(m_imgOriginScale,CV_32FC1,1.0/255);
	cvtColor(m_imgOriginScale,m_imgResizeScaleGray,CV_RGB2GRAY);
}
void CMultiROILaneDetection::FilterLinesIPM(EROINUMBER nFlag){
	//define the two kernels

	//Mat matFx;
	//Mat matFy;
	////create the convoultion kernel

	//int derivLen = 33; //23; 13; 33;
	//int smoothLen = 9; //9; 17;

	////this is for 5-pixels wide
	//float derivp[] = {
	//	1.000000e-16, 1.280000e-14, 7.696000e-13, 2.886400e-11, 7.562360e-10, 
	//	1.468714e-08, 2.189405e-07, 2.558828e-06, 2.374101e-05, 1.759328e-04, 
	//	1.042202e-03, 4.915650e-03, 
	//	1.829620e-02, 5.297748e-02, 
	//	1.169560e-01, 1.918578e-01, 
	//	2.275044e-01, 
	//	1.918578e-01, 1.169560e-01, 
	//	5.297748e-02, 1.829620e-02, 
	//	4.915650e-03, 1.042202e-03, 
	//	1.759328e-04, 2.374101e-05, 2.558828e-06, 2.189405e-07, 	1.468714e-08, 
	//	7.562360e-10, 2.886400e-11, 7.696000e-13, 1.280000e-14, 1.000000e-16
	//};

	//float smoothp[] = {
	//	-1.000000e-03,
	//	-2.200000e-02,
	//	-1.480000e-01, 
	//	-1.940000e-01, 
	//	7.300000e-01, 
	//	-1.940000e-01, 
	//	-1.480000e-01,
	//	-2.200000e-02,
	//	-1.000000e-03
	//};
	//matFx=Mat(derivLen,1,CV_32FC1,derivp);
	//matFy=Mat(1,smoothLen,CV_32FC1,smoothp);
	//if ((nFlag == LEFT_ROI3) || (nFlag == RIGHT_ROI3))
	//	resize(matFx, matFx, Size(1, matFx.rows * 2));

	Scalar dMean =mean(m_imgIPM[nFlag]);

	subtract(m_imgIPM[nFlag],dMean,m_ipmFiltered[nFlag]);

	filter2D(m_ipmFiltered[nFlag],m_ipmFiltered[nFlag],m_ipmFiltered[nFlag].depth(),
		m_MatFx, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(m_ipmFiltered[nFlag],m_ipmFiltered[nFlag],m_ipmFiltered[nFlag].depth(),
		m_MatFy, Point(-1, -1), 0.0, BORDER_REPLICATE);
	//double dStartTick = (double)getTickCount();
	Mat rowMat;
	rowMat = Mat(m_ipmFiltered[nFlag]).reshape(0,1); //1row로 누적시킴
	//get the quantile
	float fQval;
	fQval = quantile((float*) &rowMat.data[0], rowMat.cols, m_sConfig.fLowerQuantile);		//Quantile 97%
																							//필터링 결과 영상에서 threshold value를 Quantile함수를 이용하여 결정
	threshold(m_ipmFiltered[nFlag],m_filteredThreshold[nFlag],fQval,NULL,THRESH_TOZERO);	//Threshold 미만 value를 zero로, 나머지 그대로
	//ThresholdLower(imgSubImage,imgSubImage, fQtileThreshold);
	//double dEndTick = (double)getTickCount();
	//cout<<"reshape & quantile & threshold time  "<<(dEndTick-dStartTick) / getTickFrequency()*1000.0<<" msec"<<endl;

}
void CMultiROILaneDetection::GetLinesIPM(EROINUMBER nFlag){
	Mat matImage;
	matImage=m_filteredThreshold[nFlag].clone();

	//get sum of lines through horizontal or vertical
	//sumLines is a column vector

	Mat matSumLines, matSumLinesp;

	int maxLineLoc = 0;

	matSumLinesp.create(1,matImage.cols,CV_32FC1);
	reduce(matImage,matSumLinesp,0,CV_REDUCE_SUM); //reshape비슷한데, 1-row압축 reshape와 차이는 몰라
	matSumLines = Mat(matSumLinesp).reshape(0,matImage.cols);

	//max location for a detected line
	maxLineLoc = matImage.cols-1;//width-1;

	int smoothWidth = 21;
	float smoothp[] =	{
		0.000003726653172, 0.000040065297393, 0.000335462627903, 0.002187491118183,
		0.011108996538242, 0.043936933623407, 0.135335283236613, 0.324652467358350,
		0.606530659712633, 0.882496902584595, 1.000000000000000, 0.882496902584595,
		0.606530659712633, 0.324652467358350, 0.135335283236613, 0.043936933623407,
		0.011108996538242, 0.002187491118183, 0.000335462627903, 0.000040065297393,
		0.000003726653172
	};

	Mat matSmooth = Mat(1,smoothWidth,CV_32FC1,smoothp);
	filter2D(matSumLines,matSumLines,CV_32FC1,matSmooth,Point(-1,-1),0.0,1);

	//get the max and its location
	vector <int> sumLinesMaxLoc;
	vector <double> sumLinesMax;
	int nMaxLoc;
	double nMax;
	//필터 결과 영상을 누적시키고 노이즈 제거 이후 후보군에 대해서 누적값의 순위를 결정하여 벡터라이즈화
	GetVectorMax(matSumLines, nMax, nMaxLoc, m_sConfig.nLocalMaxIgnore);

	float *pfMatSumLinesData = (float*)matSumLines.data;
	//pfMatSumLinesData[matSumLines.cols*j+i];
	//loop to get local maxima
	for(int i=1+ m_sConfig.nLocalMaxIgnore; i<matSumLines.rows-1- m_sConfig.nLocalMaxIgnore; i++){
		//get that value
		//	FLOAT val = CV_MAT_ELEM(sumLines, FLOAT_MAT_ELEM_TYPE, i, 0);
		float val = pfMatSumLinesData[matSumLines.cols*i+0];
		//check if local maximum
		if( (val >pfMatSumLinesData[matSumLines.cols*(i-1)+0])
			&& (val > pfMatSumLinesData[matSumLines.cols*(i+1)+0])
			//		&& (i != maxLoc)
			&& (val >= m_sRoiInfo[nFlag].nDetectionThreshold) ){
				//iterators for the two vectors
				vector<double>::iterator j;
				vector<int>::iterator k;
				//loop till we find the place to put it in descendingly
				for(j=sumLinesMax.begin(),k=sumLinesMaxLoc.begin(); j != sumLinesMax.end()  && val<= *j; j++,k++);
				//add its index
				sumLinesMax.insert(j, val);
				sumLinesMaxLoc.insert(k, i);
		}
	}

	//check if didnt find local maxima
	if(sumLinesMax.size()==0 && nMax>m_sRoiInfo[nFlag].nDetectionThreshold){
		//put maximum
		sumLinesMaxLoc.push_back(nMaxLoc);
		sumLinesMax.push_back(nMax);
	}

	//     //sort it descendingly

	//plot the line scores and the local maxima

	//process the found maxima
	pfMatSumLinesData = (float*)matSumLines.data;
	for (int i=0; i<(int)sumLinesMax.size(); i++){
		//get subpixel accuracy
		double maxLocAcc = GetLocalMaxSubPixel(
			(double)pfMatSumLinesData[matSumLines.cols*MAX(sumLinesMaxLoc[i]-1,0)+0],
			(double)pfMatSumLinesData[matSumLines.cols*sumLinesMaxLoc[i]+0],
			(double)pfMatSumLinesData[matSumLines.cols*MIN(sumLinesMaxLoc[i]+1,maxLineLoc)+0]
		);
		maxLocAcc += sumLinesMaxLoc[i];
		maxLocAcc = MIN(MAX(0, maxLocAcc), maxLineLoc);
		//TODO: get line extent
		//put the extracted line
		SLine line;
		line.ptStartLine.x = (double)maxLocAcc + 0.5;//sumLinesMaxLoc[i]+.5;
		line.ptStartLine.y = 0.5;
		line.ptEndLine.x = line.ptStartLine.x;
		line.ptEndLine.y = m_filteredThreshold[nFlag].rows-0.5;//inImage->height-.5;

		(m_lanes[nFlag]).push_back(line);
		//		if (lineScores)
		(m_laneScore[nFlag]).push_back(sumLinesMax[i]);
	}
	//for
	//cout<<"nFlag"<<nFlag<<endl;
	//for(int i=0;i<m_laneScore[nFlag].size();i++)
	//	cout<<m_laneScore[nFlag].at(i)<<","<<endl;
	if((nFlag!=CENTER_ROI) && (nFlag !=AUTOCALIB) && (m_lanes[nFlag].size()>0)){
		if (m_bTracking[nFlag])
			GetTrackingLineCandidate(nFlag);//이전에 추적한 결과가 있을 경우, 검출 대상 차선의 위치가 월드좌표상으로 멀리 떨어진 값인지 확인
		else
			GetMaxLineScore(nFlag);
	}else if(nFlag==AUTOCALIB){
		//printf("serch all\n");
	}
	//cout<<"nFlag"<<nFlag<<endl;
	//for(int i=0;i<m_laneScore[nFlag].size();i++)
	//	cout<<m_laneScore[nFlag].at(i)<<","<<endl;

	//clean
	sumLinesMax.clear();
	sumLinesMaxLoc.clear();
}
void CMultiROILaneDetection::LineFitting(EROINUMBER nFlag){
	vector<SLine> lines = m_lanes[nFlag]; // [LYW_0724] : line fitting해야할 후보들, m_laneScore와 짝을 이룬다. (같은 index에 score가 저장되어 있음)
	vector<float> lineScores = m_laneScore[nFlag];
	/*cout<<m_lanes[nFlag].at(1).ptStartLine<<endl;
	cout<<lines.at(1).ptStartLine<<endl;
	cout<<m_laneScore[nFlag].at(1)<<endl;
	cout<<lineScores.at(1)<<endl;*/
	int width = m_sRoiInfo[nFlag].sizeIPM.width-1;
	int height = m_sRoiInfo[nFlag].sizeIPM.height-1;

	GroupLines(lines,lineScores,m_sRoiInfo[nFlag].nGroupThreshold,Size_<float>((float)width, (float)height));
	float overlapThreshold = m_sRoiInfo[nFlag].fOverlapThreshold; //0.5; //.8;
	
	vector<Rect> vecRectBoxes;

	GetLinesBoundingBoxes(lines, LINE_VERTICAL, Size_<int>(width, height),vecRectBoxes);
	GroupBoundingBoxes(vecRectBoxes, LINE_VERTICAL, overlapThreshold);

	int window = m_sRoiInfo[nFlag].nRansacLineWindow; //15;
	vector<SLine> newLines;
	vector<float> newScores;
	for (int i=0; i<(int)vecRectBoxes.size(); i++) //lines
	{
		// 	fprintf(stderr, "i=%d\n", i);
		//Line line = lines[i];
		//CvRect mask, box;
		Rect rectMask,rectBox;
		//get box
		//box = boxes[i];
		rectBox = vecRectBoxes[i];

		//get extent of window to search in
		//int xstart = (int)fmax(fmin(line.startPoint.x, line.endPoint.x)-window, 0);
		//int xend = (int)fmin(fmax(line.startPoint.x, line.endPoint.x)+window, width-1);
		int xstart = (int)max(rectBox.x - window, 0);
		int xend = (int)min(rectBox.x + rectBox.width + window, width-1);
		//get the mask
		//mask = cvRect(xstart, 0, xend-xstart+1, height);
		rectMask = Rect(xstart, 0, xend-xstart+1, height);

		//get the subimage to work on

		Mat matSubImage = m_filteredThreshold[nFlag].clone();
		//clear all but the mask

		SetMat(matSubImage,rectMask, 0);

		float lineRTheta[2]={-1,0};
		float lineScore;
		SLine line;
		//RANSAC 결과

		FitRansacLine(matSubImage, m_sRoiInfo[nFlag].nRansacNumSamples,
			m_sRoiInfo[nFlag].nRansacNumIterations,
			m_sRoiInfo[nFlag].fRansacThreshold,
			m_sRoiInfo[nFlag].nRansacScoreThreshold,
			m_sRoiInfo[nFlag].nRansacNumGoodFit,
			m_sRoiInfo[nFlag].nGetEndPoint, LINE_VERTICAL,
			&line, lineRTheta, &lineScore,nFlag);

		//store the line if found and make sure it's not
		//near horizontal or vertical (depending on type)
		//  #warning "check this screening in ransacLines"
		if (lineRTheta[0]>=0){
			bool put =true;
			//make sure it's not horizontal
			if((fabs(lineRTheta[1]) > 20*CV_PI/180))
				put = false;
			
			//IPM ROI border line rejection
			int nBorderX = (line.ptStartLine.x + line.ptEndLine.x) / 2; 
			int nBorderGap = m_sRoiInfo[nFlag].sizeIPM.width / 10;
			if (nBorderX<(nBorderGap) || nBorderX>(m_sRoiInfo[nFlag].sizeIPM.width - nBorderGap))
				put = false;

			if (put){
				newLines.push_back(line);
				newScores.push_back(lineScore);
			}
		} // if
		//clear
	} // for i

	lines.clear();
	lineScores.clear();
	//#warning "not grouping at end of getRansacLines"
	//lines = newLines;
	//lineScores = newScores;
	
	m_lanes[nFlag].clear();
	m_laneScore[nFlag].clear();
	
	if(newScores.size()>2&&nFlag==AUTOCALIB){ // [LYW_0724] : 라인2개찾았는지 검사
		//int nFirst,nSecond;
		int nFirstIdx,nSecondIdx;
		if(newScores[0]>newScores[1]){
			nFirstIdx=0,nSecondIdx=1;
			//nFirst=newScores[0],nSecond=newScores[1];
		}else{
			nFirstIdx=1,nSecondIdx=0;
			//nFirst=newScores[1],nSecond=newScores[0];
		}
		for(int i=2;i<newScores.size();i++)
		{
			if(newScores[i]>newScores[nFirstIdx]){
				nSecondIdx = nFirstIdx;
				nFirstIdx = i;
			}else if(newScores[i]>newScores[nSecondIdx]){
				nSecondIdx=i;
			}
		}
		m_lanes[nFlag].push_back(newLines[nFirstIdx]);
		m_laneScore[nFlag].push_back(newScores[nFirstIdx]);
		m_lanes[nFlag].push_back(newLines[nSecondIdx]);
		m_laneScore[nFlag].push_back(newScores[nSecondIdx]);
	}else{ // [LYW_0724] : Auto Calib가 아닌 일반적인 detection단계에서 찾은 차선을 다 넣어주는거야! 이 부분 때문에 다차선검출이 가능할 수 있어!!
		m_lanes[nFlag] = newLines;  
		m_laneScore[nFlag] = newScores;
	}
 
	

	//clean
	//	boxes.clear();
	newLines.clear();
	newScores.clear();
}
void CMultiROILaneDetection::IPM2ImLines(EROINUMBER nFlag){ // 

	if(m_lanes[nFlag].size()!=0){
		//PointImIPM2World
		//m_lanes[nFlag].
		//IPM2WORLD
		for(int i=0; i<m_lanes[nFlag].size();i++){
			m_lanesResult[nFlag].push_back(m_lanes[nFlag].at(i));
			//cout<<m_lanes[nFlag].at(i).ptStartLine<<endl;
			//cout<<m_lanesResult[nFlag].at(i).ptStartLine<<endl;

			//x-direction
			m_lanesResult[nFlag].at(i).ptStartLine.x = m_lanes[nFlag].at(i).ptStartLine.x/(m_sRoiInfo[nFlag].dXScale);
			m_lanesResult[nFlag].at(i).ptStartLine.x += m_sRoiInfo[nFlag].dXLimit[0];
			//y-direction
			m_lanesResult[nFlag].at(i).ptStartLine.y = m_lanes[nFlag].at(i).ptStartLine.y/(m_sRoiInfo[nFlag].dYScale);
			m_lanesResult[nFlag].at(i).ptStartLine.y =  m_sRoiInfo[nFlag].dYLimit[1 ] - m_lanesResult[nFlag].at(i).ptStartLine.y;

			//x-direction
			m_lanesResult[nFlag].at(i).ptEndLine.x = m_lanes[nFlag].at(i).ptEndLine.x/m_sRoiInfo[nFlag].dXScale;
			m_lanesResult[nFlag].at(i).ptEndLine.x += m_sRoiInfo[nFlag].dXLimit[0];
			//y-direction
			m_lanesResult[nFlag].at(i).ptEndLine.y = m_lanes[nFlag].at(i).ptEndLine.y/m_sRoiInfo[nFlag].dYScale;
			m_lanesResult[nFlag].at(i).ptEndLine.y = m_sRoiInfo[nFlag].dYLimit[1] - m_lanesResult[nFlag].at(i).ptEndLine.y;
			//record ground location
			m_lanesGroundResult[nFlag].push_back(m_lanesResult[nFlag].at(i));
		}
		//WORLD2IMAGE
		//convert them from world frame into camera frame
		//
		//put a dummy line at the beginning till we check that cvDiv bug
		if ((nFlag != LEFT_ROI2) && (nFlag != LEFT_ROI3)){
			SLine dummy;
			dummy.ptStartLine.x = 1.0;
			dummy.ptStartLine.y = 1.0;
			dummy.ptEndLine.x = 2.0;
			dummy.ptEndLine.y = 2.0;
			m_lanesResult[nFlag].insert(m_lanesResult[nFlag].begin(), dummy);
			//convert to mat and get in image coordinates
			Mat matMat = Mat(2, 2 * m_lanesResult[nFlag].size(), CV_32FC1);
			Lines2Mat(m_lanesResult[nFlag], matMat);
			m_lanesResult[nFlag].clear();

			TransformGround2Image(matMat, matMat);
			//get back to vector
			Mat2Lines(matMat, m_lanesResult[nFlag]);
			//remove the dummy line at the beginning
			m_lanesResult[nFlag].erase(m_lanesResult[nFlag].begin());
		}
	}

}
void CMultiROILaneDetection::GetVectorMax(const Mat &matInVector, double &dMax, int &nMaxLoc, int nIgnore){
	double tmax;
	int tmaxLoc;

	float *pfMatInVectorData = (float*)matInVector.data;
	if (matInVector.rows==1){ 
		/*initial value*/ 

		tmax = (double)pfMatInVectorData[0*matInVector.cols + matInVector.cols-1];

		tmaxLoc = matInVector.cols-1;//inVector->width-1; 
		/*loop*/ 
		for (int i=matInVector.cols-1-nIgnore; i>=0+nIgnore; i--){ 

			if (tmax<(double)pfMatInVectorData[0*matInVector.cols + i]){ 

				tmax = (double)pfMatInVectorData[0*matInVector.cols + i]; 
				tmaxLoc = i; 
			} 
		} 
	} 
	/*column vector */ 
	else{ 
		/*initial value*/ 
		tmax = (double)pfMatInVectorData[(matInVector.rows-1)*matInVector.cols + 0];
		tmaxLoc = matInVector.rows-1; 
		/*loop*/ 
		for (int i=matInVector.rows-1-nIgnore; i>=0+nIgnore; i--){ 

			if (tmax<(double)pfMatInVectorData[i*matInVector.cols + 0]){ 
				tmax = (double)pfMatInVectorData[i*matInVector.cols + 0];
				tmaxLoc = i; 
			} 
		} 
	} 
	//return
	if (dMax)
		dMax = tmax;
	if (nMaxLoc)
		nMaxLoc = tmaxLoc;
}
double CMultiROILaneDetection::GetLocalMaxSubPixel(double dVal1, double dVal2, double dVal3)
{
	//build an array to hold the x-values
	double Xp[] = {1, -1, 1, 0, 0, 1, 1, 1, 1};
	Mat X = Mat(3,3,CV_64FC1,Xp);

	//array to hold the y values
	double yp[] = {dVal1, dVal2, dVal3};
	Mat y = Mat(3, 1, CV_64FC1, yp);

	//solve to get the coefficients
	double Ap[3];
	Mat A = Mat(3, 1, CV_64FC1, Ap);

	solve(X,y,A,DECOMP_SVD);
	//get the local max
	double nMax;
	nMax = -0.5 * Ap[1] / Ap[0];

	//return
	return nMax;
}
void CMultiROILaneDetection::GetMaxLineScore(EROINUMBER nFlag){
	float fMaxScore = MAXCOMP;
	int nMaxIter = 0;
	for(int i=0;i<m_laneScore[nFlag].size(); i++){
		nMaxIter = (m_laneScore[nFlag].at(i)>fMaxScore ? i:nMaxIter);
		fMaxScore = m_laneScore[nFlag].at(nMaxIter);
	}
	
	SLine SMAxLane = m_lanes[nFlag].at(nMaxIter);
	//clear
	m_lanes[nFlag].clear();
	m_laneScore[nFlag].clear();

	//reload
	m_lanes[nFlag].push_back(SMAxLane); 
	m_laneScore[nFlag].push_back(fMaxScore);
}
void CMultiROILaneDetection::GetTrackingLineCandidate(EROINUMBER nFlag){
	vector<SLine> vecLanes;
	vector<SWorldLane> vecWorldLane;
	SLine sLineTemp;
	SWorldLane sWorldLaneTemp;
	for (int i = 0; i < m_lanes[nFlag].size(); i++){
		sLineTemp.ptStartLine.x = m_lanes[nFlag].at(i).ptStartLine.x / m_sRoiInfo[nFlag].dXScale;
		sLineTemp.ptStartLine.x += m_sRoiInfo[nFlag].dXLimit[0];
		sLineTemp.ptStartLine.y = m_lanes[nFlag].at(i).ptStartLine.y / m_sRoiInfo[nFlag].dYScale;
		sLineTemp.ptStartLine.y = m_sRoiInfo[nFlag].dYLimit[1] - m_lanes[nFlag].at(i).ptStartLine.y;

		sLineTemp.ptEndLine.x = m_lanes[nFlag].at(i).ptEndLine.x / m_sRoiInfo[nFlag].dXScale;
		sLineTemp.ptEndLine.x += m_sRoiInfo[nFlag].dXLimit[0];
		sLineTemp.ptEndLine.y = m_lanes[nFlag].at(i).ptEndLine.y / m_sRoiInfo[nFlag].dYScale;
		sLineTemp.ptEndLine.y = m_sRoiInfo[nFlag].dYLimit[1] - m_lanes[nFlag].at(i).ptEndLine.y;

		sWorldLaneTemp.fXcenter = (sLineTemp.ptStartLine.x + sLineTemp.ptEndLine.x) / 2;
		sWorldLaneTemp.fXderiv = sLineTemp.ptStartLine.x - sLineTemp.ptEndLine.x;
		//20150519/////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		vecLanes.push_back(sLineTemp);
		vecWorldLane.push_back(sWorldLaneTemp);
	}
	
//	m_lanes[nFlag]
	int nMaxIter = 0;
	float fMinDist = MINCOMP;
	float fTempComp;
	if ((nFlag == LEFT_ROI2) || (nFlag == LEFT_ROI3)){
		for (int i = 0; i < vecWorldLane.size(); i++){
			fTempComp = abs(m_sLeftTrakingLane.fXcenter-vecWorldLane.at(i).fXcenter);
			if (fTempComp < fMinDist){
				fMinDist = fTempComp;
				nMaxIter = i;
			}
		}
		SLine SMAxLane = m_lanes[nFlag].at(nMaxIter);
		float fMaxScore = m_laneScore[nFlag].at(nMaxIter);
		//clear
		m_lanes[nFlag].clear();
		m_laneScore[nFlag].clear();
		//reload
		m_lanes[nFlag].push_back(SMAxLane);
		m_laneScore[nFlag].push_back(fMaxScore);
	
	}
	else if ((nFlag == RIGHT_ROI2) || (nFlag == RIGHT_ROI3)){
		for (int i = 0; i < vecWorldLane.size(); i++){
			fTempComp = abs(m_sRightTrakingLane.fXcenter - vecWorldLane.at(i).fXcenter);
			if (fTempComp < fMinDist){
				fMinDist = fTempComp;
				nMaxIter = i;
			}
		}
		SLine SMAxLane = m_lanes[nFlag].at(nMaxIter);
		float fMaxScore = m_laneScore[nFlag].at(nMaxIter);
		//clear
		m_lanes[nFlag].clear();
		m_laneScore[nFlag].clear();
		//reload
		m_lanes[nFlag].push_back(SMAxLane);
		m_laneScore[nFlag].push_back(fMaxScore);
	}
	if (fMinDist > DIST_TRACKING_WIDTH){
		m_lanes[nFlag].clear();
		m_laneScore[nFlag].clear();
	}
}
void CMultiROILaneDetection::GetMaxLineScoreTwo(EROINUMBER nFlag){
	float fMaxScore = MAXCOMP;
	int nMaxIter = 0;
	float fScoreArr[2];
	for(int i=0;i<m_laneScore[nFlag].size(); i++){
		nMaxIter = (m_laneScore[nFlag].at(i)<fMaxScore ? i:nMaxIter);
	}
	fMaxScore = m_laneScore[nFlag].at(nMaxIter);
	SLine SMAxLane = m_lanes[nFlag].at(nMaxIter);
	//clear
	m_lanes[nFlag].clear();
	m_laneScore[nFlag].clear();

	//reload
	m_lanes[nFlag].push_back(SMAxLane);
	m_laneScore[nFlag].push_back(fMaxScore);
}
void CMultiROILaneDetection::Lines2Mat(const vector<SLine> &lines, Mat &mat)
{
	//allocate the matrix
	//*mat = cvCreateMat(2, size*2, FLOAT_MAT_TYPE);

	//loop and put values
	int j;
	float* pfMat =(float*) mat.data;
	for (int i=0; i<(int)lines.size(); i++)
	{
		j = 2*i;

		pfMat[mat.cols*0 + j]=(lines)[i].ptStartLine.x;
		pfMat[mat.cols*1 + j]=(lines)[i].ptStartLine.y;
		pfMat[mat.cols*0 + j+1]=(lines)[i].ptEndLine.x;
		pfMat[mat.cols*1 + j+1]=(lines)[i].ptEndLine.y;
	}
}
void CMultiROILaneDetection::Mat2Lines(const Mat &mat, vector<SLine> &lines)
{

	SLine line;
	float* pfMat = (float*)mat.data;
	//loop and put values
	for (int i=0; i<int(mat.cols/2); i++)
	{
		int j = 2*i;
		//get the line
		line.ptStartLine.x =pfMat[mat.cols*0 + j];
		line.ptStartLine.y =pfMat[mat.cols*1 + j];
		line.ptEndLine.x =pfMat[mat.cols*0 + j+1];
		line.ptEndLine.y =pfMat[mat.cols*1 + j+1];
		//push it
		lines.push_back(line);
	}
}
void CMultiROILaneDetection::GroupLines(vector<SLine> &lines, vector<float> &lineScores,
	float groupThreshold, Size_<float> bbox)
{

	//convert the lines into r-theta parameters
	int numInLines = lines.size();
	vector<float> rs(numInLines);
	vector<float> thetas(numInLines);
	for (int i=0; i<numInLines; i++)
		LineXY2RTheta(lines[i], rs[i], thetas[i]);

	//flag for stopping
	bool stop = false;
	while (!stop)
	{
		//minimum distance so far
		float minDist = groupThreshold+5, dist;
		vector<float>::iterator ir, jr, itheta, jtheta, minIr, minJr, minItheta, minJtheta,
			iscore, jscore, minIscore, minJscore;
		//compute pairwise distance between detected maxima
		for (ir=rs.begin(), itheta=thetas.begin(), iscore=lineScores.begin();
			ir!=rs.end(); ir++, itheta++, iscore++)
			for (jr=ir+1, jtheta=itheta+1, jscore=iscore+1;
				jr!=rs.end(); jr++, jtheta++, jscore++)
			{
				//add pi if neg
				float t1 = *itheta<0 ? *itheta : *itheta+CV_PI;
				float t2 = *jtheta<0 ? *jtheta : *jtheta+CV_PI;
				//get distance
				dist = 1 * fabs(*ir - *jr) + 1 * fabs(t1 - t2);//fabs(*itheta - *jtheta);
				//check if minimum
				if (dist<minDist)
				{
					minDist = dist;
					minIr = ir; minItheta = itheta;
					minJr = jr; minJtheta = jtheta;
					minIscore = iscore; minJscore = jscore;
				}
			}
			//check if minimum distance is less than groupThreshold
			if (minDist >= groupThreshold)
				stop = true;
			else
			{
				//put into the first
				*minIr = (*minIr + *minJr)/2;
				*minItheta = (*minItheta + *minJtheta)/2;
				*minIscore = (*minIscore + *minJscore)/2;
				//delete second one
				rs.erase(minJr);
				thetas.erase(minJtheta);
				lineScores.erase(minJscore);
			}
	}//while

	//put back the lines
	lines.clear();
	//lines.resize(rs.size());
	vector<float> newScores=lineScores;
	lineScores.clear();
	for (int i=0; i<(int)rs.size(); i++)
	{
		//get the line
		SLine line;
		//mcvIntersectLineRThetaWithBB(rs[i], thetas[i], bbox, &line);
		IntersectLineRThetaWithBB(rs[i], thetas[i], bbox, &line);
		//put in place descendingly
		vector<float>::iterator iscore;
		vector<SLine>::iterator iline;
		for (iscore=lineScores.begin(), iline=lines.begin();
			iscore!=lineScores.end() && newScores[i]<=*iscore; iscore++, iline++);
			lineScores.insert(iscore, newScores[i]);
		lines.insert(iline, line);
	}
	//clear
	newScores.clear();
}
void CMultiROILaneDetection::LineXY2RTheta(const SLine &line, float &r, float &theta)
{
	//check if vertical line x1==x2
	if(line.ptStartLine.x == line.ptEndLine.x)
	{
		//r is the x
		r = fabs(line.ptStartLine.x);
		//theta is 0 or pi
		theta = line.ptStartLine.x>=0 ? 0. : CV_PI;
	}
	//check if horizontal i.e. y1==y2
	else if(line.ptStartLine.y == line.ptEndLine.y)
	{
		//r is the y
		r = fabs(line.ptStartLine.y);
		//theta is pi/2 or -pi/2
		theta = (float) line.ptStartLine.y>=0 ? CV_PI/2 : -CV_PI/2;
	}
	//general line
	else
	{
		//tan(theta) = (x2-x1)/(y1-y2)
		theta =  atan2(line.ptEndLine.x-line.ptStartLine.x,
			line.ptStartLine.y-line.ptEndLine.y);
		//r = x*cos(theta)+y*sin(theta)
		float r1 = line.ptStartLine.x * cos(theta) + line.ptStartLine.y * sin(theta);
		r = line.ptEndLine.x * cos(theta) + line.ptEndLine.y * sin(theta);
		//adjust to add pi if necessary
		if(r1<0 || r<0)
		{
			//add pi
			theta += CV_PI;
			if(theta>CV_PI)
				theta -= 2*CV_PI;
			//take abs
			r = fabs(r);
		}
	}
}
void CMultiROILaneDetection::IntersectLineRThetaWithBB(float r, float theta, const Size_<float> bbox, SLine *outLine)
{
	//hold parameters
	double xup, xdown, yleft, yright;

	//intersect with top and bottom borders: y=0 and y=bbox.height-1
	if (cos(theta)==0) //horizontal line
	{
		xup = xdown = bbox.width+2;
	}
	else
	{
		xup = r / cos(theta);
		xdown = (r-bbox.height*sin(theta))/cos(theta);
	}

	//intersect with left and right borders: x=0 and x=bbox.widht-1
	if (sin(theta)==0) //horizontal line
	{
		yleft = yright = bbox.height+2;
	}
	else
	{
		yleft = r/sin(theta);
		yright = (r-bbox.width*cos(theta))/sin(theta);
	}

	//points of intersection

	Point2d pts[4] = {Point2d(xup, 0),Point2d(xdown,bbox.height),
		Point2d(0, yleft),Point2d(bbox.width, yright)};
	//get the starting point
	int i;
	for (i=0; i<4; i++)
	{
		//if point inside, then put it

		if(IsPointInside(pts[i], bbox))
		{
			outLine->ptStartLine.x = pts[i].x;
			outLine->ptStartLine.y = pts[i].y;
			//get out of for loop
			break;
		}
	}
	//get the ending point
	for (i++; i<4; i++)
	{
		//if point inside, then put it
		if(IsPointInside(pts[i], bbox))
		{
			outLine->ptEndLine.x = pts[i].x;
			outLine->ptEndLine.y = pts[i].y;
			//get out of for loop
			break;
		}
	}
}
bool CMultiROILaneDetection::IsPointInside(Point2d point, Size_<int> bbox)
{
	return (point.x>=0 && point.x<=bbox.width
		&& point.y>=0 && point.y<=bbox.height) ? true : false;
}
bool CMultiROILaneDetection::IsPointInside(Point2d point, Size_<float> bbox)
{
	return (point.x>=0 && point.x<=bbox.width
		&& point.y>=0 && point.y<=bbox.height) ? true : false;
}
void CMultiROILaneDetection::GetLinesBoundingBoxes(const vector<SLine> &lines, LineType type,
	Size_<int> size, vector<Rect> &boxes)
{
	//copy lines to boxes
	int start, end;
	//clear
	boxes.clear();
	switch(type)
	{
	case LINE_VERTICAL:
		for(unsigned int i=0; i<lines.size(); ++i)
		{
			//get min and max x and add the bounding box covering the whole height
			start = (int)min(lines[i].ptStartLine.x, lines[i].ptEndLine.x);
			end = (int)max(lines[i].ptStartLine.x, lines[i].ptEndLine.x);
			boxes.push_back(Rect_<int>(start, 0, end-start+1, size.height-1));
		}
		break;

	case LINE_HORIZONTAL:
		for(unsigned int i=0; i<lines.size(); ++i)
		{
			//get min and max y and add the bounding box covering the whole width
			start = (int)min(lines[i].ptStartLine.y, lines[i].ptEndLine.y);
			end = (int)max(lines[i].ptStartLine.y, lines[i].ptEndLine.y);
			boxes.push_back(Rect_<int>(0, start, size.width-1, end-start+1));
		}
		break;
	}
}
void CMultiROILaneDetection::GroupBoundingBoxes(vector<Rect> &boxes, LineType type, float groupThreshold){
	bool cont = true;

	//Todo: check if to intersect with bounding box or not

	//save boxes
	//vector<CvRect> tboxes = boxes;

	//loop to get the largest overlap (according to type) and check
	//the overlap ratio
	float overlap, maxOverlap;
	while(cont)
	{
		maxOverlap =  overlap = -1e5;
		//loop on lines and get max overlap
		vector<Rect>::iterator i, j, maxI, maxJ;
		for(i = boxes.begin(); i != boxes.end(); i++)
		{
			for(j = i+1; j != boxes.end(); j++)
			{
				switch(type)
				{
				case LINE_VERTICAL:
					//get one with smallest x, and compute the x2 - x1 / width of smallest
					//i.e. (x12 - x21) / (x22 - x21)
					overlap = i->x < j->x  ?
						(i->x + i->width - j->x) / (float)j->width :
					(j->x + j->width - i->x) / (float)i->width;

					break;

				case LINE_HORIZONTAL:
					//get one with smallest y, and compute the y2 - y1 / height of smallest
					//i.e. (y12 - y21) / (y22 - y21)
					overlap = i->y < j->y  ?
						(i->y + i->height - j->y) / (float)j->height :
					(j->y + j->height - i->y) / (float)i->height;

					break;

				} //switch

				//get maximum
				if(overlap > maxOverlap)
				{
					maxI = i;
					maxJ = j;
					maxOverlap = overlap;
				}
			} //for j
		} // for i
		// 	//debug
		// 	if(DEBUG_LINES) {
		// 	    cout << "maxOverlap=" << maxOverlap << endl;
		// 	    cout << "Before grouping\n";
		// 	    for(unsigned int k=0; k<boxes.size(); ++k)
		// 		SHOW_RECT(boxes[k]);
		// 	}

		//now check the max overlap found against the threshold
		if (maxOverlap >= groupThreshold)
		{
			//combine the two boxes
			*maxI  = Rect_<float>(min((*maxI).x, (*maxJ).x),
				min((*maxI).y, (*maxJ).y),
				max((*maxI).width, (*maxJ).width),
				max((*maxI).height, (*maxJ).height));
			//delete the second one
			boxes.erase(maxJ);
		}
		else
			//stop
			cont = false;

		// 	//debug
		// 	if(DEBUG_LINES) {
		// 	    cout << "After grouping\n";
		// 	    for(unsigned int k=0; k<boxes.size(); ++k)
		// 		SHOW_RECT(boxes[k]);
		// 	}
	} //while
}
void CMultiROILaneDetection::SetMat(Mat& imgInMat,Rect_<int> RectMask, double val)
{

	//get x-end points of region to work on, and work on the whole image height
	//(int)fmax(fmin(line.startPoint.x, line.endPoint.x)-xwindow, 0);

	int xstart = RectMask.x;
	int xend = RectMask.x + RectMask.width-1;

	int ystart = RectMask.y;
	int yend = RectMask.y + RectMask.height-1;

	//set other two windows to zero

	Mat imgMask;

	Rect_<int> rectInfunction;
	//part to the left of required region


	rectInfunction=Rect(0, 0, xstart-1, imgInMat.rows);
	//cout<<rectInfunction<<endl;
	if (rectInfunction.x<imgInMat.cols && rectInfunction.y<imgInMat.rows &&
		rectInfunction.x>=0 && rectInfunction.y>=0 && rectInfunction.width>0 && rectInfunction.height>0)
	{
		imgInMat(rectInfunction) = val;
	}
	//part to the right of required region

	rectInfunction=Rect(xend+1, 0, imgInMat.cols-xend-1,imgInMat.rows);
	//cout<<rectInfunction<<endl;
	if (rectInfunction.x<imgInMat.cols && rectInfunction.y<imgInMat.rows &&
		rectInfunction.x>=0 && rectInfunction.y>=0 && rectInfunction.width>0 && rectInfunction.height>0)
	{
		imgInMat(rectInfunction) = val;
	}

	//part to the top

	rectInfunction=Rect(xstart, 0, RectMask.width, ystart-1);
	//cout<<rectInfunction<<endl;
	if (rectInfunction.x<imgInMat.cols && rectInfunction.y<imgInMat.rows &&
		rectInfunction.x>=0 && rectInfunction.y>=0 && rectInfunction.width>0 && rectInfunction.height>0)
	{

		imgInMat(rectInfunction) = val;

	}

	//part to the bottom
	//rect = cvRect(xstart, yend+1, mask.width, inMat->height-yend-1);
	rectInfunction=Rect(xstart, yend+1, RectMask.width, imgInMat.rows-yend-1);
	//cout<<rectInfunction<<endl;
	if (rectInfunction.x<imgInMat.cols && rectInfunction.y<imgInMat.rows &&
		rectInfunction.x>=0 && rectInfunction.y>=0 && rectInfunction.width>0 && rectInfunction.height>0)
	{

		imgInMat(rectInfunction) = val;
	}


}	
void CMultiROILaneDetection::FitRansacLine(const Mat& matImage, int numSamples, int numIterations,
	float threshold, float scoreThreshold, int numGoodFit,
	bool getEndPoints, LineType lineType,
	SLine *lineXY, float *lineRTheta, float *lineScore,EROINUMBER nFlag)
{

	float* pfMatImage = (float*)matImage.data;


	//get the points with non-zero pixels

	Mat matPoints;
	bool bZeroPoint;

	bZeroPoint = GetNonZeroPoints(matImage,matPoints,true);

	if(!bZeroPoint)
		return;

	if(matPoints.cols!=1)
		if(numSamples>matPoints.cols)
			numSamples = matPoints.cols;
	//subtract half

	matPoints+=0.5;

	//normalize pixels values to get weights of each non-zero point
	//get third row of points containing the pixel values

	Mat matW = matPoints.row(2);

	//normalize it

	Mat matWeights = matW.clone();


	normalize(matWeights,matWeights,1,0,CV_L1);

	//get cumulative    sum

	CumSum(matWeights,matWeights);

	//random number generator	
	RNG rngNum(0xffffffff);

	//matrix to hold random sample
	Mat matRandInd = Mat(numSamples,1,CV_32SC1);
	Mat matSamplePoints = Mat(2,numSamples,CV_32FC1);
	//flag for points currently included in the set
	Mat matPointIn = Mat(1,matPoints.cols, CV_8SC1);

	//returned lines
	float curLineRTheta[2], curLineAbc[3];
	float bestLineRTheta[2]={-1.f,0.f}, bestLineAbc[3];
	float bestScore=0, bestDist=1e5;
	float dist, score;
	SLine curEndPointLine,bestEndPointLine;
	curEndPointLine.ptStartLine.x=-1.0;
	curEndPointLine.ptStartLine.y=-1.0;
	curEndPointLine.ptEndLine.x=-1.0;
	curEndPointLine.ptEndLine.y=-1.0;
	bestEndPointLine.ptStartLine.x=-1.0;
	bestEndPointLine.ptStartLine.y=-1.0;
	bestEndPointLine.ptEndLine.x=-1.0;
	bestEndPointLine.ptEndLine.y=-1.0;

	//variabels for getting endpoints
	//int mini, maxi;
	float minc=1e5f, maxc=-1e5f, mind, maxd;
	float x, y, c=0.;

	Point_<float> ptMin = Point_<float>(-1. , -1.);
	Point_<float> ptMax = Point_<float>(-1. , -1.);

	int *piMatRandIndData =(int*) matRandInd.data;
	float * pfMatSamplePoins = (float*) matSamplePoints.data;
	char *pcMatPointIn = (char*) matPointIn.data;
	float *pfMatPoints = (float*) matPoints.data;
	//cout<<"Iterations"<<numIterations<<endl;
	//outer loop
	for (int i=0; i<numIterations; i++)
	{
		//set flag to zero
		matPointIn.zeros(1,matPoints.cols,CV_8SC1);
		//get random sample from the points
		SampleWeighted(matWeights, numSamples, matRandInd, rngNum);


		for (int j=0; j<numSamples; j++)
		{
			//flag it as included

			pcMatPointIn[matPointIn.cols*0 + piMatRandIndData[matRandInd.cols*j + 0] ] = 1;
			//put point
			pfMatSamplePoins[matSamplePoints.cols*0 + j ] = pfMatPoints[matPoints.cols*0 + piMatRandIndData[matRandInd.cols*j + 0]];
			pfMatSamplePoins[matSamplePoints.cols*1 + j ] = pfMatPoints[matPoints.cols*1 + piMatRandIndData[matRandInd.cols*j +0]];
		}
		//fit the line
		FitRobustLine(matSamplePoints, curLineRTheta, curLineAbc);
		//get end points from points in the samplePoints
		minc = 1e5; mind = 1e5; maxc = -1e5; maxd = -1e5;
		for (int j=0; getEndPoints && j<numSamples; ++j)
		{
			//get x & y

			x = pfMatSamplePoins[matSamplePoints.cols*0 + j];  //CV_MAT_ELEM(*samplePoints, float, 0, j);//
			y = pfMatSamplePoins[matSamplePoints.cols*1 + j];//////////////////////////////////////////////////////////////////////////???

			//get the coordinate to work on
			if (lineType == LINE_HORIZONTAL)
				c = x;
			else if (lineType == LINE_VERTICAL)
				c = y;
			//compare
			if (c>maxc)
			{
				maxc = c;

				ptMax = Point_<float>(x,y);		//////////////////////////////////////////////////////////////////////////
			}
			if (c<minc)
			{
				minc = c;

				ptMin = Point_<float>(x,y);		//////////////////////////////////////////////////////////////////////////
			}
		} //for

		//loop on other points and compute distance to the line
		score=0;
		for( int j=0; j<matPoints.cols; j++)//
		{
			// 	    //if not already inside

			dist = fabs(pfMatPoints[matPoints.cols*0 + j]*curLineAbc[0] + pfMatPoints[matPoints.cols*1 + j]*curLineAbc[1]+curLineAbc[2]);//
			//check distance
			if (dist<=threshold)
			{
				//add this point

				pcMatPointIn[matPointIn.cols*0 +j ] = 1;//
				//update score

				score += pfMatImage[matImage.cols*(int)(pfMatPoints[matPoints.cols*1 + j]-0.5)+(int)(pfMatPoints[matPoints.cols*0 + j]-0.5)];//
			}
			// 	    }
		}

		//check the number of close points and whether to consider this a good fit
		//int numClose = cvCountNonZero(pointIn);
		int numClose = countNonZero(matPointIn);
		//cout << "numClose=" << numClose << "\n";
		if (numClose >= numGoodFit)
		{
			//get the points included to fit this line	
			Mat matFitPoints = Mat(2,numClose,CV_32FC1);//
			float* pfMatFitPoints = (float*) matFitPoints.data;
			int k=0;
			//loop on points and copy points included
			for (int j=0; j<matPoints.cols; j++)//
			{


				if(pcMatPointIn[matPointIn.cols*0 + j])
				{

					pfMatFitPoints[matFitPoints.cols*0 + k] = pfMatPoints[matPoints.cols*0 + j];
					pfMatFitPoints[matFitPoints.cols*1 + k] = pfMatPoints[matPoints.cols*1 + j];
					k++;
				}
			}
			//fit the line
			FitRobustLine(matSamplePoints, curLineRTheta, curLineAbc);

			//compute distances to new line
			dist = 0.;
			//compute distances to new line
			for (int j=0; j<matFitPoints.cols; j++)
				//for (int j=0; j<fitPoints->cols; j++)
			{////	
				x = pfMatFitPoints[matFitPoints.cols*0 + j];//
				y = pfMatFitPoints[matFitPoints.cols*1 + j];//

				float d = fabs( x * curLineAbc[0] +	y * curLineAbc[1] +	curLineAbc[2])
					* pfMatImage[matImage.cols*(int)(y-0.5)+ (int)(x-0.5) ];//

				dist += d;
				////
			}

			//now check if we are getting the end points
			if (getEndPoints)
			{

				//get distances


				mind = ptMin.x * curLineAbc[0] +	ptMin.y * curLineAbc[1] + curLineAbc[2];  //
				maxd = ptMax.x * curLineAbc[0] +	ptMax.y * curLineAbc[1] + curLineAbc[2];//
				//we have the index of min and max points, and
				//their distance, so just get them and compute
				//the end points

				//////////////////////////////////////////////////////////////////////////
				curEndPointLine.ptStartLine.x = ptMin.x - mind * curLineAbc[0];//
				curEndPointLine.ptStartLine.y = ptMin.y - mind * curLineAbc[1];//

				curEndPointLine.ptEndLine.x = ptMax.x	- maxd * curLineAbc[0];//
				curEndPointLine.ptEndLine.y = ptMax.y	- maxd * curLineAbc[1];//


			}

			//dist /= score;

			//clear fitPoints

			//check if to keep the line as best
			if (score>=scoreThreshold && score>bestScore)//dist<bestDist //(numClose > bestScore)
			{
				//update max
				bestScore = score; //numClose;
				bestDist = dist;
				//copy
				bestLineRTheta[0] = curLineRTheta[0];
				bestLineRTheta[1] = curLineRTheta[1];
				bestLineAbc[0] = curLineAbc[0];
				bestLineAbc[1] = curLineAbc[1];
				bestLineAbc[2] = curLineAbc[2];
				bestEndPointLine = curEndPointLine;
			}
		}
	} // for i

	//return
	if (lineRTheta)
	{
		lineRTheta[0] = bestLineRTheta[0];
		lineRTheta[1] = bestLineRTheta[1];
	}
	if (lineXY)
	{
		if (getEndPoints)
			*lineXY = bestEndPointLine;
		else
		{
			IntersectLineRThetaWithBB(lineRTheta[0], lineRTheta[1],	Size(matImage.cols-1, matImage.rows-1),lineXY);
		}
	}
	if (lineScore)
		*lineScore = bestScore;

	//clear

}
bool CMultiROILaneDetection::GetNonZeroPoints(const Mat& matInMat, Mat& matOutMat,bool floatMat)
{

	int k=0;
	//get number of non-zero points
	//int numnz = cvCountNonZero(inMat);
	int numnz =	countNonZero(matInMat);
	//allocate the point array and get the points
	if (numnz)
	{
		if (floatMat)
		{
			matOutMat = Mat(3,numnz,CV_32FC1);
		}
		else
		{	
			matOutMat = Mat(3,numnz,CV_32SC1);//=Mat(3,numnz,CV_32FC1);
		}
	}
	else
		return false;


	float* pfMatInMat = (float*) matInMat.data;
	float* pfMatOutMat = (float*) matOutMat.data;
	/*loop and allocate the points*/ 
	for (int i=0; i<matInMat.rows; i++) 
		for (int j=0; j<matInMat.cols; j++) 
			if (pfMatInMat[i*matInMat.cols+j])
			{ 
				pfMatOutMat[0*matOutMat.cols+k] = (float)j;
				pfMatOutMat[1*matOutMat.cols+k] = (float)i;
				pfMatOutMat[2*matOutMat.cols+k] = (float)pfMatInMat[i*matInMat.cols+j];

				k++; 
			} 

			//return
			return true;
}

void CMultiROILaneDetection::CumSum(const Mat &inMat, Mat &outMat)
{


	float* pfInMatData = (float*) inMat.data;
	float* pfOutMatData = (float*) outMat.data;

	if(inMat.rows==1)
		for(int i=1; i<outMat.cols; i++)
			pfOutMatData[0*outMat.cols+i]+=pfOutMatData[0*outMat.cols+i-1];
	else
		for(int i=1; i<outMat.rows; i++)
			pfOutMatData[i*outMat.cols+0]+=pfOutMatData[(i-1)*outMat.cols+0];

}

void CMultiROILaneDetection::SampleWeighted(const Mat &cumSum, int numSamples, Mat &randInd, RNG &rng)
{
	//     //get cumulative sum of the weights
	//     //OPTIMIZE:should pass it later instead of recomputing it

	//check if numSamples is equal or more
	int i=0;
	int* piRandInd=(int*)randInd.data;
	float* pfCumSumData = (float*)cumSum.data;

	if (numSamples >= cumSum.cols)
	{
		for (; i<numSamples; i++)
		{
			piRandInd[randInd.cols*i+0]=i;
		}

	}
	else
	{
		//loop
		while(i<numSamples)
		{
			//get random number

			double r = rng.uniform(0.,1.);
			//get the index from cumSum
			int j;

			for (j=0; j<cumSum.cols && r>pfCumSumData[cumSum.cols*0+j]; j++);

			//make sure this index wasnt chosen before
			bool put = true;
			for (int k=0; k<i; k++)
			{

				if( piRandInd[randInd.cols*k+0] == j )
					//put it
					put = false;
			}

			if (put)
			{
				//put it in array
				piRandInd[randInd.cols*i+0] = j ; 
				//inc
				i++;
			}
		} //while
	} //if
}


void CMultiROILaneDetection::FitRobustLine(const Mat &matPoints, float *lineRTheta, float *lineAbc)
{

	//clone the points

	Mat matClPoints = matPoints.clone();

	//get mean of the points and subtract from the original points
	float meanX=0, meanY=0;

	Scalar SMean;

	Mat matRow1, matRow2;
	//get first row, compute avg and store

	matRow1 = matClPoints.row(0);

	SMean = mean(matRow1);

	meanX = (float) SMean.val[0];

	subtract(matRow1,SMean,matRow1);

	//same for second row

	matRow2 = matClPoints.row(1);

	SMean = mean(matRow2);
	meanY = (float) SMean.val[0]; 

	subtract(matRow2,SMean,matRow2);

	//compute the SVD for the centered points array

	Mat matW = Mat(2, 1, CV_32FC1);
	Mat matV = Mat(2, 2, CV_32FC1);

	Mat matCPointst = Mat(matClPoints.cols,matClPoints.rows, CV_32FC1);


	transpose(matClPoints, matCPointst);

	m_SvdCalc(matCPointst,SVD::FULL_UV);
	matV=m_SvdCalc.vt;


	transpose(matV,matV);

	//get the [a,b] which is the second column corresponding to
	//smaller singular value
	float a, b, c;
	float* pfMatV =(float*) matV.data;
	//////////////////////////????????????????????

	float tempa = pfMatV[matV.cols*0 + 1];
	float tempb = pfMatV[matV.cols*1 + 1];
	//


	float tempc = -meanX*tempa-meanY*tempb;

	//printf("original\n %f	%f	%f\n",a,b,c);
	//	printf("mat\n %f	%f	%f\n\n",tempa,tempb,tempc);

	//	cvWaitKey(0);
	a=tempa;
	b=tempb;
	c=tempc;


	//compute r and theta

	float r, theta;
	theta = atan2(b, a);
	r = meanX * cos(theta) + meanY * sin(theta);
	//correct
	if (r<0)
	{
		//correct r
		r = -r;
		//correct theta
		theta += CV_PI;
		if (theta>CV_PI)
			theta -= 2*CV_PI;
	}
	//return
	if (lineRTheta)
	{
		lineRTheta[0] = r;
		lineRTheta[1] = theta;
	}
	if (lineAbc)
	{
		lineAbc[0] = a;
		lineAbc[1] = b;
		lineAbc[2] = c;
	}

}

void CMultiROILaneDetection::GetCameraPose(EROINUMBER nFlag, Vector<Mat> &vecMat){
	int nFrameSize = vecMat.size();
	m_imgIPM[nFlag].create(m_sRoiInfo[nFlag].sizeIPM,CV_32FC1);
	Mat imgSum = Mat::zeros(vecMat[0].size(),CV_32FC1);
	for(int i=0; i<nFrameSize;i++)
	{
		//vecMat[i].convertTo(vecMat[i],CV_64FC1);
		//imgSum = imgSum*(i+1);
		imgSum += vecMat[i]; 
		//imgSum /=(i+2);
	}
	imgSum /= nFrameSize;
	imshow("filter sum",imgSum);
	ShowImageNormalize("filter sum norm",imgSum);
	Mat rowMat;
	rowMat = Mat(imgSum).reshape(0,1);
	//get the quantile
	float fQval;
	fQval = quantile((float*) &rowMat.data[0], rowMat.cols, m_sConfig.fLowerQuantile);
	Mat imgSumThres;
	threshold(imgSum,imgSumThres,fQval,NULL,THRESH_TOZERO);
	ShowImageNormalize("filter sum thres norm",imgSumThres);
	waitKey(0);
	m_ipmFiltered[nFlag] = imgSum;
	m_filteredThreshold[nFlag] = imgSumThres;
	GetLinesIPM(nFlag);
	LineFitting(nFlag);
	cout<<"lane size"<<m_lanes[nFlag].size()<<endl;
	IPM2ImLines(nFlag);
	
}
void CMultiROILaneDetection::ClearResultVector(EROINUMBER nFlag){
	m_lanes[nFlag].clear();
	m_laneScore[nFlag].clear();
	m_lanesResult[nFlag].clear();
	m_lanesGroundResult[nFlag].clear();
}
Point CMultiROILaneDetection::TransformPointImage2Ground(Point ptIn){
	Point ptResult;
	float fArrPt[] = { ptIn.x, ptIn.y };
	Mat matUvPt(2, 1, CV_32FC1, fArrPt);
	Mat matXyPt(2, 1, CV_32FC1);
	TransformImage2Ground(matUvPt, matXyPt);
	ptResult.x = matXyPt.at<float>(0, 0);
	ptResult.y = matXyPt.at<float>(1, 0);
	return ptResult;
}
Point CMultiROILaneDetection::TransformPointGround2Image(Point ptIn){
	
	Mat matMat = Mat(2, 1, CV_32FC1);
	float* pfMat = (float*)matMat.data;
	pfMat[0] = ptIn.x;	pfMat[1] = ptIn.y;
	TransformGround2Image(matMat, matMat);
	return Point(pfMat[0], pfMat[1]);
}
void CMultiROILaneDetection::KalmanTrackingStage(EROINUMBER nFlag){
	
	
	bool bFlag=false;
	if (nFlag == KALMAN_LEFT){
		m_bLeftDraw = false;
		bFlag = m_bTracking[LEFT_ROI2];
	}
	else if (nFlag == KALMAN_RIGHT){
		m_bRightDraw = false;
		bFlag = m_bTracking[RIGHT_ROI2];
	}
	if (bFlag == false){
		//cout << "False" << endl;
		return;
	}
	//cout << "Kalman stage" << endl;
	if (nFlag == KALMAN_LEFT)
	{
		if (m_SKalmanLeftLane.cntNum == 0)
		{
			m_SKalmanLeftLane.SKalmanTrackingLineBefore = m_SKalmanLeftLane.SKalmanTrackingLine;
			//cout << "Left Kalman Start" << endl;
			KalmanSetting(m_SKalmanLeftLane, nFlag);
		}
		else
		{
			Mat matPrediction = m_SKalmanLeftLane.KF.predict();
			SLine SLinePredict;
			SLinePredict.fXcenter = matPrediction.at<float>(0);
			SLinePredict.fXderiv = matPrediction.at<float>(1);
			m_SKalmanLeftLane.matMeasurement.at<float>(0) = m_SKalmanLeftLane.SKalmanTrackingLine.fXcenter;
			m_SKalmanLeftLane.matMeasurement.at<float>(1) = m_SKalmanLeftLane.SKalmanTrackingLine.fXderiv;
			m_SKalmanLeftLane.matMeasurement.at<float>(2) = 0;
			m_SKalmanLeftLane.matMeasurement.at<float>(3) = 0;
			//m_SKalmanLeftLane.KF.measurementMatrix.at<float>(4) = 0;
			//m_SKalmanLeftLane.KF.measurementMatrix.at<float>(5) = 0;
			Mat matEstimated = m_SKalmanLeftLane.KF.correct(m_SKalmanLeftLane.matMeasurement);
			SLine SLineEstimated;
			SLineEstimated.fXcenter = matEstimated.at<float>(0);
			SLineEstimated.fXderiv = matEstimated.at<float>(1);

			/*cout << "Kalman stage " << endl;
			cout << "before : " << m_SKalmanLeftLane.SKalmanTrackingLineBefore.fXcenter;
			cout << " , " << m_SKalmanLeftLane.SKalmanTrackingLineBefore.fXderiv << endl;
			cout << "current : " << m_SKalmanLeftLane.SKalmanTrackingLine.fXcenter;
			cout << " , " << m_SKalmanLeftLane.SKalmanTrackingLine.fXderiv << endl;
			cout << "KalmanFiltered : " << SLineEstimated.fXcenter;
			cout << " , " << SLineEstimated.fXderiv << endl;*/
			
			m_sLeftTrakingLane.ptStartLane.x = (m_sLeftTrakingLane.ptStartLane.y - m_sLeftTrakingLane.ptEndLane.y) / 2
				* SLineEstimated.fXderiv + SLineEstimated.fXcenter;
			m_sLeftTrakingLane.ptEndLane.x = (-m_sLeftTrakingLane.ptStartLane.y + m_sLeftTrakingLane.ptEndLane.y) / 2
				* SLineEstimated.fXderiv + SLineEstimated.fXcenter;
			Point ptUvSt = TransformPointGround2Image(m_sLeftTrakingLane.ptStartLane);
			Point ptUvEnd = TransformPointGround2Image(m_sLeftTrakingLane.ptEndLane);
			m_sLeftTrakingLane.ptUvStartLine = ptUvSt;
			m_sLeftTrakingLane.ptUvEndLine = ptUvEnd;
			m_sLeftTrakingLane.fXcenter = SLineEstimated.fXcenter;
			m_sLeftTrakingLane.fXderiv = SLineEstimated.fXderiv;
			//line(m_imgResizeOrigin, ptUvSt, ptUvEnd, Scalar(0, 0, 255), 2);
			m_bLeftDraw = true;
			m_SKalmanLeftLane.SKalmanTrackingLineBefore = m_SKalmanLeftLane.SKalmanTrackingLine;
		}
				
		//cout << "cntNum : " << m_SKalmanLeftLane.cntNum << endl;
		m_SKalmanLeftLane.cntNum++;
	}

	if (nFlag == KALMAN_RIGHT){
		if (m_SKalmanRightLane.cntNum == 0)
		{
			m_SKalmanRightLane.SKalmanTrackingLineBefore = m_SKalmanRightLane.SKalmanTrackingLine;
			//cout << "Left Kalman Start" << endl;
			KalmanSetting(m_SKalmanRightLane, nFlag);
		}
		else
		{
			Mat matPrediction = m_SKalmanRightLane.KF.predict();
			SLine SLinePredict;
			SLinePredict.fXcenter = matPrediction.at<float>(0);
			SLinePredict.fXderiv = matPrediction.at<float>(1);
			m_SKalmanRightLane.matMeasurement.at<float>(0) = m_SKalmanRightLane.SKalmanTrackingLine.fXcenter;
			m_SKalmanRightLane.matMeasurement.at<float>(1) = m_SKalmanRightLane.SKalmanTrackingLine.fXderiv;
			m_SKalmanRightLane.matMeasurement.at<float>(2) = 0;
			m_SKalmanRightLane.matMeasurement.at<float>(3) = 0;
			//m_SKalmanLeftLane.KF.measurementMatrix.at<float>(4) = 0;
			//m_SKalmanLeftLane.KF.measurementMatrix.at<float>(5) = 0;
			Mat matEstimated = m_SKalmanRightLane.KF.correct(m_SKalmanRightLane.matMeasurement);
			SLine SLineEstimated;
			SLineEstimated.fXcenter = matEstimated.at<float>(0);
			SLineEstimated.fXderiv = matEstimated.at<float>(1);

			/*cout << "Kalman stage " << endl;
			cout << "before : " << m_SKalmanLeftLane.SKalmanTrackingLineBefore.fXcenter;
			cout << " , " << m_SKalmanLeftLane.SKalmanTrackingLineBefore.fXderiv << endl;
			cout << "current : " << m_SKalmanLeftLane.SKalmanTrackingLine.fXcenter;
			cout << " , " << m_SKalmanLeftLane.SKalmanTrackingLine.fXderiv << endl;
			cout << "KalmanFiltered : " << SLineEstimated.fXcenter;
			cout << " , " << SLineEstimated.fXderiv << endl;*/

			m_sRightTrakingLane.ptStartLane.x = (m_sRightTrakingLane.ptStartLane.y - m_sRightTrakingLane.ptEndLane.y) / 2
				* SLineEstimated.fXderiv + SLineEstimated.fXcenter;
			m_sRightTrakingLane.ptEndLane.x = (-m_sRightTrakingLane.ptStartLane.y + m_sRightTrakingLane.ptEndLane.y) / 2
				* SLineEstimated.fXderiv + SLineEstimated.fXcenter;
			Point ptUvSt = TransformPointGround2Image(m_sRightTrakingLane.ptStartLane);
			Point ptUvEnd = TransformPointGround2Image(m_sRightTrakingLane.ptEndLane);
			m_sRightTrakingLane.ptUvStartLine = ptUvSt;
			m_sRightTrakingLane.ptUvEndLine = ptUvEnd;
			m_sRightTrakingLane.fXcenter = SLineEstimated.fXcenter;
			m_sRightTrakingLane.fXderiv = SLineEstimated.fXderiv;
			//line(m_imgResizeOrigin, ptUvSt, ptUvEnd, Scalar(0, 0, 255), 2);
			m_bRightDraw = true;
			m_SKalmanRightLane.SKalmanTrackingLineBefore = m_SKalmanRightLane.SKalmanTrackingLine;
		}

		//cout << "cntNum : " << m_SKalmanLeftLane.cntNum << endl;
		m_SKalmanRightLane.cntNum++;

	}

}
void CMultiROILaneDetection::KalmanSetting(SKalman &SKalmanInput, EROINUMBER nflag){
	if (nflag == KALMAN_LEFT ){
		SKalmanInput.KF.transitionMatrix = *(Mat_<float>(6, 6) <<
			1, 0, 1, 0, 0, 0,
			0, 1, 0, 1, 0, 0,
			0, 0, 1, 0, 0, 0,
			0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 1);
		SKalmanInput.matMeasurement.setTo(Scalar(0));
		SKalmanInput.KF.statePre.at<float>(0) = SKalmanInput.SKalmanTrackingLineBefore.fXcenter;
		SKalmanInput.KF.statePre.at<float>(1) = SKalmanInput.SKalmanTrackingLineBefore.fXderiv;
		SKalmanInput.KF.statePre.at<float>(2) = 0;
		SKalmanInput.KF.statePre.at<float>(3) = 0;
		SKalmanInput.KF.statePre.at<float>(4) = 0;
		SKalmanInput.KF.statePre.at<float>(5) = 0;
		setIdentity(SKalmanInput.KF.measurementMatrix);
		setIdentity(SKalmanInput.KF.processNoiseCov, Scalar::all(1e-4));
		setIdentity(SKalmanInput.KF.measurementNoiseCov, Scalar::all(1e-1));
		setIdentity(SKalmanInput.KF.errorCovPost, Scalar::all(0.1));
	}else if (nflag == KALMAN_RIGHT){
		SKalmanInput.KF.transitionMatrix = *(Mat_<float>(6, 6) <<
			1, 0, 1, 0, 0, 0,
			0, 1, 0, 1, 0, 0,
			0, 0, 1, 0, 0, 0,
			0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 1);
		SKalmanInput.matMeasurement.setTo(Scalar(0));
		SKalmanInput.KF.statePre.at<float>(0) = SKalmanInput.SKalmanTrackingLineBefore.fXcenter;
		SKalmanInput.KF.statePre.at<float>(1) = SKalmanInput.SKalmanTrackingLineBefore.fXderiv;
		SKalmanInput.KF.statePre.at<float>(2) = 0;
		SKalmanInput.KF.statePre.at<float>(3) = 0;
		SKalmanInput.KF.statePre.at<float>(4) = 0;
		SKalmanInput.KF.statePre.at<float>(5) = 0;
		setIdentity(SKalmanInput.KF.measurementMatrix);
		setIdentity(SKalmanInput.KF.processNoiseCov, Scalar::all(1e-4));
		setIdentity(SKalmanInput.KF.measurementNoiseCov, Scalar::all(1e-1));
		setIdentity(SKalmanInput.KF.errorCovPost, Scalar::all(0.1));
	}
	
}
void CMultiROILaneDetection::TrackingStageGround(EROINUMBER nflag){
	SLine SLineLeft;
	SLine SLineRight;
	//if ((nflag == LEFT_ROI2) || (nflag == LEFT_ROI3))
	//{
		if (!m_lanesGroundResult[nflag].empty())
		{
			SLineLeft = m_lanesGroundResult[nflag][0];
			SLineLeft.fGroundHeight = m_lanesGroundResult[nflag][0].ptStartLine.y
				- m_lanesGroundResult[nflag][0].ptEndLine.y;
			SLineLeft.fXcenter = (m_lanesGroundResult[nflag][0].ptStartLine.x + m_lanesGroundResult[nflag][0].ptEndLine.x) / 2;
			SLineLeft.fXderiv = (m_lanesGroundResult[nflag][0].ptStartLine.x - m_lanesGroundResult[nflag][0].ptEndLine.x)
				/ SLineLeft.fGroundHeight;

			SLineLeft.ptStartLine.x = (SLineLeft.ptStartLine.y - m_sCameraInfo.fGroundTop)*SLineLeft.fXderiv
				+ SLineLeft.ptStartLine.x;
			SLineLeft.ptStartLine.y = m_sCameraInfo.fGroundTop;

			SLineLeft.ptEndLine.x = (SLineLeft.ptEndLine.y - m_sCameraInfo.fGroundBottom)*SLineLeft.fXderiv
				+ SLineLeft.ptEndLine.x;
			SLineLeft.ptEndLine.y = m_sCameraInfo.fGroundBottom;

			if ((nflag == LEFT_ROI2) || (nflag == LEFT_ROI3)){
				m_leftGroundTracking.push_back(SLineLeft);
				if (m_leftGroundTracking.size() > MOVING_AVERAGE_NUM){
					m_iterGroundLeft = m_leftGroundTracking.begin();
					m_leftGroundTracking.erase(m_iterGroundLeft);
				}
			}
			else if ((nflag == RIGHT_ROI2) || (nflag == RIGHT_ROI3)){
				m_rightGroundTracking.push_back(SLineLeft);
				if (m_rightGroundTracking.size() > MOVING_AVERAGE_NUM){
					m_iterGroundRight = m_rightGroundTracking.begin();
					m_rightGroundTracking.erase(m_iterGroundRight);
				}
			}
			
		}

//	}
	/*if ((nflag == RIGHT_ROI2) || (nflag == RIGHT_ROI3))
	{
		if (!m_lanesGroundResult[nflag].empty())
		{
			SLineRight = m_lanesGroundResult[nflag][0];
			SLineRight.fGroundHeight = m_lanesGroundResult[nflag][0].ptStartLine.y
				- m_lanesGroundResult[nflag][0].ptEndLine.y;
			SLineRight.fXcenter = (m_lanesGroundResult[nflag][0].ptStartLine.x + m_lanesGroundResult[nflag][0].ptEndLine.x) / 2;
			SLineRight.fXderiv = (m_lanesGroundResult[nflag][0].ptStartLine.x - m_lanesGroundResult[nflag][0].ptEndLine.x)
				/ SLineRight.fGroundHeight;

			SLineRight.ptStartLine.x = (SLineRight.ptStartLine.y - m_sCameraInfo.fGroundTop)*SLineRight.fXderiv
				+ SLineRight.ptStartLine.x;
			SLineRight.ptStartLine.y = m_sCameraInfo.fGroundTop;

			SLineRight.ptEndLine.x = (SLineRight.ptEndLine.y - m_sCameraInfo.fGroundBottom)*SLineRight.fXderiv
				+ SLineRight.ptEndLine.x;
			SLineRight.ptEndLine.y = m_sCameraInfo.fGroundBottom;

			m_rightGroundTracking.push_back(SLineRight);
			if (m_rightGroundTracking.size() > MOVING_AVERAGE_NUM){
				m_iterGroundRight = m_rightGroundTracking.begin();
				m_rightGroundTracking.erase(m_iterGroundLeft);
			}

		}
	}*/
}

void CMultiROILaneDetection::ClearDetectionResult(){
	m_bLeftDraw = false;
	m_bRightDraw = false;

	nLeftCnt = 0;
	m_leftTracking.clear();
	m_leftGroundTracking.clear();
	m_bTracking[LEFT_ROI2] = false;
	m_bTracking[LEFT_ROI3] = false;
	m_SKalmanLeftLane.cntNum = 0;

	nLeftCnt = 0;
	m_leftTracking.clear();
	m_leftGroundTracking.clear();
	m_bTracking[LEFT_ROI2] = false;
	m_bTracking[LEFT_ROI3] = false;
	m_SKalmanRightLane.cntNum = 0;
}
// my function
void SetFrameName(char* szDataName, char* szDataDir,int nFrameNum){
	strcpy(szDataName,szDataDir);
	char szNumPng[10];
	sprintf(szNumPng,"%05d.png",nFrameNum);
	strcat(szDataName,szNumPng);
}
void SetFrameNameBMP(char* szDataName, char* szDataDir,int nFrameNum){
	strcpy(szDataName,szDataDir);
	char szNumPng[10];
	sprintf(szNumPng,"%05d.BMP",nFrameNum);
	strcat(szDataName,szNumPng);
}

void ShowImageNormalize( const char str[],const Mat &pmat){
	Mat mat = pmat.clone();
	ScaleMat(mat,mat);
	namedWindow(str);
	imshow(str,mat);
}

void ScaleMat(const Mat &inMat, Mat &outMat){
	inMat.convertTo(outMat,inMat.type());
	double min;
	minMaxLoc(inMat,&min);
	subtract(inMat,min,outMat);
	double max;
	minMaxLoc(outMat,NULL,&max);
	convertScaleAbs(outMat,outMat,255.0/max);
}
void ShowResults(CMultiROILaneDetection &obj, EROINUMBER nflag){
	int nNum = nflag;
	//Draw lane IPM
	/*for(unsigned int i = 0; i < obj.m_lanes[nflag].size(); i++)
		line(obj.m_imgIPM[nflag],
		Point(obj.m_lanes[nflag][i].ptStartLine.x,obj.m_lanes[nflag][i].ptStartLine.y),
		Point(obj.m_lanes[nflag][i].ptEndLine.x,obj.m_lanes[nflag][i].ptEndLine.y),
		Scalar(0,0,255),2);*/

	rectangle(obj.m_imgResizeOrigin,obj.m_sRoiInfo[nflag].ptRoi,obj.m_sRoiInfo[nflag].ptRoiEnd,Scalar(255,0,0),2);
	//Draw lane Orignin
	/*for(unsigned int i=0; i< obj.m_lanesResult[nflag].size();i++)
		line(obj.m_imgResizeOrigin,
		Point((int)obj.m_lanesResult[nflag][i].ptStartLine.x,(int)obj.m_lanesResult[nflag][i].ptStartLine.y),
		Point((int)obj.m_lanesResult[nflag][i].ptEndLine.x,(int)obj.m_lanesResult[nflag][i].ptEndLine.y),
		Scalar(0,0,255),2);*/
	char strImg[20] ;
	//sprintf(strImgIpm,)
	sprintf(strImg,"IPM%d",nNum);
	imshow(strImg,obj.m_imgIPM[nflag]);
	sprintf(strImg,"FN%d",nNum);
	ShowImageNormalize(strImg,obj.m_ipmFiltered[nflag]);
	sprintf(strImg,"FT%d",nNum);
	ShowImageNormalize(strImg,obj.m_filteredThreshold[nflag]);
	sprintf(strImg,"Filt%d",nNum);
	imshow(strImg,obj.m_ipmFiltered[nflag]);
	

}

vector<Point> LineDivNum(Point ptTop, Point ptBottom, int num){
	vector<Point> vecResult;
	//num--;
	int nWidth = ptTop.x - ptBottom.x;
	int nWidthDiff = nWidth / (num-1);
	int nHeight = ptTop.y - ptBottom.y;
	int nHeightDiff = nHeight / (num - 1);
	for (int i = 0; i < num; i++){
		Point ptTemp;
		//cout << nWidthDiff << endl;
		ptTemp.x = ptTop.x - nWidthDiff*i;
		ptTemp.y = ptTop.y - nHeightDiff*i;
		vecResult.push_back(ptTemp);
	}
	return vecResult;

}

bool MovingAverageFilter(CMultiROILaneDetection &obj, EROINUMBER nflag){
	SLine sLineLeft;
	SLine sLineRight;
	if ((nflag == LEFT_ROI2) || (nflag == LEFT_ROI3))
	{
		if (!obj.m_lanesResult[nflag].empty())
		{
			int nTop = obj.m_sRoiInfo[LEFT_ROI2].nTop;
			int nBottom = obj.m_sRoiInfo[LEFT_ROI3].nBottom;
			Point_<double> ptCenter = Point_<double>((obj.m_lanesResult[nflag][0].ptStartLine.x + obj.m_lanesResult[nflag][0].ptEndLine.x) / 2, (obj.m_lanesResult[nflag][0].ptStartLine.y + obj.m_lanesResult[nflag][0].ptEndLine.y) / 2);

			float fYdv = obj.m_lanesResult[nflag][0].ptEndLine.y - obj.m_lanesResult[nflag][0].ptStartLine.y;
			float fXdv = obj.m_lanesResult[nflag][0].ptEndLine.x - obj.m_lanesResult[nflag][0].ptStartLine.x;
			//float fAtan = atan2(fXdv , fYdv);
			float fAtan = fXdv / fYdv;

			sLineLeft.ptStartLine.x = ptCenter.x + (nTop - ptCenter.y)*fAtan;
			sLineLeft.ptStartLine.y = nTop;
			sLineLeft.ptEndLine.x = ptCenter.x + (nBottom - ptCenter.y)*fAtan;
			sLineLeft.ptEndLine.y = nBottom;
			/*obj.m_lanesResult[nflag].front().ptStartLine.x = ptCenter.x + (nTop - ptCenter.y)*fAtan;
			obj.m_lanesResult[nflag].front().ptStartLine.y = nTop;
			obj.m_lanesResult[nflag].front().ptEndLine.x = ptCenter.x + (nBottom - ptCenter.y)*fAtan;
			obj.m_lanesResult[nflag].front().ptEndLine.y = nBottom;*/

			obj.m_leftTracking.push_back(sLineLeft);
			if (obj.m_leftTracking.size() > MOVING_AVERAGE_NUM){
				obj.m_iterLeft = obj.m_leftTracking.begin();
				obj.m_leftTracking.erase(obj.m_iterLeft);
			}
		}
		/*else if (0!=obj.m_leftTracking.size()){
			cout << "erase left		" <<nflag<< endl;
			obj.m_iterLeft = obj.m_leftTracking.begin();
			obj.m_leftTracking.erase(obj.m_iterLeft);
		}
		cout <<"size : "<< obj.m_leftTracking.size() << endl;*/
		

		
		
	}
	if ((nflag == RIGHT_ROI2) || (nflag == RIGHT_ROI3))
		if (!obj.m_lanesResult[nflag].empty())
	{
		int nTop = obj.m_sRoiInfo[RIGHT_ROI2].nTop;
		int nBottom = obj.m_sRoiInfo[RIGHT_ROI3].nBottom;
		Point_<double> ptCenter = Point_<double>((obj.m_lanesResult[nflag][0].ptStartLine.x + obj.m_lanesResult[nflag][0].ptEndLine.x) / 2, (obj.m_lanesResult[nflag][0].ptStartLine.y + obj.m_lanesResult[nflag][0].ptEndLine.y) / 2);

		float fYdv = obj.m_lanesResult[nflag][0].ptEndLine.y - obj.m_lanesResult[nflag][0].ptStartLine.y;
		float fXdv = obj.m_lanesResult[nflag][0].ptEndLine.x - obj.m_lanesResult[nflag][0].ptStartLine.x;
		//float fAtan = atan2(fXdv , fYdv);
		float fAtan = fXdv / fYdv;

		sLineRight.ptStartLine.x = ptCenter.x + (nTop - ptCenter.y)*fAtan;
		sLineRight.ptStartLine.y = nTop;
		sLineRight.ptEndLine.x = ptCenter.x + (nBottom - ptCenter.y)*fAtan;
		sLineRight.ptEndLine.y = nBottom;
		/*obj.m_lanesResult[nflag].front().ptStartLine.x = ptCenter.x + (nTop - ptCenter.y)*fAtan;
		obj.m_lanesResult[nflag].front().ptStartLine.y = nTop;
		obj.m_lanesResult[nflag].front().ptEndLine.x = ptCenter.x + (nBottom - ptCenter.y)*fAtan;
		obj.m_lanesResult[nflag].front().ptEndLine.y = nBottom;*/

		obj.m_rightTracking.push_back(sLineRight);
		if (obj.m_rightTracking.size() > MOVING_AVERAGE_NUM){
			obj.m_iterRight = obj.m_rightTracking.begin();
			obj.m_rightTracking.erase(obj.m_iterRight);
		}

	}


	
	

	/*for (int i = 0; i < obj.m_leftTracking.size(); i++){
		cout << obj.m_leftTracking[i].ptStartLine << endl;
		cout << obj.m_leftTracking[i].ptEndLine << endl;
	}*/
	/*obj.m_lanes[nflag].clear();
	obj.m_laneScore[nflag].clear();
	obj.m_lanesResult[nflag].clear();*/
	
	return true;
}

bool TrackingStageGround(CMultiROILaneDetection &obj, EROINUMBER nflag){
	SLine sLineLeft;
	SLine sLineRight;
	if ((nflag == LEFT_ROI2) || (nflag == LEFT_ROI3))
	{
		if (!obj.m_lanesGroundResult[nflag].empty())
		{
			sLineLeft = obj.m_lanesGroundResult[nflag][0];
			sLineLeft.fGroundHeight = obj.m_lanesGroundResult[nflag][0].ptStartLine.y
				- obj.m_lanesGroundResult[nflag][0].ptEndLine.y;
			sLineLeft.fXcenter = (obj.m_lanesGroundResult[nflag][0].ptStartLine.x + obj.m_lanesGroundResult[nflag][0].ptEndLine.x) / 2;
			sLineLeft.fXderiv = (obj.m_lanesGroundResult[nflag][0].ptStartLine.x - obj.m_lanesGroundResult[nflag][0].ptEndLine.x)
				/ sLineLeft.fGroundHeight;
			
			sLineLeft.ptStartLine.x = (sLineLeft.ptStartLine.y - obj.m_sCameraInfo.fGroundTop)*sLineLeft.fXderiv
				+ sLineLeft.ptStartLine.x;
			sLineLeft.ptStartLine.y = obj.m_sCameraInfo.fGroundTop;
			
			sLineLeft.ptEndLine.x = (sLineLeft.ptEndLine.y - obj.m_sCameraInfo.fGroundBottom)*sLineLeft.fXderiv
				+ sLineLeft.ptEndLine.x;
			sLineLeft.ptEndLine.y = obj.m_sCameraInfo.fGroundBottom;

			//obj.m_kalmanLeftLane.statePost.a

			obj.m_leftGroundTracking.push_back(sLineLeft);
			if (obj.m_leftGroundTracking.size() > MOVING_AVERAGE_NUM){
				obj.m_iterGroundLeft = obj.m_leftGroundTracking.begin();
				obj.m_leftGroundTracking.erase(obj.m_iterGroundLeft);
			}
		}

	}
	if ((nflag == RIGHT_ROI2) || (nflag == RIGHT_ROI3))
		if (!obj.m_lanesGroundResult[nflag].empty())
		{
			int nTop = obj.m_sRoiInfo[RIGHT_ROI2].nTop;
			int nBottom = obj.m_sRoiInfo[RIGHT_ROI3].nBottom;
			Point_<double> ptCenter = Point_<double>((obj.m_lanesGroundResult[nflag][0].ptStartLine.x + obj.m_lanesGroundResult[nflag][0].ptEndLine.x) / 2, (obj.m_lanesGroundResult[nflag][0].ptStartLine.y + obj.m_lanesGroundResult[nflag][0].ptEndLine.y) / 2);

			float fYdv = obj.m_lanesGroundResult[nflag][0].ptEndLine.y - obj.m_lanesGroundResult[nflag][0].ptStartLine.y;
			float fXdv = obj.m_lanesGroundResult[nflag][0].ptEndLine.x - obj.m_lanesGroundResult[nflag][0].ptStartLine.x;
			//float fAtan = atan2(fXdv , fYdv);
			float fAtan = fXdv / fYdv;

			sLineRight.ptStartLine.x = ptCenter.x + (nTop - ptCenter.y)*fAtan;
			sLineRight.ptStartLine.y = nTop;
			sLineRight.ptEndLine.x = ptCenter.x + (nBottom - ptCenter.y)*fAtan;
			sLineRight.ptEndLine.y = nBottom;
			/*obj.m_lanesResult[nflag].front().ptStartLine.x = ptCenter.x + (nTop - ptCenter.y)*fAtan;
			obj.m_lanesResult[nflag].front().ptStartLine.y = nTop;
			obj.m_lanesResult[nflag].front().ptEndLine.x = ptCenter.x + (nBottom - ptCenter.y)*fAtan;
			obj.m_lanesResult[nflag].front().ptEndLine.y = nBottom;*/

			obj.m_rightGroundTracking.push_back(sLineRight);
			if (obj.m_rightGroundTracking.size() > MOVING_AVERAGE_NUM){
				obj.m_iterGroundLeft = obj.m_rightGroundTracking.begin();
				obj.m_rightGroundTracking.erase(obj.m_iterGroundLeft);
			}

		}





	/*for (int i = 0; i < obj.m_leftTracking.size(); i++){
	cout << obj.m_leftTracking[i].ptStartLine << endl;
	cout << obj.m_leftTracking[i].ptEndLine << endl;
	}*/
	/*obj.m_lanes[nflag].clear();
	obj.m_laneScore[nflag].clear();
	obj.m_lanesResult[nflag].clear();*/

	return true;
}
void LoadExtractedPoint(FILE* fp, Point_<double>& ptUpLeft, Point_<double>& ptMidLeft, Point_<double>& ptUpRight, Point_<double>& ptMidRight){
	char szTemp[10];
	int nScFrameNum;
	fscanf(fp, "%s", &szTemp);//#
	fscanf(fp, "%d", &nScFrameNum);//#
	fscanf(fp, "%lf", &ptUpLeft.x); fscanf(fp, "%lf", &ptUpLeft.y);  fscanf(fp, "%lf", &ptMidLeft.x);  fscanf(fp, "%lf", &ptMidLeft.y); //#
	fscanf(fp, "%lf", &ptUpRight.x); fscanf(fp, "%lf", &ptUpRight.y); fscanf(fp, "%lf", &ptMidRight.x); fscanf(fp, "%lf", &ptMidRight.y);//#

}
float CompareLineDiff(Point2d FixedStart, Point2d FixedEnd, Point2d GroundStart, Point2d GroundEnd){
	float fHeight = abs(FixedStart.y - FixedEnd.y);
	float fDiffStart = abs(FixedStart.x - GroundStart.x);
	float fDiffEnd = abs(FixedEnd.x - GroundEnd.x);
	float fScore = (fDiffStart + fDiffEnd) / fHeight;
	return fScore;
}
void EvaluationFunc(CMultiROILaneDetection &obj, SEvaluation &structEvaluation, 
	SWorldLane GroundLeft, SWorldLane GroundRight, 
	SWorldLane FindLeft, SWorldLane FindRight){
	int nTop = GroundLeft.ptUvStartLine.y;
	int nBottom = GroundLeft.ptUvEndLine.y;
	Point2i LeftText = Point2i(obj.m_imgResizeOrigin.cols / 4, GroundLeft.ptUvStartLine.y-40);
	Point2i RightText = Point2i(obj.m_imgResizeOrigin.cols * 3 / 4, GroundLeft.ptUvStartLine.y - 40);

	Point2d GroundLeftCenter;
	Point2d GroundRightCenter;
	
	Point2d FindLeftCenter;
	Point2d FindRightCenter;
	double FindLeftAngle;
	double FindRightAngle;
	Point2d FixedLeftStart;
	Point2d FixedLeftEnd;
	Point2d FixedRightStart;
	Point2d FixedRightEnd;
	
	//GroundLeftCenter.x = (GroundLeft.ptUvStartLine.x + GroundLeft.ptUvEndLine.x) / 2;
	//GroundLeftCenter.y = (GroundLeft.ptUvStartLine.y + GroundLeft.ptUvEndLine.y) / 2;

	FindLeftCenter.x = (FindLeft.ptUvStartLine.x + FindLeft.ptUvEndLine.x) / 2;
	FindLeftCenter.y = (FindLeft.ptUvStartLine.y + FindLeft.ptUvEndLine.y) / 2;
	FindLeftAngle = (FindLeft.ptUvEndLine.x - FindLeft.ptUvStartLine.x)
		/ (FindLeft.ptUvEndLine.y - FindLeft.ptUvStartLine.y);
	FixedLeftStart.x = FindLeftCenter.x + (nTop - FindLeftCenter.y)*FindLeftAngle;
	FixedLeftStart.y = nTop;
	FixedLeftEnd.x = FindLeftCenter.x + (nBottom - FindLeftCenter.y)*FindLeftAngle;
	FixedLeftEnd.y = nBottom;

	FindRightCenter.x = (FindRight.ptUvStartLine.x + FindRight.ptUvEndLine.x) / 2;
	FindRightCenter.y = (FindRight.ptUvStartLine.y + FindRight.ptUvEndLine.y) / 2;
	FindRightAngle = (FindRight.ptUvEndLine.x - FindRight.ptUvStartLine.x)
		/ (FindRight.ptUvEndLine.y - FindRight.ptUvStartLine.y);
	FixedRightStart.x = FindRightCenter.x + (nTop - FindRightCenter.y)*FindRightAngle;
	FixedRightStart.y = nTop;
	FixedRightEnd.x = FindRightCenter.x + (nBottom - FindRightCenter.y)*FindRightAngle;
	FixedRightEnd.y = nBottom;

	
	char szTruePositive[20] = "TruePositive";
	char szFalsePositive[20] = "FalsePositive";
	char szFalseNegative[20] = "FalseNegative";
	char szTrueNegative[20] = "TrueNegative";

	float fLeftScore;
	float fRightScore;
	float fStandard = COMPARE_STANDARD;

	if ((GroundLeft.ptUvStartLine.x != EMPTY) && (GroundLeft.ptUvEndLine.x != EMPTY)){
		if (FindLeft.ptUvStartLine.x == EMPTY){
			structEvaluation.LeftFN++;
			//cout << "Left FN" << endl;
			putText(obj.m_imgResizeOrigin, szFalseNegative, LeftText,
				FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1, 8, false);
		}
		else{
			fLeftScore = CompareLineDiff(FixedLeftStart, FixedLeftEnd, GroundLeft.ptUvStartLine, GroundLeft.ptUvEndLine);
			//cout << "fLeftScore : " << fLeftScore << endl;
			if (fLeftScore < fStandard){
				structEvaluation.LeftTP++;
				//cout << "Left TP" << endl;
				putText(obj.m_imgResizeOrigin, szTruePositive, LeftText,
					FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 1, 8, false);
			}
			else{
				structEvaluation.LeftFP++;
				//cout << "Left FP" << endl;
				putText(obj.m_imgResizeOrigin, szFalsePositive, LeftText,
					FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1, 8, false);
			}
		}
		structEvaluation.nLeftGroundTruth++;
	}
	else{
		if (FindLeft.ptUvStartLine.x == EMPTY){
			structEvaluation.LeftTN++;
			//cout << "Left TN" << endl;
			putText(obj.m_imgResizeOrigin, szTrueNegative, LeftText,
				FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 1, 8, false);
		}
		else
		{
			structEvaluation.LeftFP++;
			//cout << "Left FP" << endl;
			putText(obj.m_imgResizeOrigin, szFalsePositive, LeftText,
				FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1, 8, false);
		}
	}


	if ((GroundRight.ptUvStartLine.x != EMPTY) && (GroundRight.ptUvEndLine.x != EMPTY)){
		if (FindRight.ptUvStartLine.x == EMPTY){
			structEvaluation.RightFN++;
			//cout << "Right FN" << endl;
			putText(obj.m_imgResizeOrigin, szFalseNegative, RightText,
				FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1, 8, false);
		}
		else{
			fRightScore = CompareLineDiff(FixedRightStart, FixedRightEnd, GroundRight.ptUvStartLine, GroundRight.ptUvEndLine);
			//cout << "fRightScore : " << fRightScore << endl;
			if (fRightScore < fStandard){
				structEvaluation.RightTP++;
				//cout << "Right TP" << endl;
				putText(obj.m_imgResizeOrigin, szTruePositive, RightText,
					FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 1, 8, false);
			}
			else{
				structEvaluation.RightFP++;
				//cout << "Right FP" << endl;
				putText(obj.m_imgResizeOrigin, szFalsePositive, RightText,
					FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1, 8, false);
			}
		}
		structEvaluation.nRightGroundTruth++;
	}
	else{
		if (FindRight.ptUvStartLine.x == EMPTY){
			structEvaluation.RightTN++;
			//cout << "Right TN" << endl;
			putText(obj.m_imgResizeOrigin, szTrueNegative, RightText,
				FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 1, 8, false);
		}
		else
		{
			structEvaluation.RightFP++;
			//cout << "Right FP" << endl;
			putText(obj.m_imgResizeOrigin, szFalsePositive, RightText,
				FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1, 8, false);
		}
	}

	

	//fRightScore = CompareLineDiff(FixedRightStart, FixedRightEnd, GroundRight.ptUvStartLine, GroundRight.ptUvEndLine);


	if (FindLeft.ptUvStartLine.x != EMPTY){
		line(obj.m_imgResizeOrigin, FixedLeftStart, FixedLeftEnd, Scalar(255, 255, 0), 2);
	}
	if (FindRight.ptUvStartLine.x != EMPTY){
		line(obj.m_imgResizeOrigin, FixedRightStart, FixedRightEnd, Scalar(255, 255, 0), 2);
	}
	
	
	structEvaluation.nTotalFrame++;
}///end
//
//float fYdv = obj.m_lanesResult[nflag][0].ptEndLine.y - obj.m_lanesResult[nflag][0].ptStartLine.y;
//float fXdv = obj.m_lanesResult[nflag][0].ptEndLine.x - obj.m_lanesResult[nflag][0].ptStartLine.x;
////float fAtan = atan2(fXdv , fYdv);
//float fAtan = fXdv / fYdv;
//
//sLineRight.ptStartLine.x = ptCenter.x + (nTop - ptCenter.y)*fAtan;
//sLineRight.ptStartLine.y = nTop;
//sLineRight.ptEndLine.x = ptCenter.x + (nBottom - ptCenter.y)*fAtan;
//sLineRight.ptEndLine.y = nBottom;

void DifferentialImgProcess(vector<Mat> &vecImgDiff,Mat origin,vector<Point> &vecRoiBottom){
	Mat matSumDiff;
	matSumDiff.create(vecImgDiff[0].rows, 1, CV_32FC1);
	//matSumDiff.zeros(vecImgDiff[0].rows, 1, CV_32FC1);
	vector<Mat> vecSumDiff;
	Mat matColSum = Mat::zeros(vecImgDiff[0].rows, 1, CV_32FC1);
	//matColSum.create(vecImgDiff[0].rows, 1, CV_32FC1);
	//matColSum.zeros(vecImgDiff[0].rows, 1, CV_32FC1);
	for (int i = 0; i < vecImgDiff.size(); i++){
 		reduce(vecImgDiff[i], matSumDiff, 1, CV_REDUCE_SUM);
		vecSumDiff.push_back(matSumDiff.clone());
		matColSum += matSumDiff;

		imshow("diff", vecImgDiff[i]);
		printf("diff #%d\n", i);
		//waitKey(0);
	}
	matColSum /= vecImgDiff.size();
	/*double tMax = MAXCOMP;
	float *pMatColSumData = (float*)matColSum.data;
	for (int i = 0; i < matColSum.rows; i++){
		if (tMax < pMatColSumData[i])
			tMax = pMatColSumData[i];
	}
	matColSum /= tMax/vecImgDiff[0].cols;
	printf("tMax = %f\n", tMax);*/
	Mat matDrawDiffSum = Mat::zeros(vecImgDiff[0].size(), CV_32FC3);
	for (int i = 0; i < matDrawDiffSum.rows; i++){
		//if ((matColSum.at<float>(i) / vecImgDiff[0].cols)<0.1)
		if ((matColSum.at<float>(i)) < 2.5)
		{
			line(origin, Point(0, i), Point(matColSum.at<float>(i), i), Scalar(255, 0, 0), 1);
			if (i>matColSum.rows/2)
				vecRoiBottom.push_back(Point(matColSum.at<float>(i), i));
		}			
		else
			line(origin, Point(0, i), Point(matColSum.at<float>(i), i), Scalar(0, 255, 0), 1);
	}
	imshow("matDrawDiffSum", origin);
	waitKey(0);

}