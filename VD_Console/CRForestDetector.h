/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRForest.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace cv;
#define IMAGE_WIDTH 1280
#define FIRST_ROI_WIDTH 755
#define ANGLE_THRESH 5
#define TRACKING_MARGIN_RATIO 1
#define SCALE_DIST 1

//////cvlab_1 
//#define MARGIN_RATIO 0.8
//cvlab_8 
#define MARGIN_RATIO 0.8

////cvlab_2
//#define MARGIN_RATIO 0.9

//~cvlab_6
#define PIXEL_CNT 25

////cvlab_7
//#define PIXEL_CNT 100
#define THRESHOLD_RATIO 1

#define SIZE_RATIO 0.1
//#define WIDTH_DIFFERENCE 30

struct sMaxPtInfo
{
	Point2i ptMax;
	double dMaxValue;
};

struct sMaxPts
{
	Point2i ptMax;
	double dMaxValue;
	Rect rectMax;
};


struct sPointInfo
{
	int nX;
	int nY;
	int nIntensity;
	float fScale;
};

//struct stVector{
//	void operator() (typedef Element)
//
//};
struct sImgTemp{
	int nIdx;
	vector<IplImage*> imgScaled;
};

struct sDetectLocal{
	float scale;
	Mat mROI;
};


class CRForestDetector {


private:
	//CvCapture* capture = 0;
	void detectColor(IplImage *img, std::vector<IplImage*>& imgDetect, std::vector<float>& ratios);
	void detectColor_Revised(std::vector<IplImage *>& img, IplImage* & imgDetect);
	void detectColor_Revised_Mat(const vector<Mat>& vImg, Mat& imgDetect);
	void buildHoughMap(const vector<Mat>& vImg, Mat& imgDetect, Mat& imgCount);

	const CRForest* crForest;
	int width;
	int height;
	vector<float> vec_RoiRowRatio;
	vector<int> vec_RoiY;
	vector<Size> vec_RoiSize;
	vector<Size> vec_ORoiSize;
	vector<Size> vec_SizeROI;
	vector<Point> vec_PointROI;


	
public:
	Point2i ptMain;

	VideoWriter oVideoWriter;
	Mat m_InputFrame;

	int nPatchWidth;
	int nPatchHeight;
	CvRect recRoi;

	Point2i Roi0_tl;
	vector<Point2i> vec_Roi_tl;
	vector<Point2i> vec_ORoi_tl;
	vector<Rect_<int>> vec_rectROI;
	vector<Rect> vecValid;

	int cnt;
	int nStrideX;
	int nStrideY;
	//FILE* pixelSize;
	//VideoCapture m_CapVideo;
	vector<int> nMaxCount;
	vector<Mat> vec_mFeature;


	// Constructor
	CRForestDetector(const CRForest* pRF, int w, int h) : crForest(pRF), width(w), height(h)  {}

	//mat
	Mat mTotalImg;
	Mat mTmp, mRoi1;
	IplImage* imgTmp;
	IplImage* imgTotal;

	// Training image size
	CvSize sizeTrainImg;

	std::vector<Mat> vImgDetect;
	std::vector<sDetectLocal> sLocalROI;

	//initialize roi location
	void InitializeROI(CRForestDetector& crDetect);
	void InitializeROI_scale(vector<float> scale);

	//set scaled ROIs
	vector<IplImage* > CropNScaleRoi(IplImage* imgInput, vector<float> scale);
	vector<IplImage*> CropNScaleRoi_Reivised(IplImage* imgInput, vector<float> scale);
	vector<Mat> CropNScaleRoi_Reivised_Mat(Mat imgInput, vector<float> scale);
	vector<Mat> CropNScaleRoi_Reivised_Mat2(Mat imgInput, vector<float> scale);
	vector<Mat> CropNROI_scale(Mat imgInput, vector<float> scale);

	// detect multi scale
	void detectPyramid(IplImage *img, std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float>& ratios);
	void detectPyramid_Revised(IplImage *img, std::vector<IplImage*> & imgDetect, std::vector<float> scales);
	void detectPyramid_Revised2(CRForestDetector& crDetect, std::vector<IplImage *> img, std::vector<IplImage*> & imgDetect, std::vector<float> scales, std::vector<CvRect> vecRoi);
	//void detectPyramid_Revised3(CRForestDetector& crDetect, IplImage* imgInput ,std::vector<IplImage *> img, std::vector<IplImage*> & imgDetect, std::vector<float> scales, std::vector<CvRect> vecRoi);
	void detectPyramid_Revised_Mat(CRForestDetector& crDetect, std::vector<Mat>& vec_ROIs, std::vector<float>& scale/*, std::vector<Mat> & imgDetect*/);
	void detectInPyramid( std::vector<float>& scale/*, std::vector<Mat> & imgDetect*/);
	void detectMultiROI(std::vector<Rect>& vecValid);



	// Get/Set functions
	unsigned int GetNumCenter() const {return crForest->GetNumCenter();}

	//etc function
	int StandardVehicleWidth(Point2i pBottomLeft);
	void VerifyDetection(const vector<Rect_<int>>& vecDetectVehicle, vector<Rect_<int>>& vecVerifiedVehicle);
	void VerifyDetection_check(const vector<sMaxPts>& vecMaxRect, vector<Rect_<int>>& vecVerifiedVehicle);
	//bool CheckContourSize(Mat mCheck, int nTrainWidth, int nTrainHeight);
	bool CheckContourSize(Rect rectCheck, int nTrainWidth, int nTrainHeight);
	bool CheckContourSize_Mat(Mat mCheck);
	void Normalization(Mat& mHoughSrc, Mat& mHoughDst );

	bool LoadTestVideo_Mat(CRForestDetector& crDetect, const int& nDataIdx, float scale0);
	bool LoadTestVideo_oneROI();

	void show(vector<Mat> mat_hough, vector<Mat> mat_threshold);
	int contourThreshold(int threshold);


	/*bool SetTestVideo_Mat(CRForestDetector& crDetect);
*/
};
