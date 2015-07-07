/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch

// You may use, copy, reproduce, and distribute this Software for any
// non-commercial purpose, subject to the restrictions of the
// Microsoft Research Shared Source license agreement ("MSR-SSLA").
// Some purposes which can be non-commercial are teaching, academic
// research, public demonstrations and personal experimentation. You
// may also distribute this Software with books or other teaching
// materials, or publish the Software on websites, that are intended
// to teach the use of the Software for academic or other
// non-commercial purposes.
// You may not use or distribute this Software or any derivative works
// in any form for commercial purposes. Examples of commercial
// purposes would be running business operations, licensing, leasing,
// or selling the Software, distributing the Software for use with
// commercial products, using the Software in the creation or use of
// commercial products or any other activity which purpose is to
// procure a commercial gain to you or others.
// If the Software includes source code or data, you may create
// derivative works of such portions of the Software and distribute
// the modified Software for non-commercial purposes, as provided
// herein.

// THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO
// EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT
// LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A
// PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR
// ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR
// NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL
// FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST
// PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR
// DERIVATIVE WORKS.

// NEITHER MICROSOFT NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE
// LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS MSR-SSLA,
// INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL
// DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT
// LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF
// LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE
// WORKS.

// When using this software, please acknowledge the effort that
// went into development by referencing the paper:
//
// Gall J. and Lempitsky V., Class-Specific Hough Forests for
// Object Detection, IEEE Conference on Computer Vision and Pattern
// Recognition (CVPR'09), 2009.

// Note that this is not the original software that was used for
// the paper mentioned above. It is a re-implementation for Linux.

*/




#include <stdexcept>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

#include <direct.h>
#include<stdio.h>
#include<sys/stat.h>
#include<time.h>

#include "kalmanFilter.h"
#include "CRForestDetector.h"
#define PATH_SEP "\\"
using namespace std;

// Path to trees
string treepath;
// Number of trees
int ntrees;
// Patch width
int p_width;
// Patch height
int p_height;
//Path to images
string impath;
// File with names of images
string imfiles;
// Extract features
bool xtrFeature;
// Scales
vector<float> scales;
// Ratio
vector<float> ratios;
// Output path
string outpath;
// scale factor for output image (default: 128)
int out_scale;
// Path to positive examples
string trainpospath;
// File with postive examples
string trainposfiles;
// Subset of positive images -1: all images
int subsamples_pos;
// Sample patches from pos. examples
unsigned int samples_pos;
// Path to positive examples
string trainnegpath;
// File with postive examples
string trainnegfiles;
// Subset of neg images -1: all images
int subsamples_neg;
// Samples from pos. examples
unsigned int samples_neg;
// Training image size
CvSize sizeTrain;
// offset for saving tree number
int off_tree;
// for Video capture
IplImage *imgTest = 0, *imgTestCopy = 0;
vector<IplImage*> vecImg;
//vector<Mat> vec_mImg;
vector<CvRect> vecRect;

Mat matOriginalImg;
CvCapture* capture = 0;
VideoCapture m_CapVideo;


//result of detection
vector<CvRect> vecDetection;

#define RESIZERATIO 1


//tracking
Rect_<int> rect_TROI;

Point ptVanishing;

typedef struct vecTracking{
	vector<Rect> vecBefore;
	vector<Rect> vecCandidate;
	vector<int> vecCount;
	vector<Point> vecCountPush;
	vector<Point2f> vecRtheta;
	vector<Point2f> vecAngle;
}Track_t;
float AngleTransform(const float &tempAngle, const int &scale){
	return (float)CV_PI / 180 * (tempAngle*scale);
}
#define ToRadian(degree) ((degree)*(CV_PI/180.0f))
#define ToDegree(radian) ((radian)*(180.0f/CV_PI))
float fAngle(int x, int y)
{
	return ToDegree(atan2f(y, x));
}
void kalmanTrackingStart(SKalman&, Rect&);
//void kalmanMultiTarget(Mat&, vector<Rect>&, vector<SKalman>&, float, int, int, int, int&, bool&, bool&);
void kalmanMultiTarget(Mat&, vector<Rect>&, Track_t&, vector<SKalman>&, float, int, int, int, int&, Rect&, vector<Rect>&, Mat& imgROImask, CRForestDetector& crDetect);
void clustering(vector<sMaxPts> sMaxPtRoi, int distance);
//CRForestDetector DetectVehicle;

// load config file for dataset
void loadConfig(const char* filename, int mode) {
	char buffer[400];
	ifstream in(filename);

	if (in.is_open()) {

		// Path to trees
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		treepath = buffer;
		// Number of trees
		in.getline(buffer, 400);
		in >> ntrees;
		in.getline(buffer, 400);
		// Patch width
		in.getline(buffer, 400);
		in >> p_width;
		//in >> nPatchWidth;
		in.getline(buffer, 400);
		// Patch height
		in.getline(buffer, 400);
		in >> p_height;
		//in >> nPatchHeight;
		in.getline(buffer, 400);
		// Path to images
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		impath = buffer;
		// File with names of images
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		imfiles = buffer;
		// Extract features
		in.getline(buffer, 400);
		in >> xtrFeature;
		in.getline(buffer, 400);
		// Scales
		in.getline(buffer, 400);
		int size;
		in >> size;
		scales.resize(size);
		for (int i = 0; i<size; ++i)
			in >> scales[i];
		in.getline(buffer, 400);
		// Ratio
		in.getline(buffer, 400);
		in >> size;
		ratios.resize(size);
		for (int i = 0; i<size; ++i)
			in >> ratios[i];
		in.getline(buffer, 400);
		// Output path
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		outpath = buffer;
		// Scale factor for output image (default: 128)
		in.getline(buffer, 400);
		in >> out_scale;
		in.getline(buffer, 400);
		// Path to positive examples
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		trainpospath = buffer;
		// File with postive examples
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		trainposfiles = buffer;
		// Subset of positive images -1: all images
		in.getline(buffer, 400);
		in >> subsamples_pos;
		in.getline(buffer, 400);
		// Samples from pos. examples
		in.getline(buffer, 400);
		in >> samples_pos;
		in.getline(buffer, 400);
		// Path to positive examples
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		trainnegpath = buffer;
		// File with postive examples
		in.getline(buffer, 400);
		in.getline(buffer, 400);
		trainnegfiles = buffer;
		// Subset of negative images -1: all images
		in.getline(buffer, 400);
		in >> subsamples_neg;
		in.getline(buffer, 400);
		// Samples from pos. examples
		in.getline(buffer, 400);
		in >> samples_neg;
		in.getline(buffer, 400);
		// Training img width
		in.getline(buffer, 400);
		in >> sizeTrain.width;
		in.getline(buffer, 400);
		// Training img height
		in.getline(buffer, 400);
		in >> sizeTrain.height;

	}
	else {
		cerr << "File not found " << filename << endl;
		exit(-1);
	}
	in.close();

	switch (mode) {
	case 0:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Training:         " << endl;
		cout << "Patches:          " << p_width << " " << p_height << endl;
		cout << "Train pos:        " << trainpospath << endl;
		cout << "                  " << trainposfiles << endl;
		cout << "                  " << subsamples_pos << " " << samples_pos << endl;
		cout << "Train neg:        " << trainnegpath << endl;
		cout << "                  " << trainnegfiles << endl;
		cout << "                  " << subsamples_neg << " " << samples_neg << endl;
		cout << "Trees:            " << ntrees << " " << off_tree << " " << treepath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;

	case 1:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Show:             " << endl;
		cout << "Trees:            " << ntrees << " " << treepath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;

	default:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Detection:        " << endl;
		cout << "Trees:            " << ntrees << " " << treepath << endl;
		cout << "Patches:          " << p_width << " " << p_height << endl;
		cout << "Images:           " << impath << endl;
		cout << "                  " << imfiles << endl;
		cout << "Image size:          " << sizeTrain.width << " " << sizeTrain.height << endl;
		cout << "Scales:           "; for (unsigned int i = 0; i<scales.size(); ++i) cout << scales[i] << " "; cout << endl;
		cout << "Ratios:           "; for (unsigned int i = 0; i<ratios.size(); ++i) cout << ratios[i] << " "; cout << endl;
		cout << "Extract Features: " << xtrFeature << endl;
		cout << "Output:           " << out_scale << " " << outpath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;
	}

}

// load test image filenames
void loadImFile(std::vector<string>& vFilenames) {

	char buffer[400];

	ifstream in(imfiles.c_str());
	if (in.is_open()) {

		unsigned int size;
		in >> size; //size = 10;
		in.getline(buffer, 400);
		vFilenames.resize(size);

		for (unsigned int i = 0; i<size; ++i) {
			in.getline(buffer, 400);
			vFilenames[i] = buffer;
		}

	}
	else {
		cerr << "File not found " << imfiles.c_str() << endl;
		exit(-1);
	}

	in.close();
}

//set test video 
bool SetTestVideo(){

	char cVideoPath[50];
	sprintf(cVideoPath, "%s", imfiles.c_str());

	capture = cvCaptureFromFile(cVideoPath);
	if (!capture){
		cout << "Error: Could not open the video file\n" << endl;
		exit(-1);
	}
}

bool SetTestVideo_Mat(){

	char cVideoPath[50];
	sprintf(cVideoPath, "%s", imfiles.c_str());

	if (!m_CapVideo.open(cVideoPath) && !m_CapVideo.isOpened()){
		printf("ERROR: Could not open the video file\n");
		return false;
	}
	return true;
}

bool LoadTestVideo(CRForestDetector& crDetect, CvCapture* capture, const int& nDataIdx){
	IplImage *imgTemp = 0, *imgTempClone = 0;
	vecImg.clear();

	if (!cvGrabFrame(capture)){
		cout << "Error: Could not grab a frame" << endl;
		exit(-1);
	}
	imgTemp = cvRetrieveFrame(capture);	//imgTemp : entire test image

	vecImg = crDetect.CropNScaleRoi_Reivised(imgTemp, scales);

	crDetect.imgTotal = imgTemp;
	Mat matTotalTmp(imgTemp);
	crDetect.mTotalImg = matTotalTmp;

	rectangle(crDetect.mTotalImg, Rect(crDetect.recRoi), CV_RGB(255, 255, 255), 2);

}


bool CRForestDetector::LoadTestVideo_Mat(CRForestDetector& crDetect, const int& nDataIdx, float scale0){

	Mat mImgTmp;
	m_CapVideo.read(mImgTmp);
	if (!mImgTmp.data){
		printf("ERROR: End of the video file\n");
		exit(0);
	}

	//set 1st ROI
	mImgTmp.copyTo(crDetect.mTmp);
	Rect_<int> rect_ROI;
	vec_ORoi_tl.clear();

	//crop
	for (int num_scale = 0; num_scale < scales.size(); num_scale++){

		int nX = (mImgTmp.cols - vec_ORoiSize[num_scale].width) / 2;
		int nY = mImgTmp.rows*vec_RoiRowRatio[num_scale] - vec_ORoiSize[num_scale].height;

		if (num_scale == 0){
			vec_Roi_tl.push_back(Point2i(nX, nY));
			rect_ROI = Rect(nX, nY, vec_ORoiSize[num_scale].width, vec_ORoiSize[num_scale].height);
			crDetect.mRoi1 = crDetect.mTmp(rect_ROI);
		}
		vec_ORoi_tl.push_back(Point2i(nX, nY));
	}

	crDetect.mTotalImg = mImgTmp;

	/*rectangle(crDetect.mTotalImg, Rect(crDetect.recRoi), CV_RGB(255, 255, 255), 2);

	rectangle(crDetect.mTotalImg, Rect(vec_ORoi_tl[1], vec_ORoiSize[1]), CV_RGB(255, 0, 255), 2);
	rectangle(crDetect.mTotalImg, Rect(vec_ORoi_tl[2], vec_ORoiSize[2]), CV_RGB(0, 255, 255), 2);*/
	rectangle(crDetect.mTotalImg, Rect(rect_ROI), CV_RGB(255, 255, 255), 2);
	rect_TROI = rect_ROI;
}


bool CRForestDetector::LoadTestVideo_oneROI(){

	Mat mImgTmp;
	m_CapVideo.read(mImgTmp);
	if (!mImgTmp.data){
		printf("ERROR: End of the video file\n");
		exit(0);
	}

	mImgTmp.copyTo(mTmp);
	mTotalImg = mImgTmp;

	for (int i = 0; i < vec_rectROI.size(); i++){
		rectangle(mTotalImg, vec_rectROI[i], CV_RGB(255, 255, 255), 2);
	}
	//	initial ROI
	rectangle(mTotalImg, vec_rectROI[0], CV_RGB(255, 255, 255), 2);

	mRoi1 = mTmp(vec_rectROI[0]);
	rect_TROI = vec_rectROI[0];
}


// load positive training image filenames
void loadTrainPosFile(std::vector<string>& vFilenames, std::vector<CvRect>& vBBox, std::vector<std::vector<CvPoint> >& vCenter) {

	unsigned int size, numop;
	ifstream in(trainposfiles.c_str());

	if (in.is_open()) {
		in >> size;
		in >> numop;
		cout << "Load Train Pos Examples: " << size << " - " << numop << endl;	//50 1

		vFilenames.resize(size);
		vCenter.resize(size);
		vBBox.resize(size);

		for (unsigned int i = 0; i<size; ++i) {
			// Read filename
			in >> vFilenames[i];

			// Read bounding box
			in >> vBBox[i].x; in >> vBBox[i].y;
			in >> vBBox[i].width;
			vBBox[i].width -= vBBox[i].x;
			in >> vBBox[i].height;
			vBBox[i].height -= vBBox[i].y;

			if (vBBox[i].width<p_width || vBBox[i].height<p_height) {
				cout << "Width or height are too small" << endl;
				cout << vFilenames[i] << endl;
				exit(-1);
			}

			// Read center points
			vCenter[i].resize(numop);
			for (unsigned int c = 0; c<numop; ++c) {
				in >> vCenter[i][c].x;
				in >> vCenter[i][c].y;
			}
		}

		in.close();

	}
	else {
		cerr << "File not found " << trainposfiles.c_str() << endl;
		exit(-1);
	}
}

// load negative training image filenames
void loadTrainNegFile(std::vector<string>& vFilenames, std::vector<CvRect>& vBBox) {

	unsigned int size, numop;
	ifstream in(trainnegfiles.c_str());

	if (in.is_open()) {
		in >> size;
		in >> numop;
		cout << "Load Train Neg Examples: " << size << " - " << numop << endl;

		vFilenames.resize(size);
		if (numop>0)
			vBBox.resize(size);
		else
			vBBox.clear();

		for (unsigned int i = 0; i<size; ++i) {
			// Read filename
			in >> vFilenames[i];

			// Read bounding box (if available)
			if (numop>0) {
				in >> vBBox[i].x; in >> vBBox[i].y;
				in >> vBBox[i].width;
				vBBox[i].width -= vBBox[i].x;
				in >> vBBox[i].height;
				vBBox[i].height -= vBBox[i].y;

				if (vBBox[i].width<p_width || vBBox[i].height<p_height) {
					cout << "Width or height are too small" << endl;
					cout << vFilenames[i] << endl;
					exit(-1);
				}
			}
		}
		in.close();
	}
	else {
		cerr << "File not found " << trainposfiles.c_str() << endl;
		exit(-1);
	}
}

// Show leaves
void show() {
	// Init forest with number of trees
	CRForest crForest(ntrees);

	// Load forest
	crForest.loadForest(treepath.c_str());

	// Show leaves
	crForest.show(100, 100);
}


void showproject(Mat& mTemp){
	double dMax = 0;
	minMaxLoc(mTemp, 0, &dMax);
	Mat xProject = Mat(mTemp.rows, dMax, CV_32FC1, Scalar(0));
	
	for (int i = 0; i < mTemp.rows; i++){
		for (int j = 0; j < mTemp.at<int>(0, i); j++){
			xProject.at<int>(j,i) = 1;
		}
	}
	imshow("projection",xProject);
	cvWaitKey(0);
}

void ProjectImage(Mat& img, vector<float>& vecXProject, vector<float>& vecYProject){
	int nWidth, nHeight;
	nWidth = img.cols;
	nHeight = img.rows;
	vecXProject.resize(nWidth);
	vecYProject.resize(nHeight);
	for (int i = 0; i < nHeight; i++){
		for (int j = 0; j < nWidth; j++){
			vecXProject[j] += (unsigned int)img.at<int>(i,j);
		}
	}
}

void FindMaximum(Mat& imgSrc, unsigned int nThreshold, float fScale, vector<sPointInfo>& vecPoint){
	double dMax, dTempMax;
	Point pIdx;
	vector<int> vecXLocation, vecYLocation, vecMaxIntensity;
	vector<float> vecXProjection, vecYProjection;

	vecXProjection.clear(); vecYProjection.clear();

	//prject image to x and y direction;
	ProjectImage(imgSrc, vecXProjection, vecYProjection);

	Mat mTemp(vecXProjection);

	//showproject(mTemp);

	//imshow("x_projection",mTemp);
	//cvWaitKey(0);

	int nRow = mTemp.rows;
	Mat mResult = Mat::zeros(nRow, 1, CV_32FC1);

	//get the maximum value of mTemp
	minMaxLoc(mTemp, 0, &dMax);
	dTempMax = dMax*0.95;

	//calculate the derivative of image
	for (int i = 1; i < nRow; i++)
		mResult.ptr<float>(i)[0] = mTemp.ptr<float>(i)[0] - mTemp.ptr<float>(i - 1)[0];

	//find the index of potential maximal value in x direction
	for (int i = 1; i < nRow; i++){
		if (mResult.ptr<float>(i - 1)[0] > 0 && mResult.ptr<float>(i)[0]<0 && mTemp.ptr<float>(i)[0] >dTempMax)
			vecXLocation.push_back(i);
	}

	//find index of potential maximal value in y direction and its intensity
	for (unsigned int i = 0; i < vecXLocation.size(); i++){
		mTemp = imgSrc.col(vecXLocation[i]);
		minMaxLoc(mTemp, 0, &dMax, 0, &pIdx);
		vecYProjection.push_back(pIdx.y);
		vecMaxIntensity.push_back((int)dMax);
	}

	for (unsigned int i = 0; i < vecXLocation.size(); i++){
		if ((unsigned int)vecMaxIntensity[i] > nThreshold){
			sPointInfo pointInfo;
			pointInfo.nX = vecXLocation[i] / fScale;
			pointInfo.nY = vecYLocation[i] / fScale;
			pointInfo.nIntensity = vecMaxIntensity[i] / fScale;
			pointInfo.fScale = fScale;
			vecPoint.push_back(pointInfo);
		}
	}
}

void VerifyClustering(vector<Rect_<int>>& vecDetectVehicle, Rect_<int> & rectCandidate, double& dMaxValue, vector<float>& vecDetectionValue, bool& bOverlap){
	Rect_<int> rectOverlap;
	float OverlapRate;

	for (int v = 0; v < vecDetectVehicle.size(); v++){
		rectOverlap = rectCandidate & vecDetectVehicle[v];
		OverlapRate = (float)rectOverlap.area() / vecDetectVehicle[v].area();
		//float OverlapRate = (float)(rectOverlap->area() / vecDetectVehicle[v].area());
		//printf("OverlapRate:%f\n", OverlapRate);
		if (OverlapRate > 0.6){
			//merge
			if (dMaxValue > vecDetectionValue[v]){
				vector<Rect_<int>>	vecRectList;
				Rect_<int> rectTemp = vecDetectVehicle[v];

				vecRectList.push_back(rectTemp);
				vecRectList.push_back(rectCandidate);
				groupRectangles(vecRectList, 0, 0.2);

				vecDetectVehicle[v] = vecRectList.at(0);
				vecDetectionValue[v] = dMaxValue;
			}
			bOverlap = true;
		}
	}
	if (!bOverlap){
		vecDetectVehicle.push_back(rectCandidate);
		vecDetectionValue.push_back(dMaxValue);
	}
}

// Run detector
//void detect(CRForestDetector& crDetect) {
//	//setTestVideo
//	if (SetTestVideo()) cout << "test video setted" << endl;
//
//	char buffer[200];
//
//	// Storage for output
//	//vector<vector<IplImage*> > vImgDetect(scales.size());
//	vector<IplImage*> vImgDetect(scales.size());
//
//	//Training image size
//	int nTrainWidth = sizeTrain.width;
//	int nTrainHeight = sizeTrain.height;
//	
//	int nIdx;
//	for (nIdx = 1;; nIdx++){
//
//		vecDetection.clear();
//
//		// Load image
//		IplImage *img = 0, *vImgDetectCopy = 0;
//
//		//Element
//		IplConvKernel* element1 = cvCreateStructuringElementEx(3, 13, 1, 6, CV_SHAPE_RECT);
//		IplConvKernel* element2 = cvCreateStructuringElementEx(11, 11, 5, 5, CV_SHAPE_RECT);
//
//		//threshold
//		unsigned int Threshold = 95;
//
//		//vector
//		vector<sPointInfo> vecPointInfo;
//
//		//rectangle for contour bounding box
//		CvRect rect;
//		vector<Rect_<int>> vRecDetected;
//		vector<float> vecDetectedValue;
//
//		if (LoadTestVideo(capture, nIdx)) cout << "test video loaded" << endl;
//		img = imgTest;
//
//		//cvShowImage("origin", img);
//		/*cvShowImage("origin", img);
//		cvWaitKey(0.001);*/
//
//		// Prepare scales
//		//for (unsigned int k = 0; k < vImgDetect.size(); ++k) {
//		//	vImgDetect[k].resize(ratios.size());	//ratios : ratio 들을 저장한 벡터 1개 ratio =1
//		//	for (unsigned int c = 0; c < vImgDetect[k].size(); ++c) {
//		//		vImgDetect[k][c] = cvCreateImage(cvSize(int(imgTestCopy->width*scales[k] + 0.5), int(imgTestCopy->height*scales[k] + 0.5)), IPL_DEPTH_32F, 1);
//		//	}
//		//}
//		for (unsigned int k = 0; k < vImgDetect.size(); ++k) {
//				vImgDetect[k] = cvCreateImage(cvSize(int(imgTestCopy->width*scales[k] + 0.5), int(imgTestCopy->height*scales[k] + 0.5)), IPL_DEPTH_32F, 1);
//			}
//	
//		// Detection for all scales
//		//crDetect.detectPyramid(img, vImgDetect, ratios);
//		crDetect.detectPyramid_Revised(img, vImgDetect, scales);
//
//		//detect the vehicles using Hough map
//		for (int i = 0; i < vImgDetect.size(); i++){
//			for (int j = 0; j < vImgDetect[i].size(); j++){
//				vImgDetectCopy = (IplImage*)cvClone(vImgDetect[i][j]);			
//
//				//Mean shifted image
//				IplImage* imgTemp1 = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_32F, 3);
//				IplImage* imgMeanShift = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_8U, 3);
//				IplImage* imgDilated = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_8U, 3);
//				IplImage* imgEroded = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_8U, 3);
//				IplImage* imgContour = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_8U, 1);
//
//				cvCvtColor(vImgDetectCopy, imgTemp1, CV_GRAY2RGB);
//				cvConvertScale(imgTemp1, imgMeanShift, 255);
//
//				// MeanshiftFiltering
//				cvPyrMeanShiftFiltering(imgMeanShift, imgMeanShift, 15, 30, 1);
//				//cvShowImage("Mean-shift clustering", imgMeanShift);
//
//				//threshold
//				cvThreshold(imgMeanShift, imgMeanShift, 130, 255, CV_THRESH_BINARY);
//				//cvShowImage("threshold", imgMeanShift);
//
//				//dilate
//				cvDilate(imgMeanShift, imgDilated, element1, 1);
//				//cvShowImage("1. dilate", imgDilated);
//
//				//cvConvertScale(vImgDetectCopy, imgMeanShift,1);
//				cvCvtColor(imgDilated, imgContour, CV_RGB2GRAY);
//
//				//contour vector
//				CvMemStorage* contours = cvCreateMemStorage(0);
//				cvClearMemStorage(contours);
//				CvSeq* firstContour;
//
//				//find contours in current frame
//				cvFindContours(imgContour, contours, &firstContour, sizeof(CvContour),CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//				
//				//find max value
//				CvPoint ptMax;
//				double dMaxValue;
//				CvRect rectBound;
//				float fReverseScale = (float)1 / scales[i];
//				//cout << "reverse scale: " << fReverseScale << endl;
//
//				for (; firstContour; firstContour=firstContour->h_next){
//					//find bouding box
//					rectBound = cvBoundingRect(firstContour);
//					cvSetImageROI(vImgDetectCopy, rectBound);
//					cvMinMaxLoc(vImgDetectCopy, 0, &dMaxValue, 0, &ptMax, 0);
//		
//					CvPoint ptRealMax = cvPoint((rectBound.x + ptMax.x)*fReverseScale, (rectBound.y+ptMax.y)*fReverseScale);
//					int nTopLeftPointX = (ptRealMax.x - nTrainWidth / 2);
//					int nTopLeftPointY = (ptRealMax.y - nTrainHeight / 2);
//					CvRect recCandidate = cvRect(nTopLeftPointX, nTopLeftPointY, nTrainWidth*fReverseScale, nTrainHeight*fReverseScale);
//					Rect_<int> recTemp = Rect(nTopLeftPointX, nTopLeftPointY, nTrainWidth*fReverseScale, nTrainHeight*fReverseScale);
//					
//					if (vRecDetected.size() == 0){
//						vRecDetected.push_back(recTemp);
//						vecDetectedValue.push_back(dMaxValue);
//					}
//					else{
//						bool fOverlap = false;
//						VerifyClustering(vRecDetected, recTemp, dMaxValue, vecDetectedValue, fOverlap);
//					}
//				}
//				firstContour = 0;			
//			}
//		}
//		for (int i = 0; i < vRecDetected.size(); i++){ 
//			rectangle(matOriginalImg, vRecDetected[i], CV_RGB(255, 0, 0), 2);
//		}
//		imshow("result", matOriginalImg);
//		cvWaitKey(1);
//
//	}
//}
void detect_Revised(CRForestDetector& crDetect) {
	//setTestVideo
	if (SetTestVideo()) cout << "test video setted" << endl;

	char buffer[200];

	// Storage for output
	//vector<vector<IplImage*> > vImgDetect(scales.size());
	vector<IplImage*> vImgDetect(scales.size());

	//Training image size
	int nTrainWidth = sizeTrain.width;
	int nTrainHeight = sizeTrain.height;



	int nIdx;
	for (nIdx = 1;; nIdx++){

	
		vecDetection.clear();
		vector<Rect_<int>> vecVerifiedVehicle;
		vecVerifiedVehicle.clear();

		// Load image
		IplImage *img = 0, *vImgDetectCopy = 0;

		//Element
		IplConvKernel* element1 = cvCreateStructuringElementEx(3, 13, 1, 6, CV_SHAPE_RECT);
		IplConvKernel* element2 = cvCreateStructuringElementEx(11, 11, 5, 5, CV_SHAPE_RECT);

		//threshold
		unsigned int Threshold = 95;


		//rectangle for contour bounding box
		CvRect rect;
		vector<Rect_<int>> vRecDetected;
		vRecDetected.clear();
		vector<float> vecDetectedValue;

		//if (LoadTestVideo(capture, nIdx)) cout << "test video loaded" << endl;
		LoadTestVideo(crDetect, capture, nIdx);

		//Mat diff = crDetect.mCurrent - crDetect.mPrevious;
		//
		//add(diff,crDetect.mDiffAdd,crDetect.mDiffAdd);
		//imshow("diff-added", crDetect.mDiffAdd);

		img = crDetect.imgTotal;

		Rect_<int> recTemp;

		//cvShowImage("origin", img);
		///*cvShowImage("origin", img);
		/*cvWaitKey(0.001);*///*/



		// Prepare scales
		for (unsigned int k = 0; k < vImgDetect.size(); ++k) {
			//vImgDetect[k] = cvCreateImage(cvSize(int(vecImg[k]->width*scales[k] + 0.5), int(vecImg[k]->height*scales[k] + 0.5)), IPL_DEPTH_32F, 1);

			vImgDetect[k] = cvCreateImage(cvSize(int(imgTestCopy->width*scales[k] + 0.5), int(imgTestCopy->height*scales[k] + 0.5)), IPL_DEPTH_32F, 1);
			//cout << "test: scale = " << scales[k] << endl;
		}

		// Detection for all scales
		//Hough Voting 시행
		crDetect.detectPyramid_Revised(img, vImgDetect, scales);
		//crDetect.detectPyramid_Revised2(crDetect, vecImg, vImgDetect, scales, vecRect);

		//detect the vehicles using Hough map
		for (int i = 0; i < vImgDetect.size(); i++){
			//for (int j = 0; j < vImgDetect[i].size(); j++){
			vImgDetectCopy = (IplImage*)cvClone(vImgDetect[i]);
			//cvShowImage("Hough vote", vImgDetect[i]);
			//cvWaitKey(0);
			//Mean shifted image
			IplImage* imgTemp1 = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_32F, 3);
			IplImage* imgMeanShift = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_8U, 3);
			IplImage* imgDilated = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_8U, 3);
			IplImage* imgEroded = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_8U, 3);
			IplImage* imgContour = cvCreateImage(cvSize((int)vImgDetectCopy->width, (int)vImgDetectCopy->height), IPL_DEPTH_8U, 1);

			cvCvtColor(vImgDetectCopy, imgTemp1, CV_GRAY2RGB);
			cvConvertScale(imgTemp1, imgMeanShift, 255);

			// MeanshiftFiltering
			cvPyrMeanShiftFiltering(imgMeanShift, imgMeanShift, 15, 20, 1);
			//cvShowImage("Mean-shift clustering", imgMeanShift);

			//threshold
			cvThreshold(imgMeanShift, imgMeanShift, 110, 255, CV_THRESH_BINARY);
			//cvShowImage("threshold", imgMeanShift);

			//dilate
			cvDilate(imgMeanShift, imgDilated, element1, 1);
			//cvShowImage("1. dilate", imgDilated);

			/*cvShowImage("dilate", imgDilated);
			cvWaitKey(0);
			*/
			//cvConvertScale(vImgDetectCopy, imgMeanShift,1);
			cvCvtColor(imgDilated, imgContour, CV_RGB2GRAY);

			//contour vector
			CvMemStorage* contours = cvCreateMemStorage(0);
			cvClearMemStorage(contours);
			CvSeq* firstContour;

			//find contours in current frame
			cvFindContours(imgContour, contours, &firstContour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			//find max value
			CvPoint ptMax;
			double dMaxValue;
			CvRect rectBound;
			float fReverseScale = (float)1 / scales[i];
			//cout << "reverse scale: " << fReverseScale << endl;


			for (; firstContour; firstContour = firstContour->h_next){
				//find bouding box
				rectBound = cvBoundingRect(firstContour);
				cvSetImageROI(vImgDetectCopy, rectBound);
				cvMinMaxLoc(vImgDetectCopy, 0, &dMaxValue, 0, &ptMax, 0);

				CvPoint ptRealMax = cvPoint((rectBound.x + ptMax.x)*fReverseScale, (rectBound.y + ptMax.y)*fReverseScale);
				int nTopLeftPointX = (ptRealMax.x - nTrainWidth / 2) + crDetect.recRoi.x;
				int nTopLeftPointY = (ptRealMax.y - nTrainHeight / 2) + crDetect.recRoi.y;
				//CvRect recCandidate = cvRect(nTopLeftPointX, nTopLeftPointY, nTrainWidth*fReverseScale, nTrainHeight*fReverseScale);
				recTemp = Rect(nTopLeftPointX, nTopLeftPointY, nTrainWidth*fReverseScale, nTrainHeight*fReverseScale);

				if (vRecDetected.size() == 0){
					vRecDetected.push_back(recTemp);
					vecDetectedValue.push_back(dMaxValue);
				}
				else{
					bool fOverlap = false;
					VerifyClustering(vRecDetected, recTemp, dMaxValue, vecDetectedValue, fOverlap);
				}
			}
			firstContour = 0;
			//}
		}
		for (int i = 0; i < vRecDetected.size(); i++){
			rectangle(crDetect.mTotalImg, vRecDetected[i], CV_RGB(255, 255, 0), 2);
		}

		//verification
		crDetect.VerifyDetection(vRecDetected, vecVerifiedVehicle);
		if (vecVerifiedVehicle.size() == 0) printf("Not Verified\n");
		else{
			for (int i = 0; i < vecVerifiedVehicle.size(); i++){
				rectangle(crDetect.mTotalImg, vecVerifiedVehicle[i], CV_RGB(255, 0, 0), 2);
			}
		}

		imshow("result", crDetect.mTotalImg);
		cvWaitKey(1);
		//mPrevious = 
	}
}
void detect_Revised_Mat(CRForestDetector& crDetect) {

	SetTestVideo_Mat();

	char buffer[200];

	//tracking
	vector<SKalman> MultiKF;
	// tracker //
	Track_t High;
/*	ptVanishing.x = 640 - 20;
	ptVanishing.y = 360 - 0;*/

	ptVanishing = crDetect.ptMain;

	//Training image size
	int nTrainWidth = sizeTrain.width;
	int nTrainHeight = sizeTrain.height;

	//initialize
	crDetect.InitializeROI(crDetect);

	int nPrevContourArea = 0;
	//threshold
	double dAverage = 0;

	int nIdx;
	for (nIdx = 1;; nIdx++){
		int nContourArea = 0;
		int nNumContour;
		int nInit = 0;
		Mat vImgDetectCopy;
		vector<Rect_<int>> vecVerifiedVehicle;
		vector<Mat> HoughMap;
		//vector<Mat> Meanfilter;
		vector<Mat> Threshold;
		vector<sMaxPtInfo> sMaxPt;
		vector<sMaxPts> sMaxPtRoi;

		vecDetection.clear();
		crDetect.nMaxCount.clear();

		//Element
		IplConvKernel* element1 = cvCreateStructuringElementEx(3, 13, 1, 6, CV_SHAPE_RECT);
		IplConvKernel* element2 = cvCreateStructuringElementEx(11, 11, 5, 5, CV_SHAPE_RECT);
		cv::Mat element_dilate(3, 13, CV_8U, cv::Scalar(1));
		cv::Mat element_erode(11, 11, CV_8U, cv::Scalar(1));

		////threshold
		//unsigned int usThreshold = 110;//video1
		//unsigned int usThreshold = 245;//video2
		//unsigned int usThreshold = 190;//video7
		unsigned int usThreshold = 220;//video13

		//rectangle for contour bounding box
		CvRect rect;
		vector<Rect_<int>> vRecDetected;
		vector<float> vecDetectedValue;

		//check time
		double dTimeBegin = cvGetTickCount();

		if (!crDetect.LoadTestVideo_Mat(crDetect, nIdx, scales.at(0))) cout << "test video not loaded" << endl;
		crDetect.detectPyramid_Revised_Mat(crDetect, crDetect.vImgDetect, scales);

		//detect the vehicles using Hough map
		for (int num_scale = 0; num_scale < crDetect.vImgDetect.size(); num_scale++){
			nNumContour = 0;
			double nContourSizeAvg = 0;
			crDetect.vImgDetect[num_scale].copyTo(vImgDetectCopy);
			Mat imgTemp, imgMeanShift,imgNormalize, imgThresholded, imgThresholded_contour, imgDilated,/*imgDilated_af,*/ imgEroded, imgContour;

			Point ptMax_before;
			double dMaxValue_before;

			/*//MeanShift
			cvtColor(vImgDetectCopy, imgMeanShift, CV_GRAY2RGB);*/
			HoughMap.push_back(vImgDetectCopy);

			imgNormalize=vImgDetectCopy.clone();

			//normalization 
			crDetect.Normalization(vImgDetectCopy, imgNormalize);

			//convertScaleAbs(vImgDetectCopy, imgMeanShift, 255);
			convertScaleAbs(vImgDetectCopy, imgNormalize, 255);
			//imgMeanShift = vImgDetectCopy;

			/*
			//MeanShift
			pyrMeanShiftFiltering(imgMeanShift, imgMeanShift, 15, 20);
			Meanfilter.push_back(imgMeanShift);
			*/
			//minMaxLoc(vImgDetectCopy, NULL, &dMaxValue_before, NULL, &ptMax_before);
			//			cout <<"Original -	"<< "Roi #" << num_scale << " Max Value: " << dMaxValue_before << endl;


			//threshold
			/*dAverage = (sum(imgNormalize)[0]) / (double)(imgNormalize.cols*imgNormalize.rows);
			cout << "average value: " << dAverage<<endl;
		*/
			//cout <<"Max count " <<crDetect.nMaxCount[num_scale] << endl;

			//threshold(imgNormalize, imgThresholded, crDetect.nMaxCount[num_scale], 255, CV_THRESH_BINARY);
			threshold(imgNormalize, imgThresholded, usThreshold, 255, CV_THRESH_BINARY);
			//threshold(imgNormalize, imgThresholded, dAverage, 255, CV_THRESH_BINARY);
			//threshold(imgNormalize, imgThresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			
			imgThresholded.copyTo(imgThresholded_contour);
			Threshold.push_back(imgThresholded);
			//usThreshold = dMaxValue_before*THRESHOLD_RATIO;

			////MeanShift
			//cvtColor(imgThresholded_contour, imgThresholded_contour, CV_RGB2GRAY);

			vector<Vec4i> hierarchy;
			vector<vector<Point>> normalContours;

			findContours(imgThresholded_contour, normalContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

			hierarchy.clear();
			vector<vector<Point>> N_contours_poly(normalContours.size());	//contour 개수만큼 벡터 선언
			vector<Rect> N_boundRect(normalContours.size());	//contour 개수만큼 사각형 선언

			vector<sMaxPtInfo> sMaxPt_current;

			for (int i = 0; i < normalContours.size(); i++){		//contour 개수만큼 for문 돌림
				approxPolyDP(Mat(normalContours[i]), N_contours_poly[i], 3, true);
				N_boundRect[i] = boundingRect(Mat(N_contours_poly[i]));

				double dArea = contourArea(normalContours[i], false);
				nContourSizeAvg += dArea;

				nNumContour++;

				if (dArea <= PIXEL_CNT/* / (THRESHOLD_RATIO)*/) {
					continue;
				}

				Point ptMax;
				double dMaxValue;
				minMaxLoc(vImgDetectCopy(N_boundRect[i]), NULL, &dMaxValue, NULL, &ptMax);

				if (dMaxValue <= 0.8)continue;
			
				ptMax += N_boundRect[i].tl();		//x,y

				float fReverseScale = 1 / (float)scales[num_scale];
				ptMax.x = ptMax.x*fReverseScale;
				ptMax.y = ptMax.y*fReverseScale;
				ptMax += crDetect.vec_ORoi_tl[num_scale];

				Rect_<int> recTemp = Rect(ptMax.x - nTrainWidth*fReverseScale / 2, ptMax.y - nTrainHeight*fReverseScale / 2, nTrainWidth*fReverseScale, nTrainHeight*fReverseScale);
				sMaxPts sPt;
				int flag = 0;

				if (nInit == 0){
					//store initial detection points
					/*sMaxPts sPt;*/
					sPt.ptMax = ptMax;
					sPt.dMaxValue = dMaxValue;
					sPt.rectMax = recTemp;
					sMaxPtRoi.push_back(sPt);
					//continue;
				}
				else{
					//compare
					for (int i = 0; i < sMaxPtRoi.size(); i++){
						int nDistance = norm(sMaxPtRoi[i].ptMax - ptMax);
						if (nDistance < 50){
							//cout<<"Euclidean distance: "<<nDistance<<endl;
							if (sMaxPtRoi[i].dMaxValue < dMaxValue){
								sMaxPtRoi[i].ptMax = ptMax;
								sMaxPtRoi[i].dMaxValue = dMaxValue;
								sMaxPtRoi[i].rectMax = recTemp;
							}
							flag = 1;
						}
					}
				}
				if (!flag){
					sPt.ptMax = ptMax;
					sPt.dMaxValue = dMaxValue;
					sPt.rectMax = recTemp;
					sMaxPtRoi.push_back(sPt);

				}
				/*if (sMaxPtRoi.size() == 0){
					vRecDetected.push_back(sMaxPtRoi);
					vecDetectedValue.push_back(dMaxValue);
				}
				else{
					bool fOverlap = false;
					VerifyClustering(vRecDetected, recTemp, dMaxValue, vecDetectedValue, fOverlap);
				}*/

				////draw points
				//int red = 0; int green = 0; int blue = 0;
				//if (num_scale == 0) {
				//	red = 255; green = 255; blue = 255;
				//}
				//else if (num_scale == 1) {
				//	red = 255; blue = 255;
				//}
				//else if (num_scale == 2) {
				//	green = 255; blue = 255;
				//}
				//circle(crDetect.mTotalImg, ptMax, 3, CV_RGB(red, green, blue), 5);
				////cout<<"X: "<<ptMax.x<<" Y: "<<ptMax.y<<endl;	

			}
			//nContourSizeAvg/=normalContours.size();
			nContourSizeAvg /= nNumContour;
			//cout<<"ROI# "<<num_scale<<" average contour size: "<<nContourSizeAvg<<endl;	//

			nContourArea += nContourSizeAvg;
			nPrevContourArea = nContourArea;
			nInit++;
		}

		double dTimeEnd = cvGetTickCount();
		double dTrainingTime = (double)(0.001 * (dTimeEnd - dTimeBegin) / cvGetTickFrequency());
		printf("[time]: %3.2f msec\n", dTrainingTime);

		imshow("HoughMap - roi 2", HoughMap[1]);
		imshow("HoughMap - roi 3", HoughMap[2]);
		imshow("HoughMap - roi 1", HoughMap[0]);

		/*	imshow("Meanfilter - roi 2", Meanfilter[1]);
		imshow("Meanfilter - roi 3", Meanfilter[2]);
		imshow("Meanfilter - roi 1", Meanfilter[0]);*/

		imshow("Threshold - roi 2", Threshold[1]);
		imshow("Threshold - roi 3", Threshold[2]);
		imshow("Threshold - roi 1", Threshold[0]);

		HoughMap.clear();
		//		Meanfilter.clear();
		Threshold.clear();

		//프레임별 검출 박스 출력
		crDetect.vImgDetect.clear();
		/*for (int i = 0; i < vRecDetected.size(); i++){
		rectangle(crDetect.mTotalImg, vRecDetected[i], CV_RGB(255, 255, 0), 2);
		}*/

		//clustering
		clustering(sMaxPtRoi,10);

		//verification
		//crDetect.VerifyDetection(vRecDetected, vecVerifiedVehicle);
		crDetect.VerifyDetection_check(sMaxPtRoi, vecVerifiedVehicle);

		for (int i = 0; i < vecVerifiedVehicle.size(); i++){
			rectangle(crDetect.mTotalImg, vecVerifiedVehicle[i], CV_RGB(255, 0, 0), 2);
		}
		
		circle(crDetect.mTotalImg, ptVanishing, 1, CV_RGB(255, 0, 0), 2);
		vector<Rect> vecDetect;
		vector<Rect> vecValidRec;
		Point pCurrent;
		float scaledist = 0.4;
		int cntCandidate = 3; // 후보 ROI 검증 횟수, 검증되면 tracking 시작 2
		int cntBefore =5; // 못찾은 횟수가 연속으로 n프레임 이상이면 ROI 제거 4
		int frameCandiate = 5; // 못찾은 횟수가 연속으로 n프레임 이상이면 후보 ROI 제거
		vector<Rect> vecRectTracking;
		Mat imgROImask = Mat::zeros(crDetect.mTotalImg.size(), CV_8UC1);
		imgROImask(rect_TROI).setTo(Scalar::all(255));
		kalmanMultiTarget(crDetect.mTotalImg, vecVerifiedVehicle, High, MultiKF, scaledist, cntCandidate, cntBefore, frameCandiate, nIdx, rect_TROI, vecValidRec, imgROImask, crDetect);
		vector<Rect> vecValid;
		vecValid = High.vecBefore;

		for (int i = 0; i < vecValid.size(); i++)
		{
			rectangle(crDetect.mTotalImg, vecValid[i], MultiKF[i].rgb, 2);
		}

		Size half = Size(crDetect.mTotalImg.cols / 2, crDetect.mTotalImg.rows / 2);
		resize(crDetect.mTotalImg, crDetect.mTotalImg, half);
		imshow("result", crDetect.mTotalImg);
		char c = cvWaitKey(1);
		if (c == 27) exit(-1);

		//crDetect.oVideoWriter.write(crDetect.mTotalImg);
		//cout << "frame# " << nIdx << endl;
	}
	crDetect.vec_Roi_tl.clear();
	crDetect.vec_ORoi_tl.clear();

	//fclose(crDetect.pixelSize);
}

void detect_he(CRForestDetector& crDetect) {
	//
	if (!SetTestVideo_Mat())	exit(-1);

	char buffer[200];

	//tracking
	vector<SKalman> MultiKF;
	// tracker //
	Track_t High;
	ptVanishing.x = 640 - 20;
	ptVanishing.y = 360 - 0;

	crDetect.sizeTrainImg.width = sizeTrain.width;
	crDetect.sizeTrainImg.height = sizeTrain.width;

	//Training image size
	int nTrainWidth = sizeTrain.width;
	int nTrainHeight = sizeTrain.height;

	//initialize
	crDetect.InitializeROI_scale(scales);

	int nPrevContourArea = 0;
	//threshold
	double dAverage = 0;

	int nIdx;
	for (nIdx = 1;; nIdx++){
		int nContourArea = 0;
		int nNumContour;
		int nInit = 0;
		Mat vImgDetectCopy;
		vector<Rect_<int>> vecVerifiedVehicle;
		vector<Mat> HoughMap;
		//vector<Mat> Meanfilter;
		vector<Mat> Threshold;
		vector<sMaxPtInfo> sMaxPt;
		vector<sMaxPts> sMaxPtRoi;

		vecDetection.clear();
		crDetect.nMaxCount.clear();
		
		////threshold
		unsigned int usThreshold = 180;//video1	check	GC margin
		//unsigned int usThreshold = 180;//video1	check	GC margin	//100 10
		//unsigned int usThreshold = 245;//video2
		//unsigned int usThreshold = 190;//video7,15
		//unsigned int usThreshold = 220;//video13
		//unsigned int usThreshold = 248;//video18	check	GC margin
		//unsigned int usThreshold = 150;//video5
		//unsigned int usThreshold = 215;//video7
		//unsigned int usThreshold = 200;//video 8
		//unsigned int usThreshold = 190;//video 9
		//unsigned int usThreshold = 210;//video 11		middle
		//unsigned int usThreshold = 225;//video 12		one
		//unsigned int usThreshold = 200;//video 13
		//unsigned int usThreshold = 220;//video 17
		//unsigned int usThreshold = 220;//video 19	crop	GC margin

		//rectangle for contour bounding box
		CvRect rect;
		vector<Rect_<int>> vRecDetected;
		vector<float> vecDetectedValue;

		//check time
		double dTimeBegin = cvGetTickCount();

		if (!crDetect.LoadTestVideo_oneROI()) cout << "test video not loaded" << endl;
		crDetect.detectInPyramid(scales);

		unsigned int thres = 5;

		//detect the vehicles using Hough map
		for (int num_scale = 0; num_scale < crDetect.vImgDetect.size(); num_scale++){
			nNumContour = 0;
			double nContourSizeAvg = 0;
			crDetect.vImgDetect[num_scale].copyTo(vImgDetectCopy);
			Mat imgTemp, imgMeanShift, imgNormalize, imgThresholded, imgThresholded_contour, imgDilated,/*imgDilated_af,*/ imgEroded, imgContour;

			Point ptMax_before;
			double dMaxValue_before;
			vector<sPointInfo> vecPoint;

			/*//MeanShift
			cvtColor(vImgDetectCopy, imgMeanShift, CV_GRAY2RGB);*/
		
			//cout << "roi" << num_scale << " width " << vImgDetectCopy.cols << " height " << vImgDetectCopy.rows<<endl;

			//projection

			FindMaximum(vImgDetectCopy, thres, scales[num_scale], vecPoint);

			HoughMap.push_back(vImgDetectCopy);
			imgNormalize = vImgDetectCopy.clone();

			//normalization 
			crDetect.Normalization(vImgDetectCopy, imgNormalize);

			//convertScaleAbs(vImgDetectCopy, imgMeanShift, 255);
			convertScaleAbs(vImgDetectCopy, imgNormalize, 255);
			//imgMeanShift = vImgDetectCopy;

			//FindMaximum(imgNormalize, thres, scales[num_scale], vecPoint);
			/*Mat mHough;
			imgNormalize.copyTo(mHough);
			cvtColor(mHough, mHough, CV_GRAY2RGB);*/
	/*
			for (int i = 0; i < vecPoint.size(); i++)
				circle(mHough, Point(vecPoint[i].nX, vecPoint[i].nY), 3, CV_RGB(1, 0, 0), 20);
			imshow("points", mHough);*/

			/*
			//MeanShift
			pyrMeanShiftFiltering(imgMeanShift, imgMeanShift, 15, 20);
			Meanfilter.push_back(imgMeanShift);
			*/
			//minMaxLoc(vImgDetectCopy, NULL, &dMaxValue_before, NULL, &ptMax_before);
			//			cout <<"Original -	"<< "Roi #" << num_scale << " Max Value: " << dMaxValue_before << endl;


			//threshold
			/*dAverage = (sum(imgNormalize)[0]) / (double)(imgNormalize.cols*imgNormalize.rows);
			cout << "average value: " << dAverage<<endl;
			*/
			//cout <<"Max count " <<crDetect.nMaxCount[num_scale] << endl;

			//threshold(imgNormalize, imgThresholded, crDetect.nMaxCount[num_scale], 255, CV_THRESH_BINARY);
			threshold(imgNormalize, imgThresholded, usThreshold, 255, CV_THRESH_BINARY);
			//threshold(imgNormalize, imgThresholded, dAverage, 255, CV_THRESH_BINARY);
			//threshold(imgNormalize, imgThresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);


			imgThresholded.copyTo(imgThresholded_contour);
			Threshold.push_back(imgThresholded);
			//usThreshold = dMaxValue_before*THRESHOLD_RATIO;

			////MeanShift
			//cvtColor(imgThresholded_contour, imgThresholded_contour, CV_RGB2GRAY);

			vector<Vec4i> hierarchy;
			vector<vector<Point>> normalContours;

			findContours(imgThresholded_contour, normalContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

			hierarchy.clear();
			vector<vector<Point>> N_contours_poly(normalContours.size());	//contour 개수만큼 벡터 선언
			vector<Rect> N_boundRect(normalContours.size());	//contour 개수만큼 사각형 선언

			vector<sMaxPtInfo> sMaxPt_current;
			int nContourThresh = crDetect.contourThreshold(usThreshold);

			for (int i = 0; i < normalContours.size(); i++){		//contour 개수만큼 for문 돌림
				approxPolyDP(Mat(normalContours[i]), N_contours_poly[i], 3, true);
				N_boundRect[i] = boundingRect(Mat(N_contours_poly[i]));

				double dArea = contourArea(normalContours[i], false);
				nContourSizeAvg += dArea;

				nNumContour++;

				if (dArea <= nContourThresh) {
					continue;
				}

				Point ptMax;
				double dMaxValue;
				minMaxLoc(vImgDetectCopy(N_boundRect[i]), NULL, &dMaxValue, NULL, &ptMax);

				if (dMaxValue <= 0.8)continue;

				ptMax += N_boundRect[i].tl();		//x,y

				float fReverseScale = 1 / (float)scales[num_scale];
				ptMax.x = ptMax.x*fReverseScale + crDetect.vec_rectROI[num_scale].x;
				ptMax.y = ptMax.y*fReverseScale + crDetect.vec_rectROI[num_scale].y;
				//ptMax += crDetect.vec_ORoi_tl[num_scale];

				Rect_<int> recTemp = Rect(ptMax.x - nTrainWidth*fReverseScale / 2, ptMax.y - nTrainHeight*fReverseScale / 2, nTrainWidth*fReverseScale, nTrainHeight*fReverseScale);
				sMaxPts sPt;
				int flag = 0;

				if (nInit == 0){
					//store initial detection points
					/*sMaxPts sPt;*/
					sPt.ptMax = ptMax;
					sPt.dMaxValue = dMaxValue;
					sPt.rectMax = recTemp;
					sMaxPtRoi.push_back(sPt);
					//continue;
				}
				else{
					//compare
					for (int i = 0; i < sMaxPtRoi.size(); i++){
						int nDistance = norm(sMaxPtRoi[i].ptMax - ptMax);
						if (nDistance < 50){
							//cout<<"Euclidean distance: "<<nDistance<<endl;
							if (sMaxPtRoi[i].dMaxValue < dMaxValue){
								sMaxPtRoi[i].ptMax = ptMax;
								sMaxPtRoi[i].dMaxValue = dMaxValue;
								sMaxPtRoi[i].rectMax = recTemp;
							}
							flag = 1;
						}
					}
				}
				if (!flag){
					sPt.ptMax = ptMax;
					sPt.dMaxValue = dMaxValue;
					sPt.rectMax = recTemp;
					sMaxPtRoi.push_back(sPt);

				}
				/*if (sMaxPtRoi.size() == 0){
				vRecDetected.push_back(sMaxPtRoi);
				vecDetectedValue.push_back(dMaxValue);
				}
				else{
				bool fOverlap = false;
				VerifyClustering(vRecDetected, recTemp, dMaxValue, vecDetectedValue, fOverlap);
				}*/

				////draw points
				//int red = 0; int green = 0; int blue = 0;
				//if (num_scale == 0) {
				//	red = 255; green = 255; blue = 255;
				//}
				//else if (num_scale == 1) {
				//	red = 255; blue = 255;
				//}
				//else if (num_scale == 2) {
				//	green = 255; blue = 255;
				//}
				//circle(crDetect.mTotalImg, ptMax, 3, CV_RGB(red, green, blue), 5);
				////cout<<"X: "<<ptMax.x<<" Y: "<<ptMax.y<<endl;	

			}
			//nContourSizeAvg/=normalContours.size();
			nContourSizeAvg /= nNumContour;
			//cout<<"ROI# "<<num_scale<<" average contour size: "<<nContourSizeAvg<<endl;	//

			nContourArea += nContourSizeAvg;
			nPrevContourArea = nContourArea;
			nInit++;
		}

		
	
		//crDetect.show(HoughMap, Threshold);

		//HoughMap.clear();
		////Meanfilter.clear();
		//Threshold.clear();

		//프레임별 검출 박스 출력
		crDetect.vImgDetect.clear();
		/*for (int i = 0; i < vRecDetected.size(); i++){
		rectangle(crDetect.mTotalImg, vRecDetected[i], CV_RGB(255, 255, 0), 2);
		}*/

		//clustering
		clustering(sMaxPtRoi, 10);

		//verification
		//crDetect.VerifyDetection(vRecDetected, vecVerifiedVehicle);
		crDetect.VerifyDetection_check(sMaxPtRoi, vecVerifiedVehicle);
		Mat mTrackImg;
		mTrackImg=crDetect.mTotalImg.clone();

			for (int i = 0; i < vecVerifiedVehicle.size(); i++){
				rectangle(crDetect.mTotalImg, vecVerifiedVehicle[i], CV_RGB(255, 0, 0), 2);
				}

		circle(crDetect.mTotalImg, ptVanishing, 1, CV_RGB(255, 0, 0), 2);
		circle(mTrackImg, ptVanishing, 1, CV_RGB(255, 0, 0), 2);
		vector<Rect> vecDetect;
		vector<Rect> vecValidRec;
		Point pCurrent;
		float scaledist = SCALE_DIST;
		int cntCandidate = 2; // 후보 ROI 검증 횟수, 검증되면 tracking 시작 2
		int cntBefore = 4; // 못찾은 횟수가 연속으로 n프레임 이상이면 ROI 제거 4
		int frameCandiate = 4; // 못찾은 횟수가 연속으로 n프레임 이상이면 후보 ROI 제거
		vector<Rect> vecRectTracking;
		Mat imgROImask = Mat::zeros(crDetect.mTotalImg.size(), CV_8UC1);
		imgROImask(rect_TROI).setTo(Scalar::all(255));
		kalmanMultiTarget(crDetect.mTotalImg, vecVerifiedVehicle, High, MultiKF, scaledist, cntCandidate, cntBefore, frameCandiate, nIdx, rect_TROI, vecValidRec, imgROImask,crDetect);
		//vector<Rect> vecValid;
		crDetect.vecValid = High.vecBefore;

		for (int i = 0; i < crDetect.vecValid.size(); i++)
		{
			rectangle(mTrackImg, crDetect.vecValid[i], MultiKF[i].rgb, 3);
		}

		Size half = Size(crDetect.mTotalImg.cols / 2, crDetect.mTotalImg.rows / 2);
		resize(crDetect.mTotalImg, crDetect.mTotalImg, half);
		resize(mTrackImg, mTrackImg, half);
		imshow("detection", crDetect.mTotalImg);
		imshow("tracking", mTrackImg);

		crDetect.show(HoughMap, Threshold);

		HoughMap.clear();
		//Meanfilter.clear();
		Threshold.clear();

		double dTimeEnd = cvGetTickCount();
		double dTrainingTime = (double)(0.001 * (dTimeEnd - dTimeBegin) / cvGetTickFrequency());
		printf("[time]: %3.2f msec\n", dTrainingTime);

		char c = cvWaitKey(1);
		if (c == 27) exit(-1);

		crDetect.vec_mFeature.clear();
		crDetect.vecValid.clear();
		//crDetect.oVideoWriter.write(crDetect.mTotalImg);
		//cout << "frame# " << nIdx << endl;
	}
	//fclose(crDetect.pixelSize);
}

void kalmanMultiTarget(Mat& srcImage, vector<Rect>& vecRectTracking, Track_t& Set, vector<SKalman>& MultiKF, float scaledist, int cntCandidate, int cntBefore, int frameCandiate, int& cntframe, Rect& ROIset, vector<Rect>& vecValidRec, Mat& imgROImask, CRForestDetector& crDetect)
{
	Point pCurrent;
	
	//vecBefore: 여러번 검출된 candidate
	//vecBefore가 존재하는 경우
	//4.
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


		//kalmant tracking 결과인 MultiKF와 vecRectTracking 비교 후 filtering
		for (int j = 0; j < vecRectTracking.size(); j++)	
		{
			float fDistXCandi = ptVanishing.x - (vecRectTracking[j].x + vecRectTracking[j].width / 2);
			float fDistYCandi = ptVanishing.y - (vecRectTracking[j].y + vecRectTracking[j].height / 2);
			float fDistCandi = fDistXCandi*fDistXCandi + fDistYCandi*fDistYCandi;
			fDistCandi = sqrt(fDistCandi);
			float fAngleCandi = fAngle(fDistXCandi, fDistYCandi);
	
			// kalman tracking 결과와 1차 검출 사이 거리
			float dist = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x)*abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x)
				+ abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y)*abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y);
			dist = sqrt(dist);
			
			float pCurrnetXY = pCurrent.x*pCurrent.x + pCurrent.y*pCurrent.y;
			pCurrnetXY = sqrt(pCurrnetXY);
			
			//거리 정보 바탕
			Point2i ptCandiTL = Point2i(Set.vecBefore[i].x, Set.vecBefore[i].y);
			float fStandardWidth = crDetect.StandardVehicleWidth(ptCandiTL);

			//1차 검출과 소실점 사이 거리가 kalman tracking과 소실점사이 거리보다 작은 경우 
			//candidate과 vecRectTracking 사이 dx, dy를 pCurrent에 저장
			if (((dist < pCurrnetXY && dist < scaledist *Set.vecBefore[i].width /*&&abs(fDistXCandi - fDistXbefore)<fStandardWidth * TRACKING_MARGIN_RATIO && fDistYCandi>0*/ )&& (abs(fAngleBefore - fAngleCandi) < ANGLE_THRESH) ) /*&& vecRectTracking[j].area() >= Set.vecBefore[i].area()*0.8 && vecRectTracking[j].y + vecRectTracking[j].height / 2 - (Set.vecBefore[i].y + (Set.vecBefore[i].height) / 2) <= 1*/)
			{
				pCurrent.x = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x);
				pCurrent.y = abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y); 
				num = j;
			}
		}

		//이전 프레임에서 추적이 존재하는 경우
		//1차 검출과 tracking 결과의 거리 차이가 조건을 만족하지 않는 경우 
		if (num == -1 && MultiKF.size() > 0)
		{
			MultiKF[i].bOK = false;
			MultiKF[i].ptCenter.x = MultiKF[i].matPrediction.at<float>(0);
			MultiKF[i].ptCenter.y = MultiKF[i].matPrediction.at<float>(1);
			MultiKF[i].speedX = MultiKF[i].matPrediction.at<float>(2);
			MultiKF[i].speedY = MultiKF[i].matPrediction.at<float>(3);
			MultiKF[i].width = MultiKF[i].matPrediction.at<float>(4);
			MultiKF[i].height = MultiKF[i].matPrediction.at<float>(5);

			//vecBefore의 값을 tracking 결과로 대체 (좌표)
			Set.vecBefore[i].x = MultiKF[i].ptPredict.x - (Set.vecBefore[i].width) / 2;
			Set.vecBefore[i].y = MultiKF[i].ptPredict.y - (Set.vecBefore[i].height) / 2;

		}
		else if (Set.vecBefore.size() != 0)
		{
			//Kalman parameter 갱신
			MultiKF[i].bOK = true;
			MultiKF[i].speedX = vecRectTracking[num].x + (vecRectTracking[num].width) / 2 - (Set.vecBefore[i].x + (Set.vecBefore[i].width) / 2);
			MultiKF[i].speedY = vecRectTracking[num].y + (vecRectTracking[num].height) / 2 - (Set.vecBefore[i].y + (Set.vecBefore[i].height) / 2);

			Set.vecBefore[i] = vecRectTracking[num];
			vecRectTracking.erase(vecRectTracking.begin() + num);
			MultiKF[i].ptCenter.x = Set.vecBefore[i].x + (Set.vecBefore[i].width) / 2;
			MultiKF[i].ptCenter.y = Set.vecBefore[i].y + (Set.vecBefore[i].height) / 2;
			MultiKF[i].width = Set.vecBefore[i].width;
			MultiKF[i].height = Set.vecBefore[i].height;
			Set.vecCount[i] = cntframe; //vecCount 갱신

		}
		/*cout << "Set.vecBefore[i].y + Set.vecBefore[i].height : " << (Set.vecBefore[i].y + Set.vecBefore[i].height) << endl;
		cout << "Set.vecBefore[i].x + Set.vecBefore[i].width : " << Set.vecBefore[i].x + Set.vecBefore[i].width << endl;
		cout << "Set.vecBefore[i] : " << Set.vecBefore[i] << endl;*/

		bool bcheck = false;

		Point2i ptCandiTL = (Set.vecBefore[i].x, Set.vecBefore[i].y);
		float fStandardWidth = crDetect.StandardVehicleWidth(ptCandiTL);

		//	Tracking box 검증 
		// 반복 candidate과 tracking 결과 필터링
		if (Set.vecBefore[i].y < 0 || Set.vecBefore[i].x < 0 || Set.vecBefore[i].y + Set.vecBefore[i].height >= srcImage.rows || Set.vecBefore[i].x + Set.vecBefore[i].width >= srcImage.cols/* || abs(MultiKF[i].ptCenter.x - Set.vecBefore[i].x)> fStandardWidth * TRACKING_MARGIN_RATIO*//* && fDistYCandi>0*/)
		{
			Set.vecBefore.erase(Set.vecBefore.begin() + i);
			MultiKF.erase(MultiKF.begin() + i);
			Set.vecCount.erase(Set.vecCount.begin() + i);
			i--;
			bcheck = true;
		}

		if (bcheck == false && (/*imgROImask.at<uchar>((Set.vecBefore[i].y < 0 ? 0 : Set.vecBefore[i].y), (Set.vecBefore[i].x < 0 ? 0 : Set.vecBefore[i].x)) == 0 ||*/ imgROImask.at<uchar>((Set.vecBefore[i].y + Set.vecBefore[i].height < srcImage.rows ? Set.vecBefore[i].y + Set.vecBefore[i].height : srcImage.rows - 1), (Set.vecBefore[i].x + Set.vecBefore[i].width < srcImage.cols ? Set.vecBefore[i].x + Set.vecBefore[i].width : srcImage.cols - 1)) == 0 || (cntframe - Set.vecCount[i])>cntBefore) && Set.vecBefore.size() != 0 && MultiKF.size() != 0){
			Set.vecBefore.erase(Set.vecBefore.begin() + i);
			MultiKF.erase(MultiKF.begin() + i);
			Set.vecCount.erase(Set.vecCount.begin() + i);
			i--;
		}
	}
	///// Kalman filtering 시작 //////////////x
	//3. 
	for (int i = 0; i < Set.vecCandidate.size(); i++)
	{
		pCurrent.x = srcImage.cols;
		pCurrent.y = srcImage.rows;
		///////////
		int num = -1;
		//vecRectTracking: 새로 갱신된 1차 검출들
		for (int j = 0; j < vecRectTracking.size(); j++)
		{
			//Candidate과 vecRectTracking 사이 거리 계산 (dist)
			float dist = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2))*abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2))
				+ abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2))*abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2));
			dist = sqrt(dist);
			// 초기 pCurrentXY는 이미지 대각선 길이
			//
			float pCurrnetXY = pCurrent.x*pCurrent.x + pCurrent.y*pCurrent.y;
			pCurrnetXY = sqrt(pCurrnetXY);
			//candidate과 vecRectTracking 사이 거리가 pCurrentXY보다 작다면
			//거리가 candidate의 너비*비례상수 값보다 작은 경우
			if (dist < pCurrnetXY && dist < scaledist * 2 * Set.vecCandidate[i].width )
			{
				Point2i ptCandiTL = Point2i(vecRectTracking[j].x, vecRectTracking[j].y);
				float fStandardWidth = crDetect.StandardVehicleWidth(ptCandiTL);

				float fDistXCandi = ptVanishing.x - (vecRectTracking[j].x + vecRectTracking[j].width / 2);
				float fDistYCandi = ptVanishing.y - (vecRectTracking[j].y + vecRectTracking[j].height / 2);
				float fDistCandi = fDistXCandi*fDistXCandi + fDistYCandi*fDistYCandi;
				fDistCandi = sqrt(fDistCandi);
				float fAngleCandi = fAngle(fDistXCandi, fDistYCandi);
			/*	cout << "fStandardWidth : " << fStandardWidth << endl;
				cout << "dx : " <<abs( vecRectTracking[j].x - Set.vecCandidate[i].x )<< endl;*/
				
				//candidate과 소실점 사이 거리가 vecRectTracking과 소실점사이 거리보다 작은 경우 
				//candidate과 소실점 사이 각도와 vecRectTracking과 소실점사이 각도의 차이가 임계값보자 작은 경우
				//candidate과 vecRectTracking 사이 dx, dy를 pCurrent에 저장
				if (Set.vecRtheta[i].x < fDistCandi && abs(Set.vecRtheta[i].y - fAngleCandi) < ANGLE_THRESH/*&&abs(vecRectTracking[j].x - Set.vecCandidate[i].x)<fStandardWidth * TRACKING_MARGIN_RATIO*/)
				{
					pCurrent.x = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2));
					pCurrent.y = abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2));
					num = j;	//새로 갱신된 1차 검출의 index 저장 
				}
			}
		}
		int speedtempX = 0;
		int speedtempY = 0;
		//위 조건을 만족하는 candidate이 있는 경우
		if (num > -1 && vecRectTracking.size() > 0)
		{
			Set.vecCountPush[i].x++;			// 찾으면 +1
			Set.vecCountPush[i].y = cntframe;	// 찾았을때의 frame 갱신
			speedtempX = vecRectTracking[num].x - Set.vecCandidate[i].x;	//한프레임사이 x축에서 이동거리 
			speedtempY = vecRectTracking[num].y - Set.vecCandidate[i].y;	//한프레임사이 y축에서 이동거리
			Set.vecCandidate[i] = vecRectTracking[num];
			//////////
			//Candidate의 dist 값과 angle값을 매칭 1차 검출의 정보로 갱신
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
		// cntCandidate넘게 찾으면 새로운 target으로 인식 (candidate이 여러 프레임에서 검출된 경우)
		if (Set.vecCountPush[i].x >= cntCandidate) 
		{
			////
			//vecBefore : 최종 tracking target
			//tracking 관련 MultiKF 저장
			Set.vecBefore.push_back(Set.vecCandidate[i]);	//Candidate를 vecBefore에 저장
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
	
	//1.
	//vecRectTracking : 1차 검출 
	//1차 검출 box들을 candidate 선정
	
	for (int i = 0; i < vecRectTracking.size(); i++)
	{
		//Set.vecRtheta.push_back(

		float fDistXCandi = ptVanishing.x - (vecRectTracking[i].x + vecRectTracking[i].width / 2);
		float fDistYCandi = ptVanishing.y - (vecRectTracking[i].y + vecRectTracking[i].height / 2);
		float fDistCandi = fDistXCandi*fDistXCandi + fDistYCandi*fDistYCandi;
		fDistCandi = sqrt(fDistCandi);

		float fAngleCandi = fAngle(fDistXCandi, fDistYCandi);
		Set.vecRtheta.push_back(Point2f(fDistCandi, fAngleCandi));	//candidate의 각도 정보 저장
		Set.vecCandidate.push_back(vecRectTracking[i]);	//candidiate 벡터에 저장
		Set.vecCountPush.push_back(Point(0, cntframe));	//frame 정보 저장
	}
	//2. 
	//frameCandidate이상 연속으로 못찾으면 제거
	//cntframe: 현재 frame 번호
	//candidiate들의 frame 번호와 비교하면서 차이가 frameCandidate이상이면 candidate의 정보를 제거한다
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

	/// Kalman filtering 시작 //////////////
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

void clustering(vector<sMaxPts> sMaxPtRoi, int distance)
{

	for (int i = 0; i < sMaxPtRoi.size(); i++)
	{
		int cntSame = 0;
		for (int j = i + 1; j < sMaxPtRoi.size(); j++)
		{
			if (i == j)
				continue;
			int recX = abs(sMaxPtRoi[i].rectMax.x - sMaxPtRoi[j].rectMax.x);
			int recY = abs(sMaxPtRoi[i].rectMax.y - sMaxPtRoi[j].rectMax.y);
			int recW = abs(sMaxPtRoi[i].rectMax.x + sMaxPtRoi[i].rectMax.width - sMaxPtRoi[j].rectMax.x - sMaxPtRoi[j].rectMax.width);
			int recH = abs(sMaxPtRoi[i].rectMax.y + sMaxPtRoi[i].rectMax.height - sMaxPtRoi[j].rectMax.y - sMaxPtRoi[j].rectMax.height);

			if ((recX < distance && recY < distance && recW < distance && recH < distance) || (recX + recY + recW + recH) < distance * 3)
			{
				cntSame++;

				sMaxPtRoi[i].rectMax.x = (int)(sMaxPtRoi[i].rectMax.x*(cntSame + 1) + sMaxPtRoi[i].rectMax.x) / (cntSame + 2);
				sMaxPtRoi[i].rectMax.y = (int)(sMaxPtRoi[i].rectMax.y*(cntSame + 1) + sMaxPtRoi[j].rectMax.y) / (cntSame + 2);
				sMaxPtRoi[i].rectMax.width = (int)(sMaxPtRoi[i].rectMax.width*(cntSame + 1) + sMaxPtRoi[j].rectMax.width) / (cntSame + 2);
				sMaxPtRoi[i].rectMax.height = (int)(sMaxPtRoi[i].rectMax.height*(cntSame + 1) + sMaxPtRoi[j].rectMax.height) / (cntSame + 2);
				//cout <<"clustering : "<<i<<"-- "<<  vecRect[i].x<<" " <<vecRect[i].y <<" "<< vecRect[i].width <<" "<<vecRect[i].height<<endl;
				sMaxPtRoi.erase(sMaxPtRoi.begin() + j);
				j--;
			}
		}
		cntSame = 0;
	}
}

// Extract patches from training data
void extract_Patches(CRPatch& Train, CvRNG* pRNG) {	//	CRPatch Train(&cvRNG, p_width, p_height, 2);

	vector<string> vFilenames;
	vector<CvRect> vBBox;
	vector<vector<CvPoint> > vCenter;

	// load positive file list
	loadTrainPosFile(vFilenames, vBBox, vCenter);

	// load postive images and extract patches
	for (int i = 0; i<(int)vFilenames.size(); ++i) {

		if (i % 50 == 0) cout << i << " " << flush;

		if (subsamples_pos <= 0 || (int)vFilenames.size() <= subsamples_pos || (cvRandReal(pRNG)*double(vFilenames.size()) < double(subsamples_pos))) {

			// Load image
			IplImage *img = 0;
			img = cvLoadImage((trainpospath + "\\" + vFilenames[i]).c_str(), CV_LOAD_IMAGE_COLOR);
			if (!img) {
				cout << "Could not load image file: " << (trainpospath + "//" + vFilenames[i]).c_str() << endl;
				exit(-1);
			}

			// Extract positive training patches
			Train.extractPatches(img, samples_pos, 1, &vBBox[i], &vCenter[i]);

			// Release image
			cvReleaseImage(&img);

		}

	}
	cout << endl;

	// load negative file list
	loadTrainNegFile(vFilenames, vBBox);

	// load negative images and extract patches
	for (int i = 0; i<(int)vFilenames.size(); ++i) {

		if (i % 50 == 0) cout << i << " " << flush;

		if (subsamples_neg <= 0 || (int)vFilenames.size() <= subsamples_neg || (cvRandReal(pRNG)*double(vFilenames.size()) < double(subsamples_neg))) {

			// Load image
			IplImage *img = 0;
			img = cvLoadImage((trainnegpath + "/" + vFilenames[i]).c_str(), CV_LOAD_IMAGE_COLOR);

			if (!img) {
				cout << "Could not load image file: " << (trainnegpath + "/" + vFilenames[i]).c_str() << endl;
				exit(-1);
			}

			// Extract negative training patches
			if (vBBox.size() == vFilenames.size())
				Train.extractPatches(img, samples_neg, 0, &vBBox[i]);
			else
				Train.extractPatches(img, samples_neg, 0);

			// Release image
			cvReleaseImage(&img);

		}

	}
	cout << endl;
}


// Init and start detector
void run_detect() {
	// Init forest with number of trees
	CRForest crForest(ntrees);

	// Load forest
	crForest.loadForest(treepath.c_str());

	// Init detector
	CRForestDetector crDetect(&crForest, p_width, p_height);

	// run detector
	//detect_Revised_Mat(crDetect);
	detect_he(crDetect);

}

// Init and start training
void run_train() {
	// Init forest with number of trees
	CRForest crForest(ntrees);

	// Init random generator
	time_t t = time(NULL);
	int seed = (int)t;

	CvRNG cvRNG(seed);	//opencv Random Number Generator

	// Create directory
	string tpath(treepath);
	tpath.erase(tpath.find_last_of(PATH_SEP));	//treetable

	
	int nResult = mkdir(tpath.c_str());
	if (nResult == -1) {
		cout << "Error: You can't make directory for tree" << endl;
		exit(-1);
	}
	else cout << "Tree directory has made" << endl;

	// Init training data
	CRPatch Train(&cvRNG, p_width, p_height, 2); //resize patch vector to contain 2 elements

	// Extract training patches
	extract_Patches(Train, &cvRNG);

	// Train forest
	crForest.trainForest(20, 15, &cvRNG, Train, 2000);

	// Save forest
	crForest.saveForest(treepath.c_str(), off_tree);

}

int main(/*int argc, char* argv[]*/)
{
	int mode = 2;

	cout << "Usage: CRForest-Detector.exe mode [config.txt] [tree_offset]" << endl;
	cout << "mode: 0 - train; 1 - show; 2 - detect" << endl;
	cout << "tree_offset: output number for trees" << endl;
	cout << "Load default: mode - 2" << endl;
	//cin >> mode;

	off_tree = 0;
	loadConfig("config2.txt", mode);

	switch (mode) {
	case 0:
		// train forest
		run_train();
		break;

	case 1:
		// train forest
		show();
		break;

	default:

		// detection
		run_detect();
		break;
	}


	return 0;
}





