#include <iostream>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdlib.h>
#include "CRForestDetector.h"
//#include "global.h"
//#include "head.h"
//#include "struct.h"

using namespace std;
using namespace cv;

bool CRForestDetector::CheckContourSize(Rect rectCheck, int nTrainWidth, int nTrainHeight){
	int nSize = (rectCheck.width)*(rectCheck.height);
	int nStandardSize = nTrainWidth*nTrainHeight;
	
	if(nSize>=nStandardSize*SIZE_RATIO) return true;
	else return false;	
}


bool CRForestDetector::CheckContourSize_Mat(Mat mCheck){
	
	int nPixelCnt = 0;
	for(int i=0;i<mCheck.rows;i++){
		for(int j=0;j<mCheck.cols;j++){
			if(mCheck.at<int>(j,i)>0){
				nPixelCnt++;
			}
		}
	}
	cout<<"contour pixel: "<<nPixelCnt<<endl;
	if(nPixelCnt<5)return false;
	
}

void CRForestDetector::InitializeROI(CRForestDetector& crDetect){
	//cvlab_1,2x,3x,4x,7x
	crDetect.vec_RoiRowRatio.push_back(0.72);
	crDetect.vec_RoiRowRatio.push_back(0.62); //0.56
	crDetect.vec_RoiRowRatio.push_back(0.57);	//0.53

	////cvlab_5,6,8
	//crDetect.vec_RoiRowRatio.push_back(0.72);
	//crDetect.vec_RoiRowRatio.push_back(0.57); 
	//crDetect.vec_RoiRowRatio.push_back(0.53);

	////cvlab_1,2x,3x,4x,7x
	//crDetect.vec_RoiRowRatio.push_back(0.72);
	//crDetect.vec_RoiRowRatio.push_back(0.62); //0.56
	//crDetect.vec_RoiRowRatio.push_back(0.57);	//0.53

	////reduce size of ROI1
	//crDetect.vec_RoiRowRatio.push_back(0.67);
	//crDetect.vec_RoiRowRatio.push_back(0.62); //0.56
	//crDetect.vec_RoiRowRatio.push_back(0.57);	//0.53

	crDetect.vec_RoiSize.push_back(Size(426, 145));
	//crDetect.vec_RoiSize.push_back(Size(426, 127));
	crDetect.vec_RoiSize.push_back(Size(308, 64));
	crDetect.vec_RoiSize.push_back(Size(180, 37));

	crDetect.vec_ORoiSize.push_back(Size(852, 290)); // original ROI1 size
	//crDetect.vec_ORoiSize.push_back(Size(852, 254));
	crDetect.vec_ORoiSize.push_back(Size(616, 128));
	crDetect.vec_ORoiSize.push_back(Size(360, 74));
}


void CRForestDetector::InitializeROI_scale( vector<float> scale){
	ptMain = Point2i(640,360);

	vec_RoiY.push_back(472); //resize 0.34
	vec_RoiY.push_back(436);
	vec_RoiY.push_back(410);
	vec_RoiY.push_back(392);	//resize 1.0

	vec_SizeROI.resize(scale.size());
	vec_PointROI.resize(scale.size());

	vec_SizeROI[0].width = FIRST_ROI_WIDTH;	//resize 전 initial ROI width
	
	for (int i = 0; i < scale.size(); i++){
		vec_SizeROI[i].height = sizeTrainImg.width/scale[i];
		vec_PointROI[i].y = vec_RoiY[i] - vec_SizeROI[i].height;
	}

	for (int i = 1; i < scale.size(); i++){
		vec_SizeROI[i].width = (vec_RoiY[i] - ptMain.y)*vec_SizeROI[0].width / (vec_RoiY[0] - ptMain.y);
		}

	for (int i = 0; i < scale.size(); i++){
		vec_PointROI[i].x = (IMAGE_WIDTH - vec_SizeROI[i].width) / 2;
		vec_rectROI.push_back(Rect(vec_PointROI[i], vec_SizeROI[i]));
	}
}


vector<IplImage* > CRForestDetector::CropNScaleRoi(IplImage* imgInput, vector<float> scale){	//scale 안씀
	//resized 이미지를 입력으로 입력받음.
	//iplimage --> mat  //   mat --> Iplimage
	Mat mInput(imgInput);
	Rect_<int> rect_ROI_2m = Rect(106,117,426,144);
	Rect_<int> rect_ROI_9m = Rect(146, 153, 308, 64);
	Rect_<int> rect_ROI_22m = Rect(216, 162, 180, 37);

	/*Mat mRoi_2m = mInput(rect_ROI_2m); 
	Mat mRoi_9m = mInput(rect_ROI_9m);
	Mat mRoi_22m = mInput(rect_ROI_22m);*/

	Mat mRoi_2m_scale, mRoi_9m_scale, mRoi_22m_scale; 

	 mRoi_2m_scale = mInput(rect_ROI_2m);
	 mRoi_9m_scale = mInput(rect_ROI_9m);
	 mRoi_22m_scale = mInput(rect_ROI_22m);

	//resize(mRoi_2m, mRoi_2m_scale, Size(rect_ROI_2m.width/**scale[0]*/, rect_ROI_2m.height/**scale[0]*/));
	//resize(mRoi_9m, mRoi_9m_scale, Size(rect_ROI_9m.width/**scale[1]*/, rect_ROI_9m.height/**scale[1]*/));
	//resize(mRoi_22m, mRoi_22m_scale, Size(rect_ROI_22m.width/**scale[2]*/, rect_ROI_22m.height/**scale[2]*/));

	IplImage* imgRoi1, *imgRoi2, *imgRoi3;
	imgRoi1 = cvCreateImage(cvSize(mRoi_2m_scale.cols, mRoi_2m_scale.rows), imgInput->depth, imgInput->nChannels);
	imgRoi2 = cvCreateImage(cvSize(mRoi_9m_scale.cols, mRoi_9m_scale.rows), imgInput->depth, imgInput->nChannels);
	imgRoi3 = cvCreateImage(cvSize(mRoi_22m_scale.cols, mRoi_22m_scale.rows), imgInput->depth, imgInput->nChannels);

	IplImage iplTmp = mRoi_2m_scale;
	cvCopy(&iplTmp, imgRoi1);
	iplTmp = mRoi_9m_scale;
	cvCopy(&iplTmp, imgRoi2);
	iplTmp = mRoi_22m_scale;
	cvCopy(&iplTmp, imgRoi3);

	vector<IplImage*> CropedNScaledRois;
	CropedNScaledRois.push_back(imgRoi1);
	CropedNScaledRois.push_back(imgRoi2);
	CropedNScaledRois.push_back(imgRoi3);
	
	return CropedNScaledRois;
}
vector<IplImage*> CRForestDetector::CropNScaleRoi_Reivised(IplImage* imgInput, vector<float> scale){
	Rect_<int> rect_ROI;
	Mat mRoi;
	Size sizeRoi;
	IplImage* imgRoi;
	vector<IplImage*> CropedRois;
	
	Mat mInput(imgInput);
	
	for (int i = 0; i < scale.size(); i++){
		//resize with scale
		Mat mTemp;
		mInput.copyTo(mTemp);

		sizeRoi = Size((int)(mInput.cols*scale[i]), (int)(mInput.rows*scale[i]));
		resize(mTemp, mTemp, sizeRoi);

		//crop ROI
		int nX = sizeRoi.width/2-vec_RoiSize[i].width/2;
		int nY = sizeRoi.height*vec_RoiRowRatio[i]/2-vec_RoiSize[i].height/2;
		rect_ROI = Rect(nX, nY, vec_RoiSize[i].width, vec_RoiSize[i].height);
		mRoi = mTemp(rect_ROI);

		imgRoi = cvCreateImage(cvSize(mRoi.cols, mRoi.rows), imgInput->depth, imgInput->nChannels);
		CropedRois.push_back(imgRoi);
	}
	return CropedRois;
}

vector<Mat> CRForestDetector::CropNScaleRoi_Reivised_Mat(Mat imgInput, vector<float> scale){	
	Rect_<int> rect_ROI;
	Mat mRoi;
	Size sizeRoi;
	IplImage* imgRoi;
	vector<Mat> CropedRois;

	Mat mInput(imgInput);

	for (int i = 0; i < scale.size(); i++){
		//resize with scale
		Mat mTemp;
		mInput.copyTo(mTemp);

		sizeRoi = Size((int)(mInput.cols*scale[i]), (int)(mInput.rows*scale[i]));
		resize(mTemp, mTemp, sizeRoi);

		//crop ROI 
		int nX = sizeRoi.width / 2 - vec_RoiSize[i].width / 2;
		int nY = sizeRoi.height*vec_RoiRowRatio[i] / 2 - vec_RoiSize[i].height / 2;
		rect_ROI = Rect(nX, nY, vec_RoiSize[i].width, vec_RoiSize[i].height);
		mRoi = mTemp(rect_ROI);

		CropedRois.push_back(mRoi);
	}
	return CropedRois;
}

vector<Mat> CRForestDetector::CropNScaleRoi_Reivised_Mat2(Mat imgInput, vector<float> scale){	//cropped image
	Rect_<int> rect_ROI;
	Mat mRoi, mTmp;
	Size/* sizeRect,*/sizeRoi;
	IplImage* imgRoi;
	vector<Mat> CropedRois;


	for (int i = 0; i < scale.size(); i++){
		//resize with scale
		Mat mTemp;
		imgInput.copyTo(mTemp);
		if (i == 0){
			mRoi = imgInput;
		}
		else{
			//crop
			int nX = (vec_ORoiSize[0].width - vec_ORoiSize[i].width) / 2;
			int nY = (vec_ORoiSize[0].height - mTotalImg.rows*(vec_RoiRowRatio[0] - vec_RoiRowRatio[i]))/2 ;
			//int nY = (vec_ORoiSize[0].height - mTotalImg.rows*(vec_RoiRowRatio[0] - vec_RoiRowRatio[i])-vec_ORoiSize[i].height) ;

			vec_Roi_tl.push_back(Point2i(nX, nY));
			

			rect_ROI = Rect(nX, nY, vec_ORoiSize[i].width, vec_ORoiSize[i].height);
			mRoi = imgInput(rect_ROI);
		}
		//resize
		sizeRoi = Size((int)(mRoi.cols*scale[i]), (int)(mRoi.rows*scale[i]));
		resize(mRoi, mRoi, sizeRoi);
		
		CropedRois.push_back(mRoi);
	}
	return CropedRois;
}


vector<Mat> CRForestDetector::CropNROI_scale(Mat imgInput, vector<float> scale){	//cropped image
	Mat mRoi;
	Size sizeRoi;
	vector<Mat> CropedRois;

	for (int i = 0; i < scale.size(); i++){
		//crop
		mRoi = imgInput(vec_rectROI[i]);
		//resize
		sizeRoi = Size((int)(mRoi.cols*scale[i]), (int)(mRoi.rows*scale[i]));
		resize(mRoi, mRoi, sizeRoi);
		CropedRois.push_back(mRoi);
	}

	return CropedRois;
}


int CRForestDetector::StandardVehicleWidth(Point2i pBottomLeft){
	int nStandardWidth = 0;

	//Resize 1
	if (pBottomLeft.x >= 0 & pBottomLeft.x < 580){
		nStandardWidth = 1.4059*pBottomLeft.y - 510.3;
	}
	else if (pBottomLeft.x >= 580 & pBottomLeft.x < 716){
		nStandardWidth = 1.4639*pBottomLeft.y - 514.46;
	}
	else if (pBottomLeft.x >= 716 & pBottomLeft.x < 1280){
		nStandardWidth = 1.4487*pBottomLeft.y - 519.62;
	}

	////Resize 0.7
	//if (pBottomLeft.x >= 0 & pBottomLeft.x < 406){
	//	nStandardWidth = 1.4059*pBottomLeft.y - 357.21;
	//}
	//else if (pBottomLeft.x >= 406 & pBottomLeft.x < 501){
	//	nStandardWidth = 1.4639*pBottomLeft.y - 360.12;
	//}
	//else if (pBottomLeft.x >= 501 & pBottomLeft.x < 896){
	//	nStandardWidth = 1.4487*pBottomLeft.y - 363.74;
	//}
	if (nStandardWidth <= 15) nStandardWidth = 15;
	
	return nStandardWidth;	//pixel value
}

void CRForestDetector::VerifyDetection(const vector<Rect_<int>>& vecDetectVehicle, vector<Rect_<int>>& vecVerifiedVehicle){

	//차량 후보 검증
	for (int i = 0; i < vecDetectVehicle.size(); i++){
		int nx = 0, ny = 0, nWidth = 0, nHeight = 0, nStandardWidth = 0, diff = 0;
		Point2i pBottomRight = (0, 0);
		int nMargin = 0;

		pBottomRight.x = vecDetectVehicle.at(i).br().x;
		pBottomRight.y = vecDetectVehicle.at(i).br().y;
		nWidth = vecDetectVehicle.at(i).width;
		//nHeight = vecDetectVehicle.at(i).height;

		//pCenter.x = (int)(nx - nWidth / 2);
		//pCenter.y = (int)(ny - nHeight / 2);
		//
		////Calculate vehicle center

		nStandardWidth = StandardVehicleWidth(pBottomRight);

		diff = abs(nStandardWidth - nWidth);
		nMargin = nStandardWidth*MARGIN_RATIO;
		//nMargin = nStandardWidth*0.5;
		//printf("diff: %d	Margin:%d	br.x: %d	br.y: %d	width: %d\n", diff, nMargin, pBottomRight.x, pBottomRight.y, nWidth);
		//printf("diff: %d	nStandardWidth:%d	br.x: %d	br.y: %d	width: %d\n", diff, nStandardWidth, pBottomRight.x, pBottomRight.y, nWidth);

		if (diff <= nMargin){
			vecVerifiedVehicle.push_back(vecDetectVehicle.at(i));
		}
	}
}

void CRForestDetector::VerifyDetection_check(const vector<sMaxPts>& vecMaxRect, vector<Rect_<int>>& vecVerifiedVehicle){

	//차량 후보 검증
	for (int i = 0; i < vecMaxRect.size(); i++){
		int nx = 0, ny = 0, nWidth = 0, nHeight = 0, nStandardWidth = 0, diff = 0;
		Point2i pBottomRight = (0, 0);
		int nMargin = 0;

		pBottomRight.x = vecMaxRect.at(i).rectMax.br().x;
		pBottomRight.y = vecMaxRect.at(i).rectMax.br().y;
		nWidth = vecMaxRect.at(i).rectMax.width;
		//nHeight = vecDetectVehicle.at(i).height;

		//pCenter.x = (int)(nx - nWidth / 2);
		//pCenter.y = (int)(ny - nHeight / 2);
		//
		////Calculate vehicle center

		nStandardWidth = StandardVehicleWidth(pBottomRight);

		diff = abs(nStandardWidth - nWidth);
		nMargin = nStandardWidth*MARGIN_RATIO;
		//nMargin = nStandardWidth*0.5;
		//printf("diff: %d	Margin:%d	br.x: %d	br.y: %d	width: %d\n", diff, nMargin, pBottomRight.x, pBottomRight.y, nWidth);
		//printf("diff: %d	nStandardWidth:%d	br.x: %d	br.y: %d	width: %d\n", diff, nStandardWidth, pBottomRight.x, pBottomRight.y, nWidth);

		if (diff <= nMargin){
			vecVerifiedVehicle.push_back(vecMaxRect.at(i).rectMax);
		}
	}
}

void CRForestDetector::Normalization(Mat& mHoughSrc, Mat& mHoughDst){
	float fValue = 0; 
	float fSigValue = 0;

	for (int i = 0; i < mHoughSrc.rows; i++){
		for (int j = 0; j < mHoughSrc.cols; j++){
			fValue = mHoughSrc.at<float>(i, j);
			fSigValue = 1 / (1 + exp(5 - fValue));
			mHoughDst.at<float>(i, j) = fSigValue;
		}
	}
}

void CRForestDetector::show(vector<Mat> mat_hough, vector<Mat> mat_threshold){

	imshow("HoughMap 1", mat_hough[0]);
	imshow("HoughMap 2", mat_hough[1]);
	imshow("HoughMap 3", mat_hough[2]);
	imshow("HoughMap 4", mat_hough[3]);

	imshow("Threshold 1", mat_threshold[0]);
	imshow("Threshold 2", mat_threshold[1]);
	imshow("Threshold 3", mat_threshold[2]);
	imshow("Threshold 4", mat_threshold[3]);

}

int CRForestDetector::contourThreshold(int threshold){
	float fAlpha = threshold / (float)255;
	int  nContourSizeTheshold = 50 * pow(10, fAlpha - 1);
	return nContourSizeTheshold;
}