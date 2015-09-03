/*
*  main DMclass_test.cpp
*  measure the distace of detected object
*
*  Created by T.K.Woo on July/20/2015.
*  Copyright 2015 CVLAB at Inha. All rights reserved.
*
*/
#include "DistMeasure.h"
#include<iostream>
#include<opencv2\opencv.hpp>
#include<string.h>
#include<fstream>
#include<math.h>

using namespace std;
using namespace cv;


#define BASELINE 0.25 // unit : m
#define FOCAL 1200 // unit : pixels   //240x10^-5:0.012=240:f

#define NUM_DISP 48
#define PI 3.141592
#define BOUND_DIST 20 //unit : m

int main()
{
	//////////////////////////////////// file load ////////////////////////////////////////////
	string strDBFilePath = "F:/00.workspace/2014-2연구실/sane/StereoVision_sane/StereoVision_sane/DB2015/공인DB/documentation/GroundTruth/"; // *.db file loading
	string strDBFileName = "GroundTruth2D_part1_stereo.db";
	string str3dDBFileName = "GroundTruth3D_part1_stereo.db";

	//string strLeftImagePath = "./공인DB/TestData_c0_part2/TestData/c0/"; // image file path
	//string strRightImagePath = "./공인DB/TestData_c1_part2/TestData/c1/"; // image file path
	string strLeftImagePath = "F:/00.workspace/2014-2연구실/sane/StereoVision_sane/StereoVision_sane/DB2015/공인DB/TestData_c0_part1/TestData/c0/"; // image file path
	string strRightImagePath = "F:/00.workspace/2014-2연구실/sane/StereoVision_sane/StereoVision_sane/DB2015/공인DB/TestData_c1_part1/TestData/c1/";

	FILE *fp = fopen((strDBFilePath + strDBFileName).c_str(), "rt");
	FILE *fp3D = fopen((strDBFilePath + str3dDBFileName).c_str(), "rt");
	FILE *fpDistErr = fopen("DaimlerPart1_dist.txt", "wt");

	char temp[100];
	char temp3d[1000];
	for (int i = 0; i<5; i++)              // first to fifth line will be ignore 
	{
		fscanf(fp, "%s", &temp);
		fscanf(fp3D, "%s", &temp);
	}

	//////////////////////////////////// variable /////////////////////////////////////////////
	Mat imgLeftInput, imgRightInput;
	Mat imgDisplay, imgDisparity, imgLeftGT;
	int ww, hh;
	int zero;

	enum { STEREO_BM = 0, STEREO_SGBM = 1 };
	bool alg = STEREO_BM;
	int SADWindowSize = 0;
	//const int numberOfDisparities = NUM_DISP;
	bool no_display = false;

	double dtime, dtime_aver = 0;
	int nObjFrameCnt = 0;

	int color_mode = alg == STEREO_BM ? 0 : -1;

	//////////////////////////////////////processing////////////////////////////////////////////
	////////////////////////////////////// off line ////////////////////////////////////////////
	// class param setting
	CDistMeasure objDistMeasure;
	//objDistMeasure.SetParam(BASELINE, FOCAL, 0, -1.8907);
	//objDistMeasure.m_nDistAlg = CDistMeasure::STEREOBM;
	//objDistMeasure.m_flgVideo = false; // video or image
	objDistMeasure.SetParam(CDistMeasure::Daimler);
	objDistMeasure.m_flgVideo = false; // video or image

	//////////////////////////////////// on line /////////////////////////////////////////////

	int cntframe = 0;
	while (1)   
	{
		////////////////////////////////DB read///////////////////////////////////////////////////////////
		cntframe++;
		fscanf(fp, "%s", &temp);
		fscanf(fp3D, "%s", &temp3d);
		imgLeftInput = imread(strLeftImagePath + (string)temp, CV_LOAD_IMAGE_GRAYSCALE);
		imgRightInput = imread(strRightImagePath + (string)temp, CV_LOAD_IMAGE_GRAYSCALE);
		cout << (string)temp << endl;
		if (imgLeftInput.empty() || imgRightInput.empty())
		{
			cout << "no images" << endl;
			break;
		}

		//int Count_Frame = 0;	
		//Count_Frame++;
		//printf("%d", Count_Frame);

		cvtColor(imgLeftInput, imgLeftGT, CV_GRAY2BGR);
		fscanf(fp, "%d", &ww);
		fscanf(fp, "%d", &hh);
		fscanf(fp, "%d", &zero);
		fscanf(fp3D, "%d", &temp3d);
		fscanf(fp3D, "%d", &temp3d);
		fscanf(fp3D, "%d", &temp3d);

		int numOfObj;
		int numOfObj3d;
		fscanf(fp, "%d", &numOfObj);
		fscanf(fp3D, "%d", &numOfObj3d);
		//if(numOfObj==numOfObj3d) {cout << " good " << endl;return 0;}

		vector<Rect_<int>> vecRectGT;
		vector<Mat> vecImgRoiDisp8;
		vector<double> vecdRoiDistance;
		vector<double> vecdRoiDistGT;

		
		//Rect_<int> rect;

		for (int i = 0; i<numOfObj; i++)
		{
			char sharp[20];
			char ques[20];
			int objectClass; int objectClass3d;
			int objectID; int objectID3d;
			int uniqueID; int uniqueID3d;
			double confi; double confi3d;

			fscanf(fp, "%s", &sharp);
			fscanf(fp3D, "%s", &ques);
			if (ques[0] == '?') {
				//cout << "goodgood" << endl;
				fscanf(fp3D, "%d", &objectID3d);
				fscanf(fp3D, "%d", &uniqueID3d);
				fscanf(fp3D, "%lf", &confi3d);
				double dDepthGT = 0;
				double dtemp = 0;
				fscanf(fp3D, "%lf", &dtemp);
				fscanf(fp3D, "%lf", &dtemp);
				fscanf(fp3D, "%lf", &dDepthGT);
				fscanf(fp3D, "%lf", &dtemp);
				fscanf(fp3D, "%lf", &dtemp);
				fscanf(fp3D, "%lf", &dtemp);

				fscanf(fp, "%d", &objectClass);
				fscanf(fp, "%d", &objectID);
				fscanf(fp, "%d", &uniqueID);
				fscanf(fp, "%lf", &confi);
				Rect_<int> rect; Rect_<int> rect3d;
				int x2, y2; int x2_3d, y2_3d;
				fscanf(fp, "%d", &rect.x);
				fscanf(fp, "%d", &rect.y);
				fscanf(fp, "%d", &x2);
				fscanf(fp, "%d", &y2);
				rect.width = x2 - rect.x;
				rect.height = y2 - rect.y;
				if (objectClass == 0)
				{
					vecdRoiDistGT.push_back(dDepthGT);
					vecRectGT.push_back(rect);
					fscanf(fp, "%d", &zero);
				}
				else{
					fscanf(fp, "%d", &zero);
				}
				continue;
			}

			fscanf(fp, "%d", &objectClass);
			fscanf(fp3D, "%d", &objectClass3d);

			fscanf(fp, "%d", &objectID);
			fscanf(fp3D, "%d", &objectID3d);

			fscanf(fp, "%d", &uniqueID);
			fscanf(fp3D, "%d", &uniqueID3d);

			fscanf(fp, "%lf", &confi);
			fscanf(fp3D, "%lf", &confi3d);

			Rect_<int> rect; Rect_<int> rect3d;
			int x2, y2; int x2_3d, y2_3d;
			fscanf(fp, "%d", &rect.x);
			fscanf(fp, "%d", &rect.y);
			fscanf(fp, "%d", &x2);
			fscanf(fp, "%d", &y2);
			fscanf(fp3D, "%d", &rect3d.x); fscanf(fp3D, "%d", &rect3d.y); fscanf(fp3D, "%d", &x2_3d); fscanf(fp3D, "%d", &y2_3d);

			rect.width = x2 - rect.x;
			rect.height = y2 - rect.y;

			//	if (objectClass == 0)
			fscanf(fp, "%d", &zero);
			fscanf(fp3D, "%d", &zero);
		}
		fscanf(fp, "%s", &temp);
		fscanf(fp3D, "%s", &temp3d);
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////// stereo matching process ///////////////////////////////////////
		imgLeftGT.copyTo(imgDisplay);

		int64 t = getTickCount();
		if (vecRectGT.size() != 0){
			objDistMeasure.SetImage(imgLeftInput, imgRightInput, vecRectGT);
			objDistMeasure.CalcDistImg(CDistMeasure::FVLM);
		}

		for (int i = 0; i < objDistMeasure.m_vecdDistance.size(); i++)
		{
			cout << "dist : " << objDistMeasure.m_vecdDistance[i] << endl;
		}
		
		t = getTickCount() - t;
		dtime = t * 1000 / getTickFrequency();
		printf("image , Time elapsed: %fms\n", dtime);
		if (vecdRoiDistance.size() != 0) { dtime_aver += dtime; nObjFrameCnt++; }

		
		
		
		if (cntframe > 4000) {
			break;
		}
	}
	dtime_aver /= nObjFrameCnt;
	cout << "time aver : " << dtime_aver << "ms" << endl;
	fclose(fpDistErr);
	fclose(fp3D);
	fclose(fp);
	return 0;
}




