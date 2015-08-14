//승준 다차선모드 복붙

/* 
[LYW_0815]
영완 코드 다차선 시도 시작 :
*/ 

#include "highgui.h"
#include "cv.h"
#include "opencv2/opencv.hpp"
#include "MultiROILaneDetection_autocalib.h"
#include <vector>

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <iomanip>
#include <ctime>
#include<math.h>
using namespace cv;
using namespace std;

#define EVALUATION 0

//
#define PI 3.14159265358979323846
#define FIRSTFRAMENUM 1
#define RESIZEFACTOR 1


//#define AUTOX 100
#define AUTOX 0
#define AUTOY 200
//#define AUTOWIDTH 400
#define AUTOWIDTH 639
#define AUTOHEIGHT 80


#define CV2X 0
#define CV2Y 185
#define CV2WIDTH 639
#define CV2HEIGHT 70

// 0:png, 1:Video
enum FRAMETYPE{
	PNG,
	VIDEO
};
enum DB_INTRINSIC{
	CVLAB,  //CVLAB 자체DB (Urban and Expressway)
	CVLAB2,
	CVLAB3,
	PRESCAN,
	AMOL	//AMOL 제공 DB (Only Urban)
};
//#define INTRINSIC_DB PRESCAN
#define INTRINSIC_DB CVLAB
enum DB_CVLAB{
	CV1,//4월 CVLAB DB
	CV2,//3월 CVLAB DB
	CV3,
	CV4
};
#define DB_CVINIT CV1        

enum DB_ROAD{
	URBAN,
	EXPRESSWAY
};
#define DB_ROADINFO URBAN
//#define DB_ROADINFO EXPRESSWAY

string g_strOriginalWindow = "OriginalImg";


int nFlagInputFrameType;

int g_nResizeFacor;
double g_dTotlaTick = 0;
double d_totalProcessTime = 0;
int nCntProcess = 0;
int g_nAutoCalibFrameNumber = 90;
int DB_NUM = INTRINSIC_DB;
int INIT_CV = DB_CVINIT;
int INIT_ROADINFO = DB_ROADINFO;
int main()
{
	//g_nResizeFacor = RESIZEFACTOR;
	if (DB_NUM == AMOL) // VGA
	{
		g_nResizeFacor = 1;
	}
	if (DB_NUM == CVLAB) //HD
	{
		g_nResizeFacor = 2;
	}
	if (DB_NUM == PRESCAN) //HD
	{
		g_nResizeFacor = 2;
	}
	//test
	//end test
	nFlagInputFrameType = 0;
	//char szPrescanDB_dir[200] = "../../2015_PreScanDB/SimResults_20150310_144214_height_1200_tilt_4/CameraSensor_1/CameraSensor_1_";
	//char szPrescanDB_dir[200] = "../../2015_PreScanDB/SimResults_20150310_144506_height_1200_tilt_6/CameraSensor_1/CameraSensor_1_";
	//char szPrescanDB_dir[200] = "../../2015_PreScanDB/SimResults_20150327_214946_height_1200_tilt_4_lane_Change/CameraSensor_1/CameraSensor_1_";
	//char szPrescanDB_dir[200] = "../../2015_PreScanDB/SimResults_20150327_192252_height_1200_tilt_6_lane_Change/CameraSensor_1/CameraSensor_1_";
	//char szPrescanDB_dir[200] = "../../[DB]/2015지경부무인자동차/01_차선이탈/01_30kph/01_좌측이탈/스테레오카메라 Left BMP/DB";
	char szPrescanDB_dir[200];// = "H:/[DB]amol/S3C1_CAM0_IMG/S3C1_CAM0_IMG/S3C1_";
	//char szPrescanDB_dir[200] = "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_1/2015-04-13-14h-20m-45s_straight_";
	if (DB_NUM == AMOL)
	{
		strcpy(szPrescanDB_dir, "H:/[DB]amol/S3C1_CAM0_IMG/S3C1_CAM0_IMG/S3C1_");
	}
	if (DB_NUM == CVLAB)
	{
		if (INIT_CV == CV1)
		{
			if (INIT_ROADINFO == URBAN){
				//일반도로 4월
				//strcpy(szPrescanDB_dir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_1/2015-04-13-14h-20m-45s_straight_");
				//strcpy(szPrescanDB_dir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_2/2015-04-13-14h-20m-45s_straight_2_");
				strcpy(szPrescanDB_dir, "./[DB]FreeScaleDemo/Cloudy/Urban/Straight_2/2015-04-13-14h-20m-45s_straight_2_");
				//strcpy(szPrescanDB_dir, "D:/02.project/MultiROILaneDetectionVS2013/MultiROILaneDetectionVS2013/[DB]FreeScaleDemo/Cloudy/Urban/Straight_2/2015-04-13-14h-20m-45s_straight_2_");

			}
			if (INIT_ROADINFO == EXPRESSWAY)
			{
				//고속도로 4월
				strcpy(szPrescanDB_dir, "./[DB]CVLAB_Lane/Cloudy/Expressway/Straight_1/2015-04-13-14h-50m-44s_F_normal_");
			}
		}
		if (INIT_CV == CV2){
			if (INIT_ROADINFO == URBAN){
				cout << "NONE" << endl;
			}
			if (INIT_ROADINFO == URBAN){
				//일반도로 3월
				strcpy(szPrescanDB_dir, "H:/[DB]CVLAB_Lane/Purity/Expressway/Straight_1/2015-03-07-16h-48m-15s_F_event_");
			}

			if (INIT_ROADINFO == EXPRESSWAY)
			{
				//고속도로 3월
				strcpy(szPrescanDB_dir, "H:/[DB]CVLAB_Lane/Purity/Expressway/Straight_1/2015-03-07-16h-48m-15s_F_event_");
				//strcpy(szPrescanDB_dir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_2/2015-04-13-14h-20m-45s_straight_2_");
			}


		}


	}
	else if (DB_NUM == PRESCAN){
		strcpy(szPrescanDB_dir, "./[DB]AutoCalibration/Pitch_0_Yaw_0/Pitch_0_Yaw_0_");
	}
	//////////////////////////////////////////////////////////////////////////
	//test DB
	//char szTestDir[200] = "H:/[DB]amol/S1C1_CAM0_IMG/S1C1_CAM0_IMG/S1C1_";
	//char szTestDir[200] = "H:/[DB]amol/S1C2_CAM0_IMG/S1C2_CAM0_IMG/S1C2_";
	//char szTestDir[200] = "H:/[DB]amol/S2C1_CAM0_IMG/S2C1_CAM0_IMG/S2C1_";
	//char szTestDir[200] = "H:/[DB]amol/S2C2_CAM0_IMG/S2C2_CAM0_IMG/S2C2_";
	//char szTestDir[200] = "H:/[DB]amol/S2C3_CAM0_IMG/S2C3_CAM0_IMG/S2C3_";
	//char szTestDir[200] = "H:/[DB]amol/S2C4_CAM0_IMG/S2C4_CAM0_IMG/S2C4_";
	//char szTestDir[200] = "H:/[DB]amol/S2C5_CAM0_IMG/S2C5_CAM0_IMG/S2C5_";
	//char szTestDir[200] = "H:/[DB]amol/S2C6_CAM0_IMG/S2C6_CAM0_IMG/S2C6_";
	//char szTestDir[200] = "H:/[DB]amol/S2C7_CAM0_IMG/S2C7_CAM0_IMG/S2C7_";
	//char szTestDir[200] = "H:/[DB]amol/S3C1_CAM0_IMG/S3C1_CAM0_IMG/S3C1_";
	//char szTestDir[200] = "H:/[DB]amol/S3C2_CAM0_IMG/S3C2_CAM0_IMG/S3C2_";
	//char szTestDir[200] = "H:/[DB]amol/S3C3_CAM0_IMG/S3C3_CAM0_IMG/S3C3_";
	//char szTestDir[200] = "H:/[DB]amol/S3C4_CAM0_IMG/S3C4_CAM0_IMG/S3C4_";
	//char szTestDir[200] = "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_1/2015-04-13-14h-20m-45s_straight_";
	//////////////////////////////////////////////////////////////////////////
	char szTestDir[200] = "./[DB]AutoCalibration/Pitch_0_Yaw_0/Pitch_0_Yaw_0_";
	strcpy(szTestDir, szPrescanDB_dir);
	//////////////////////////////////////////////////////////////////////////
	if (DB_NUM == AMOL)
	{
		strcpy(szTestDir, "H:/[DB]amol/S3C1_CAM0_IMG/S3C1_CAM0_IMG/S3C1_");
		//strcpy(szTestDir, "H:/[DB]amol/S3C3_CAM0_IMG/S3C3_CAM0_IMG/S3C3_");
	}
	if (DB_NUM == CVLAB)
	{
		if (DB_ROADINFO == URBAN){
			strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Purity/Urban/Straight_1/2015-04-13-09h-07m-32s_F_normal_");  // PUS 평가 완료

			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_1/2015-04-13-14h-20m-45s_straight_");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_2/2015-04-13-14h-20m-45s_straight_2_");  // for demo
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_3/2015-04-13-14h-20m-45s_straight_3_");		// CUS 평가 완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_4/2015-04-13-14h-20m-45s_straight_4_");


			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/BackLight/Urban/Straight_1/2015-03-02-09h-40-00s_");//3월 //평가불가
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/BackLight/Urban/Straight_2/2015-04-23-09h-17m-10s_F_event_");//4월		// BUS 평가 완료

			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Rainy/Urban/Straight_1/2015-04-13-17h-37m-00s_");		// RUS 평가 완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Rainy/Urban/Straight_2/2015-04-13-18h-43m-00s_");		// RUS 평가 완료

			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Night/Urban/Straight_1/2015-04-17-20h-26m-47s_night_straight_1_");	// NUS1 평가 완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Night/Urban/Straight_2/2015-04-17-20h-29m-51s_night_straight_2_");		// NUS2 평가 완료
		}


		if (DB_ROADINFO == EXPRESSWAY){
			//////////////////////////////////////////////////////////////////////////
			//Expressway
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Purity/Expressway/Straight_1/2015-03-07-16h-48m-15s_F_event_");		// PES 평가 완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Purity/Expressway/Departure_1/2015-03-11-15h-11m-21s_F_event_");
			strcpy(szTestDir, "./[DB]CVLAB_Lane/Cloudy/Expressway/Straight_1/2015-04-13-14h-50m-44s_F_normal_");	// CES 평가 완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Cloudy/Expressway/Departure_1/2015-04-13-14h-37m-02s_F_normal_");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Backlight/Expressway/Straight_1/2015-04-16-17h-41m-50s_F_event_");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Backlight/Expressway/Straight_3/2015-04-16-17h-41m-50s_F_event_");		// BES 평가 완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Backlight/Expressway/Departure_1/2015-03-11-16h-27m-22s_F_event_");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Rainy/Expressway/Straight_1/2015-04-13-18h-33m-21s_F_event_");		// RES 평가완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Rainy/Expressway/Departure_1/2015-04-13-18h-33m-21s_F_event_");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Night/Expressway/Straight_1/2015-04-11-23h-50m-32s_F_event_");		//NES 평가완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Night/Expressway/Departure_1/2015-04-11-23h-50m-32s_F_event_");

			//Demo
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Rainy/Expressway/Demo_1/Demo_");
		}

	}
	char szAnnotationSaveFile[200];
	char szSaveText[20] = "0_SaveText.txt";
	strcpy(szAnnotationSaveFile, szTestDir);
	strcat(szAnnotationSaveFile, szSaveText);
	CMultiROILaneDetection obj;		//constructor
	obj.nLeftCnt = 0;
	obj.nRightCnt = 0;
	if (DB_NUM == CVLAB)
	{
		obj.m_sCameraInfo.sizeFocalLength.width = (float)656;
		obj.m_sCameraInfo.sizeFocalLength.height = (float)656;
		obj.m_sCameraInfo.ptOpticalCenter.x = (float)320;
		obj.m_sCameraInfo.ptOpticalCenter.y = (float)180;
		//obj.m_sCameraInfo.fHeight = (float)1100;
		obj.m_sCameraInfo.fHeight = (float)1050;
	}
	if (DB_NUM == AMOL)
	{
		obj.m_sCameraInfo.sizeFocalLength.width = (float)656;
		obj.m_sCameraInfo.sizeFocalLength.height = (float)656;
		obj.m_sCameraInfo.ptOpticalCenter.x = (float)320;
		obj.m_sCameraInfo.ptOpticalCenter.y = (float)240;
		obj.m_sCameraInfo.fHeight = (float)1250;
	}
	if (DB_NUM == PRESCAN){
		obj.m_sCameraInfo.sizeFocalLength.width = (float)656;
		obj.m_sCameraInfo.sizeFocalLength.height = (float)656;
		obj.m_sCameraInfo.ptOpticalCenter.x = (float)320;
		obj.m_sCameraInfo.ptOpticalCenter.y = (float)180;
		obj.m_sCameraInfo.fHeight = (float)1250;
	}
	obj.m_sCameraInfo.fPitch = 4.0 * PI / 180;
	obj.m_sCameraInfo.fYaw = 0.0 * PI / 180;

	obj.m_sConfig.fLineWidth = (float)2000;		//not used
	obj.m_sConfig.fLineHeight = (float)304.8;	//not used
	obj.m_sConfig.nRansacIteration = 40;	//not used
	obj.m_sConfig.nRansacThreshold = (float)0.2;	//not used
	obj.m_sConfig.fVanishPortion = (float) 0.0;
	obj.m_sConfig.fLowerQuantile = (float) 0.97;
	obj.m_sConfig.nLocalMaxIgnore = 0;		//Local maxima boundary reject pixels
	//obj.m_sConfig.nDetectionThreshold = 6;
	//obj.m_sConfig.nDetectionThreshold = 4;





	Size sizeOrigImg;
	Size sizeResizeImg;
	//fileInformation_t preScanDB_t2;
	strcpy(obj.m_sPreScanDB.szDataDir, szPrescanDB_dir);

	unsigned int nTotalFrame = 90;

	obj.m_nFrameNum = FIRSTFRAMENUM;
	//first frame read
	SetFrameName(obj.m_sPreScanDB.szDataName, obj.m_sPreScanDB.szDataDir, obj.m_nFrameNum);
	obj.m_imgOrigin = imread(string(obj.m_sPreScanDB.szDataName));
	if (obj.m_imgOrigin.empty())
	{
		printf("error, empty Images...");
		return -1;
	}

	///end


	sizeOrigImg.width = obj.m_imgOrigin.cols;
	sizeOrigImg.height = obj.m_imgOrigin.rows;
	sizeResizeImg = Size(sizeOrigImg.width / g_nResizeFacor, sizeOrigImg.height / g_nResizeFacor);
	obj.m_sCameraInfo.sizeCameraImage = sizeResizeImg;
	resize(obj.m_imgOrigin, obj.m_imgResizeOrigin, sizeResizeImg);
	cout << obj.m_imgResizeOrigin.size() << endl;
	//ROI IPM Transform make
	// CENTER_ROI
	obj.m_sRoiInfo[CENTER_ROI].nLeft = 216;
	obj.m_sRoiInfo[CENTER_ROI].nRight = 433;
	obj.m_sRoiInfo[CENTER_ROI].nTop = 185;
	obj.m_sRoiInfo[CENTER_ROI].nBottom = 220;
	obj.m_sRoiInfo[CENTER_ROI].sizeRoi.width = obj.m_sRoiInfo[CENTER_ROI].nRight - obj.m_sRoiInfo[CENTER_ROI].nLeft;
	obj.m_sRoiInfo[CENTER_ROI].sizeRoi.height = obj.m_sRoiInfo[CENTER_ROI].nBottom - obj.m_sRoiInfo[CENTER_ROI].nTop;
	obj.m_sRoiInfo[CENTER_ROI].ptRoi.x = obj.m_sRoiInfo[CENTER_ROI].nLeft;
	obj.m_sRoiInfo[CENTER_ROI].ptRoi.y = obj.m_sRoiInfo[CENTER_ROI].nTop;
	obj.m_sRoiInfo[CENTER_ROI].ptRoiEnd.x = obj.m_sRoiInfo[CENTER_ROI].ptRoi.x + obj.m_sRoiInfo[CENTER_ROI].sizeRoi.width;
	obj.m_sRoiInfo[CENTER_ROI].ptRoiEnd.y = obj.m_sRoiInfo[CENTER_ROI].ptRoi.y + obj.m_sRoiInfo[CENTER_ROI].sizeRoi.height;

	obj.m_sRoiInfo[CENTER_ROI].sizeIPM.width = 213;
	//	obj.m_sRoiInfo[CENTER_ROI].nRight - obj.m_sRoiInfo[CENTER_ROI].nLeft;
	obj.m_sRoiInfo[CENTER_ROI].sizeIPM.height = 80;
	//	obj.m_sRoiInfo[CENTER_ROI].nBottom - obj.m_sRoiInfo[CENTER_ROI].nTop+50;
	obj.m_sRoiInfo[CENTER_ROI].nDetectionThreshold = 4;
	obj.m_sRoiInfo[CENTER_ROI].nGetEndPoint = 0;
	obj.m_sRoiInfo[CENTER_ROI].nGroupThreshold = 10;
	obj.m_sRoiInfo[CENTER_ROI].fOverlapThreshold = 0.3;

	obj.m_sRoiInfo[CENTER_ROI].nRansacNumSamples = 2;	//Ransac
	obj.m_sRoiInfo[CENTER_ROI].nRansacNumIterations = 40;
	obj.m_sRoiInfo[CENTER_ROI].nRansacNumGoodFit = 10;
	obj.m_sRoiInfo[CENTER_ROI].fRansacThreshold = 0.2;
	obj.m_sRoiInfo[CENTER_ROI].nRansacScoreThreshold = 0;
	obj.m_sRoiInfo[CENTER_ROI].nRansacLineWindow = 15;



	//LEFT_ROI2
	obj.m_sRoiInfo[LEFT_ROI2].nLeft = 216 + 10 + 40;
	obj.m_sRoiInfo[LEFT_ROI2].nRight = 216 + 80 + 20;
	obj.m_sRoiInfo[LEFT_ROI2].nTop = 205 - 30;
	obj.m_sRoiInfo[LEFT_ROI2].nBottom = 205 + 15 - 30;
	obj.m_sRoiInfo[LEFT_ROI2].sizeRoi.width = obj.m_sRoiInfo[LEFT_ROI2].nRight - obj.m_sRoiInfo[LEFT_ROI2].nLeft;
	obj.m_sRoiInfo[LEFT_ROI2].sizeRoi.height = obj.m_sRoiInfo[LEFT_ROI2].nBottom - obj.m_sRoiInfo[LEFT_ROI2].nTop;
	obj.m_sRoiInfo[LEFT_ROI2].ptRoi.x = obj.m_sRoiInfo[LEFT_ROI2].nLeft;
	obj.m_sRoiInfo[LEFT_ROI2].ptRoi.y = obj.m_sRoiInfo[LEFT_ROI2].nTop;
	obj.m_sRoiInfo[LEFT_ROI2].ptRoiEnd.x = obj.m_sRoiInfo[LEFT_ROI2].ptRoi.x + obj.m_sRoiInfo[LEFT_ROI2].sizeRoi.width;
	obj.m_sRoiInfo[LEFT_ROI2].ptRoiEnd.y = obj.m_sRoiInfo[LEFT_ROI2].ptRoi.y + obj.m_sRoiInfo[LEFT_ROI2].sizeRoi.height;

	obj.m_sRoiInfo[LEFT_ROI2].sizeIPM.width =
		(obj.m_sRoiInfo[LEFT_ROI2].nRight - obj.m_sRoiInfo[LEFT_ROI2].nLeft)*1.5;
	obj.m_sRoiInfo[LEFT_ROI2].sizeIPM.height =
		(obj.m_sRoiInfo[LEFT_ROI2].nBottom - obj.m_sRoiInfo[LEFT_ROI2].nTop) * 2;
	obj.m_sRoiInfo[LEFT_ROI2].nDetectionThreshold = 1.5;
	obj.m_sRoiInfo[LEFT_ROI2].nGetEndPoint = 0;
	obj.m_sRoiInfo[LEFT_ROI2].nGroupThreshold = 10;
	obj.m_sRoiInfo[LEFT_ROI2].fOverlapThreshold = 0.3;

	obj.m_sRoiInfo[LEFT_ROI2].nRansacNumSamples = 2;	//Ransac
	obj.m_sRoiInfo[LEFT_ROI2].nRansacNumIterations = 40;
	obj.m_sRoiInfo[LEFT_ROI2].nRansacNumGoodFit = 10;
	obj.m_sRoiInfo[LEFT_ROI2].fRansacThreshold = 0.2;
	obj.m_sRoiInfo[LEFT_ROI2].nRansacScoreThreshold = 0;
	obj.m_sRoiInfo[LEFT_ROI2].nRansacLineWindow = 15;

	//LEFT_ROI3
	obj.m_sRoiInfo[LEFT_ROI3].nLeft = 216;
	obj.m_sRoiInfo[LEFT_ROI3].nRight = 216 + 80;
	obj.m_sRoiInfo[LEFT_ROI3].nTop = 205 - 5;
	obj.m_sRoiInfo[LEFT_ROI3].nBottom = 205 + 15 - 5;
	obj.m_sRoiInfo[LEFT_ROI3].sizeRoi.width = obj.m_sRoiInfo[LEFT_ROI3].nRight - obj.m_sRoiInfo[LEFT_ROI3].nLeft;
	obj.m_sRoiInfo[LEFT_ROI3].sizeRoi.height = obj.m_sRoiInfo[LEFT_ROI3].nBottom - obj.m_sRoiInfo[LEFT_ROI3].nTop;
	obj.m_sRoiInfo[LEFT_ROI3].ptRoi.x = obj.m_sRoiInfo[LEFT_ROI3].nLeft;
	obj.m_sRoiInfo[LEFT_ROI3].ptRoi.y = obj.m_sRoiInfo[LEFT_ROI3].nTop;
	obj.m_sRoiInfo[LEFT_ROI3].ptRoiEnd.x = obj.m_sRoiInfo[LEFT_ROI3].ptRoi.x + obj.m_sRoiInfo[LEFT_ROI3].sizeRoi.width;
	obj.m_sRoiInfo[LEFT_ROI3].ptRoiEnd.y = obj.m_sRoiInfo[LEFT_ROI3].ptRoi.y + obj.m_sRoiInfo[LEFT_ROI3].sizeRoi.height;

	obj.m_sRoiInfo[LEFT_ROI3].sizeIPM.width =
		(obj.m_sRoiInfo[LEFT_ROI3].nRight - obj.m_sRoiInfo[LEFT_ROI3].nLeft)*1.5;
	obj.m_sRoiInfo[LEFT_ROI3].sizeIPM.height =
		(obj.m_sRoiInfo[LEFT_ROI3].nBottom - obj.m_sRoiInfo[LEFT_ROI3].nTop) * 2;
	obj.m_sRoiInfo[LEFT_ROI3].nDetectionThreshold = 2;
	obj.m_sRoiInfo[LEFT_ROI3].nGetEndPoint = 0;
	obj.m_sRoiInfo[LEFT_ROI3].nGroupThreshold = 10;
	obj.m_sRoiInfo[LEFT_ROI3].fOverlapThreshold = 0.3;

	obj.m_sRoiInfo[LEFT_ROI3].nRansacNumSamples = 2;	//Ransac
	obj.m_sRoiInfo[LEFT_ROI3].nRansacNumIterations = 40;
	obj.m_sRoiInfo[LEFT_ROI3].nRansacNumGoodFit = 10;
	obj.m_sRoiInfo[LEFT_ROI3].fRansacThreshold = 0.2;
	obj.m_sRoiInfo[LEFT_ROI3].nRansacScoreThreshold = 0;
	obj.m_sRoiInfo[LEFT_ROI3].nRansacLineWindow = 15;

	//RIGHT_ROI2
	obj.m_sRoiInfo[RIGHT_ROI2].nLeft = 433 - 80 - 20;
	obj.m_sRoiInfo[RIGHT_ROI2].nRight = 433 - 10 - 40;
	obj.m_sRoiInfo[RIGHT_ROI2].nTop = 205 - 30;
	obj.m_sRoiInfo[RIGHT_ROI2].nBottom = 205 + 15 - 30;
	obj.m_sRoiInfo[RIGHT_ROI2].sizeRoi.width = obj.m_sRoiInfo[RIGHT_ROI2].nRight - obj.m_sRoiInfo[RIGHT_ROI2].nLeft;
	obj.m_sRoiInfo[RIGHT_ROI2].sizeRoi.height = obj.m_sRoiInfo[RIGHT_ROI2].nBottom - obj.m_sRoiInfo[RIGHT_ROI2].nTop;
	obj.m_sRoiInfo[RIGHT_ROI2].ptRoi.x = obj.m_sRoiInfo[RIGHT_ROI2].nLeft;
	obj.m_sRoiInfo[RIGHT_ROI2].ptRoi.y = obj.m_sRoiInfo[RIGHT_ROI2].nTop;
	obj.m_sRoiInfo[RIGHT_ROI2].ptRoiEnd.x = obj.m_sRoiInfo[RIGHT_ROI2].ptRoi.x + obj.m_sRoiInfo[RIGHT_ROI2].sizeRoi.width;
	obj.m_sRoiInfo[RIGHT_ROI2].ptRoiEnd.y = obj.m_sRoiInfo[RIGHT_ROI2].ptRoi.y + obj.m_sRoiInfo[RIGHT_ROI2].sizeRoi.height;

	obj.m_sRoiInfo[RIGHT_ROI2].sizeIPM.width =
		(obj.m_sRoiInfo[RIGHT_ROI2].nRight - obj.m_sRoiInfo[RIGHT_ROI2].nLeft)*1.5;
	obj.m_sRoiInfo[RIGHT_ROI2].sizeIPM.height =
		(obj.m_sRoiInfo[RIGHT_ROI2].nBottom - obj.m_sRoiInfo[RIGHT_ROI2].nTop) * 2;
	obj.m_sRoiInfo[RIGHT_ROI2].nDetectionThreshold = 1.5;
	obj.m_sRoiInfo[RIGHT_ROI2].nGetEndPoint = 0;
	obj.m_sRoiInfo[RIGHT_ROI2].nGroupThreshold = 10;
	obj.m_sRoiInfo[RIGHT_ROI2].fOverlapThreshold = 0.3;

	obj.m_sRoiInfo[RIGHT_ROI2].nRansacNumSamples = 5;	//Ransac 정확도 parameter
	obj.m_sRoiInfo[RIGHT_ROI2].nRansacNumIterations = 40;
	obj.m_sRoiInfo[RIGHT_ROI2].nRansacNumGoodFit = 10;
	obj.m_sRoiInfo[RIGHT_ROI2].fRansacThreshold = 0.2;
	obj.m_sRoiInfo[RIGHT_ROI2].nRansacScoreThreshold = 0;
	obj.m_sRoiInfo[RIGHT_ROI2].nRansacLineWindow = 15; 

	//RIGHT_ROI3
	obj.m_sRoiInfo[RIGHT_ROI3].nLeft = 433 - 80;
	obj.m_sRoiInfo[RIGHT_ROI3].nRight = 433;
	obj.m_sRoiInfo[RIGHT_ROI3].nTop = 205 - 5;
	obj.m_sRoiInfo[RIGHT_ROI3].nBottom = 205 + 15 - 5;
	obj.m_sRoiInfo[RIGHT_ROI3].sizeRoi.width = obj.m_sRoiInfo[RIGHT_ROI3].nRight - obj.m_sRoiInfo[RIGHT_ROI3].nLeft;
	obj.m_sRoiInfo[RIGHT_ROI3].sizeRoi.height = obj.m_sRoiInfo[RIGHT_ROI3].nBottom - obj.m_sRoiInfo[RIGHT_ROI3].nTop;
	obj.m_sRoiInfo[RIGHT_ROI3].ptRoi.x = obj.m_sRoiInfo[RIGHT_ROI3].nLeft;
	obj.m_sRoiInfo[RIGHT_ROI3].ptRoi.y = obj.m_sRoiInfo[RIGHT_ROI3].nTop;
	obj.m_sRoiInfo[RIGHT_ROI3].ptRoiEnd.x = obj.m_sRoiInfo[RIGHT_ROI3].ptRoi.x + obj.m_sRoiInfo[RIGHT_ROI3].sizeRoi.width;
	obj.m_sRoiInfo[RIGHT_ROI3].ptRoiEnd.y = obj.m_sRoiInfo[RIGHT_ROI3].ptRoi.y + obj.m_sRoiInfo[RIGHT_ROI3].sizeRoi.height;

	obj.m_sRoiInfo[RIGHT_ROI3].sizeIPM.width =
		(obj.m_sRoiInfo[RIGHT_ROI3].nRight - obj.m_sRoiInfo[RIGHT_ROI3].nLeft)*1.5;
	obj.m_sRoiInfo[RIGHT_ROI3].sizeIPM.height =
		(obj.m_sRoiInfo[RIGHT_ROI3].nBottom - obj.m_sRoiInfo[RIGHT_ROI3].nTop) * 2;
	obj.m_sRoiInfo[RIGHT_ROI3].nDetectionThreshold = 2;
	obj.m_sRoiInfo[RIGHT_ROI3].nGetEndPoint = 0;
	obj.m_sRoiInfo[RIGHT_ROI3].nGroupThreshold = 10;
	obj.m_sRoiInfo[RIGHT_ROI3].fOverlapThreshold = 0.3;

	obj.m_sRoiInfo[RIGHT_ROI3].nRansacNumSamples = 2;	//Ransac
	obj.m_sRoiInfo[RIGHT_ROI3].nRansacNumIterations = 40;
	obj.m_sRoiInfo[RIGHT_ROI3].nRansacNumGoodFit = 10;
	obj.m_sRoiInfo[RIGHT_ROI3].fRansacThreshold = 0.2;
	obj.m_sRoiInfo[RIGHT_ROI3].nRansacScoreThreshold = 0;
	obj.m_sRoiInfo[RIGHT_ROI3].nRansacLineWindow = 15;


	// AUTOCALIB
	if (DB_NUM == CVLAB)
	{
		if (INIT_CV == CV1)
		{
			obj.m_sRoiInfo[AUTOCALIB].nLeft = AUTOX;
			obj.m_sRoiInfo[AUTOCALIB].nRight = AUTOX + AUTOWIDTH;
			obj.m_sRoiInfo[AUTOCALIB].nTop = AUTOY;
			obj.m_sRoiInfo[AUTOCALIB].nBottom = AUTOY + AUTOHEIGHT;
		}
		if (INIT_CV == CV2){
			obj.m_sRoiInfo[AUTOCALIB].nLeft = CV2X;
			obj.m_sRoiInfo[AUTOCALIB].nRight = CV2X + CV2WIDTH;
			obj.m_sRoiInfo[AUTOCALIB].nTop = CV2Y;
			obj.m_sRoiInfo[AUTOCALIB].nBottom = CV2Y + CV2HEIGHT;
		}

	}
	//////////////////////////////////////////////////////////////////////////
	//Amol
	if (DB_NUM == AMOL)
	{
		obj.m_sRoiInfo[AUTOCALIB].nLeft = 216 - 20 - 20 - 20 - 10;
		obj.m_sRoiInfo[AUTOCALIB].nRight = 433 + 20 + 20 + 20 - 10;
		obj.m_sRoiInfo[AUTOCALIB].nTop = 185 + 30 + 30 + 30;
		obj.m_sRoiInfo[AUTOCALIB].nBottom = 220 + 30 + 60 + 60;
	}
	if (DB_NUM == PRESCAN)
	{
		obj.m_sRoiInfo[AUTOCALIB].nLeft = AUTOX;
		obj.m_sRoiInfo[AUTOCALIB].nRight = AUTOX + AUTOWIDTH;
		obj.m_sRoiInfo[AUTOCALIB].nTop = AUTOY - 10;
		obj.m_sRoiInfo[AUTOCALIB].nBottom = AUTOY + AUTOHEIGHT - 30 - 10 - 10;
	}

	obj.m_sRoiInfo[AUTOCALIB].sizeRoi.width = obj.m_sRoiInfo[AUTOCALIB].nRight - obj.m_sRoiInfo[AUTOCALIB].nLeft;
	obj.m_sRoiInfo[AUTOCALIB].sizeRoi.height = obj.m_sRoiInfo[AUTOCALIB].nBottom - obj.m_sRoiInfo[AUTOCALIB].nTop;
	obj.m_sRoiInfo[AUTOCALIB].ptRoi.x = obj.m_sRoiInfo[AUTOCALIB].nLeft;
	obj.m_sRoiInfo[AUTOCALIB].ptRoi.y = obj.m_sRoiInfo[AUTOCALIB].nTop;
	obj.m_sRoiInfo[AUTOCALIB].ptRoiEnd.x = obj.m_sRoiInfo[AUTOCALIB].ptRoi.x + obj.m_sRoiInfo[AUTOCALIB].sizeRoi.width;
	obj.m_sRoiInfo[AUTOCALIB].ptRoiEnd.y = obj.m_sRoiInfo[AUTOCALIB].ptRoi.y + obj.m_sRoiInfo[AUTOCALIB].sizeRoi.height;

	obj.m_sRoiInfo[AUTOCALIB].sizeIPM.width = //213;
		(obj.m_sRoiInfo[AUTOCALIB].nRight - obj.m_sRoiInfo[AUTOCALIB].nLeft);
	obj.m_sRoiInfo[AUTOCALIB].sizeIPM.height =// 80;
		(obj.m_sRoiInfo[AUTOCALIB].nBottom - obj.m_sRoiInfo[AUTOCALIB].nTop + 50);
	obj.m_sRoiInfo[AUTOCALIB].nDetectionThreshold = 3;
	obj.m_sRoiInfo[AUTOCALIB].nGetEndPoint = 0;
	obj.m_sRoiInfo[AUTOCALIB].nGroupThreshold = 10;
	obj.m_sRoiInfo[AUTOCALIB].fOverlapThreshold = 0.3;
	//건드릴 필요 없는 파라미터
	obj.m_sRoiInfo[AUTOCALIB].nRansacNumSamples = 2;	//Ransac
	obj.m_sRoiInfo[AUTOCALIB].nRansacNumIterations = 40;
	obj.m_sRoiInfo[AUTOCALIB].nRansacNumGoodFit = 10;
	obj.m_sRoiInfo[AUTOCALIB].fRansacThreshold = 0.2;
	obj.m_sRoiInfo[AUTOCALIB].nRansacScoreThreshold = 0;
	obj.m_sRoiInfo[AUTOCALIB].nRansacLineWindow = 15;

	//[LYW]Auto calibrarion start
	double dTickTestTotal = 0;
	int testIteration = 1;
	//	for(int i=0;i<testIteration;i++){
	double dStartTickTest = (double)getTickCount();
	obj.InitialResizeFunction(sizeResizeImg);
	//obj.SetRoiIpmCofig(CENTER_ROI);
	//obj.SetRoiIpmCofig(LEFT_ROI2);
	//obj.SetRoiIpmCofig(LEFT_ROI3);
	//obj.SetRoiIpmCofig(RIGHT_ROI2);
	//obj.SetRoiIpmCofig(RIGHT_ROI3);
	obj.SetRoiIpmCofig(AUTOCALIB);
	//////////////////////////////////////////////////////////////////////////
	//threshold(obj.m_imgIPM[1],obj.m_imgIPM[1],0.5);
	//cvtColor(obj.m_imgIPM,obj.m_filteredThreshold,CV_GRAY2RGBA)
	//threshold(obj.m_filteredThreshold,obj.m_filteredThreshold,125,255,THRESH_TOZERO);
	//GaussianBlur(,,Size(3,3),1,1))
	//Mat erode33(3,3,CV_8UC1,255);
	//erode(erode33,erode33,erode33);
	////dilate()
	//boxFilter(erode33,erode33,-1,Size(3,3));
	//Sobel(erode33,erode33,-1,3,3);
	//////////////////////////////////////////////////////////////////////////
	double dEndTickTest = (double)getTickCount();
	dTickTestTotal += (dEndTickTest - dStartTickTest);
	//	}
	cout << endl << endl << "##########		ROI setting & IPM Calibration time total  " << dTickTestTotal / testIteration / getTickFrequency()*1000.0 << " msec		###########" << endl << endl << endl;
	///end
	//////////////////////////////////////////////////////////////////////////


	//Auto Calibration data structure
	Vector <Mat> vecIpmFiltered;
	Vector < vector <Mat> > vecIpmArray;
	vector <Mat> originImg;
	vector <Mat> originImgClr;
	vector <Mat> vecFilteredThresImg;
	vector <vector <int> > vecPitchYaw;
	//End Auto Calibration data structure
	//////////////////////////////////////////////////////////////////////////
	//int nStartNum = 1;





	//main loof start
	//[LYW_0724]: 90프레임 누적
	for (int i = FIRSTFRAMENUM; obj.m_nFrameNum<nTotalFrame; obj.m_nFrameNum++, i++){

		//frame read
		if (obj.m_nFrameNum != FIRSTFRAMENUM){
			SetFrameName(obj.m_sPreScanDB.szDataName, obj.m_sPreScanDB.szDataDir, obj.m_nFrameNum);
			obj.m_imgOrigin = imread(string(obj.m_sPreScanDB.szDataName));

			if (obj.m_imgOrigin.empty())
			{
				printf("empty\n");
				break;
			}

			obj.InitialResizeFunction(sizeResizeImg);
			//resize(obj.m_imgOrigin,obj.m_imgResizeOrigin,sizeResizeImg);
		}///end		

		originImg.push_back(obj.m_imgResizeScaleGray.clone());
		originImgClr.push_back(obj.m_imgResizeOrigin.clone());
		//vecIpmFiltered.clear();
		//vecIpmFiltered.push_back(obj.m_ipmFiltered[AUTOCALIB]);
		//vecIpmArray.push_back(vecIpmFiltered[i]);
		//double dStartTick = (double)getTickCount();
		//CENTER_ROI
		//obj.StartLanedetection(CENTER_ROI);

		//LEFT_ROI2
		//obj.StartLanedetection(LEFT_ROI2);

		//LEFT_ROI3
		//obj.StartLanedetection(LEFT_ROI3);

		//RIGHT_ROI2
		//obj.StartLanedetection(RIGHT_ROI2);

		//RIGHT_ROI3
		//obj.StartLanedetection(RIGHT_ROI3);

		//AUTOCALIB
		//obj.StartLanedetection(AUTOCALIB);

		/*double dEndTick = (double)getTickCount();
		Point ptVanSt,ptVanEnd;
		ptVanSt.x = 0;
		ptVanSt.y = obj.m_sCameraInfo.ptVanishingPoint.y;
		ptVanEnd.x = obj.m_imgResizeOrigin.cols-1;
		ptVanEnd.y = obj.m_sCameraInfo.ptVanishingPoint.y;
		line(obj.m_imgResizeOrigin,ptVanSt,ptVanEnd,Scalar(0,255,0),2);*/
		//circle(obj.m_imgResizeOrigin,obj.m_sCameraInfo.ptVanishingPoint,2,Scalar(0,0,255),2,2);
		//ShowResults(obj,CENTER_ROI);
		//ShowResults(obj,LEFT_ROI2);
		//ShowResults(obj,LEFT_ROI3);
		//ShowResults(obj,RIGHT_ROI2);
		//ShowResults(obj,RIGHT_ROI3);
		//ShowResults(obj,AUTOCALIB);
		//////////////////////////////////////////////////////////////////////////
		//Auto Calibration Data Push
		//vecIpmFiltered.push_back(obj.m_ipmFiltered[CENTER_ROI].clone());
		//obj.PushBackResult(AUTOCALIB,vecIpmFiltered);



		//result & processing time show
		//cout<<"		processing time  "<<(dEndTick-dStartTick) / getTickFrequency()*1000.0<<" msec"<<endl;

		//imshow(g_strOriginalWindow,obj.m_imgResizeOrigin);
		//imshow("origin",obj.m_imgOrigin);
		//printf("frame num = %04d\n",obj.m_nFrameNum);
		//g_dTotlaTick+=(dEndTick-dStartTick);
		/*if('q'==waitKey(1)){
		cout<<"Program quit, Total processing time  "<<(g_dTotlaTick) / getTickFrequency()*1000.0<<" msec"<<endl;
		exit(0);
		}*/

		/*obj.m_lanes[CENTER_ROI].clear();
		obj.m_laneScore[CENTER_ROI].clear();
		obj.m_lanesResult[CENTER_ROI].clear();

		obj.m_lanes[LEFT_ROI3].clear();
		obj.m_laneScore[LEFT_ROI3].clear();
		obj.m_lanesResult[LEFT_ROI3].clear();*/
	}
	//show 90fr 확인용
	for (int i = 0; i<originImg.size(); i++){
		Mat imgOriginTemp = originImgClr[i].clone();
		rectangle(imgOriginTemp, Rect(obj.m_sRoiInfo[AUTOCALIB].nLeft, obj.m_sRoiInfo[AUTOCALIB].nTop, obj.m_sRoiInfo[AUTOCALIB].sizeRoi.width, obj.m_sRoiInfo[AUTOCALIB].sizeRoi.height),
			Scalar(0, 255, 0));
		imshow("saved", imgOriginTemp);
		cout << i << endl;
		waitKey(1);
	}
	//	dTickTestTotal= 0.0;
	int nNumberOfAutoCalibIter = 0;
	//	double dStartTickTest_AutoCalib = (double)getTickCount();
	FILE *fo;
	fstream fout;
	char txtPath[20] = "result_tilt6.txt";
	fout.open(txtPath, ios::out);
	float fMaxPitch = 0;
	float fMaxYaw = 0;
	double dMaxScore = -99;
	double dCompareScore = 0;
	vector <SLine> AutoCalibLane;
	//	for (float pitch = 1; pitch<7.5; pitch += 1){
	//		for (int yaw = -2; yaw <= 2; yaw++){
	//for (float pitch = 0; pitch < 3; pitch += 0.3){ //pitch 0이 없을 경우 최초 라인 1개 밖에 못찾음 에러 미해결
	for (float pitch = 0; pitch < 5; pitch += 0.3){
		for (int yaw = -1; yaw <= 1; yaw++){
			double dStartTickTest_AutoCalib = (double)getTickCount();
			obj.m_sCameraInfo.fPitch = (float)pitch * PI / 180;
			obj.m_sCameraInfo.fYaw = (float)yaw * PI / 180;
			obj.SetRoiIpmCofig(AUTOCALIB); //


			//			obj.InitialResizeFunction(sizeResizeImg);

			Mat imgSum;// = Mat::zeros(obj.m_ipmFiltered[AUTOCALIB].size(),CV_32FC1);
			for (int i = 0; i<originImg.size(); i++){
				obj.m_imgResizeScaleGray = originImg[i];

				obj.GetIPM(AUTOCALIB);
				obj.FilterLinesIPM(AUTOCALIB);
				if (imgSum.empty())
				{
					imgSum = Mat::zeros(obj.m_ipmFiltered[AUTOCALIB].size(), CV_32FC1);
					//printf("no\n");
				}
				//	imshow("ddd",originImg[i]);
				//	waitKey(0);
				imgSum += obj.m_ipmFiltered[AUTOCALIB].clone();
				//imgSum += obj.m_filteredThreshold[AUTOCALIB].clone();
				//if (i!=0)
				//imgSum /= (2);
			}
			double dEndTickTest_AutoCalib = (double)getTickCount();
			dTickTestTotal += (dEndTickTest_AutoCalib - dStartTickTest_AutoCalib);
			imgSum /= originImg.size();
			Mat rowMat;
			rowMat = Mat(imgSum).reshape(0, 1); // 1-row로 압축




			//get the quantile
			float fQval;
			fQval = quantile((float*)&rowMat.data[0], rowMat.cols, obj.m_sConfig.fLowerQuantile);
			Mat imgSumThres;
			threshold(imgSum, imgSumThres, fQval, NULL, THRESH_TOZERO);
			obj.m_ipmFiltered[AUTOCALIB] = imgSum;
			obj.m_filteredThreshold[AUTOCALIB] = imgSumThres;

			obj.GetLinesIPM(AUTOCALIB);
			obj.LineFitting(AUTOCALIB);

			nNumberOfAutoCalibIter++;
			//imgSum/=originImg.size();
			char namePitchYaw[20] = "name";
			char nameYaw[5];
			sprintf(namePitchYaw, "pitch:%f", pitch);
			sprintf(nameYaw, " Yaw:%d", yaw);
			strcat(namePitchYaw, nameYaw);
			//printf(namePitchYaw);
			//printf("\n");
			//fout<<namePitchYaw<<endl;
			fout << pitch << "\t" << yaw << "\t";
			dCompareScore = 0.0;
			for (int i = 0; i<obj.m_laneScore[AUTOCALIB].size(); i++){
				//printf("%d : %f ",i,obj.m_laneScore[AUTOCALIB][i]);
				fout << obj.m_laneScore[AUTOCALIB][i] << "\t";
				dCompareScore += obj.m_laneScore[AUTOCALIB][i];
			}
			if (dCompareScore>dMaxScore){ //검출 차선이 2개일 경우라는 조건을 추가해야함 (아직 미추가)
				dMaxScore = dCompareScore;
				fMaxPitch = pitch;
				fMaxYaw = yaw;
				obj.IPM2ImLines(AUTOCALIB);
				AutoCalibLane.clear();
				if (obj.m_lanesResult[AUTOCALIB].size() != 0){
					AutoCalibLane.push_back(obj.m_lanesResult[AUTOCALIB][0]);
					AutoCalibLane.push_back(obj.m_lanesResult[AUTOCALIB][1]);
				}

			}
			fout << endl;
			cout << endl;
			ShowImageNormalize(namePitchYaw, imgSum); // [LYW_0724] : 확인용
			//imshow(string(namePitchYaw), imgSum);
			waitKey(1);



		}
		fout << "AutoCalibration Result " << endl << "pitch : " << fMaxPitch << " yaw : " << fMaxYaw << endl;
		cout << "AutoCalibration Result " << endl << "pitch : " << fMaxPitch << " yaw : " << fMaxYaw << endl;
	}
	fout.close();
	//	double dEndTickTest_AutoCalib = (double)getTickCount();
	//	cout<<AutoCalibLane[0].ptStartLine<<endl;
	//	cout << AutoCalibLane[0].ptEndLine << endl;
	//	cout << AutoCalibLane[1].ptStartLine << endl;
	//	cout << AutoCalibLane[1].ptEndLine << endl;
	cout << "AutoCalib processing time  " << (dTickTestTotal / 1) / getTickFrequency()*1000.0 << " msec" << endl;
	//	cout<<"AutoCalib processing time  "<<(dEndTickTest_AutoCalib-dStartTickTest_AutoCalib) / getTickFrequency()*1000.0<<" msec"<<endl;
	//ShowResults(obj, AUTOCALIB);
	if (AutoCalibLane.size() != 2)
	{
		cout << " Auto calibration Fail, plz check intrinsic parameter " << AutoCalibLane.size() << endl;
		return 0;
	}


	//[LYW_0724] : Adaptive ROI setting start
	//////////////////////////////////////////////////////////////////////////
	//center line initial by auto calibration
	//////////////////////////////////////////////////////////////////////////

	obj.m_sCameraInfo.fPitch = (float)fMaxPitch * PI / 180;
	obj.m_sCameraInfo.fYaw = (float)fMaxYaw * PI / 180;
	obj.SetRoiIpmCofig(AUTOCALIB);

	obj.m_sImgCenter.ptStartLine.x = (AutoCalibLane[0].ptStartLine.x + AutoCalibLane[1].ptStartLine.x) / 2;
	obj.m_sImgCenter.ptStartLine.y = (AutoCalibLane[0].ptStartLine.y + AutoCalibLane[1].ptStartLine.y) / 2;
	obj.m_sImgCenter.ptEndLine.x = (AutoCalibLane[0].ptEndLine.x + AutoCalibLane[1].ptEndLine.x) / 2;
	obj.m_sImgCenter.ptEndLine.y = (AutoCalibLane[0].ptEndLine.y + AutoCalibLane[1].ptEndLine.y) / 2;

	obj.m_sWorldCenterInit.ptStartLane = obj.TransformPointImage2Ground(obj.m_sImgCenter.ptStartLine);
	obj.m_sWorldCenterInit.ptEndLane = obj.TransformPointImage2Ground(obj.m_sImgCenter.ptEndLine);
	obj.m_sWorldCenterInit.fXcenter = (obj.m_sWorldCenterInit.ptStartLane.x + obj.m_sWorldCenterInit.ptEndLane.x) / 2;
	obj.m_sWorldCenterInit.fXderiv = obj.m_sWorldCenterInit.ptStartLane.x - obj.m_sWorldCenterInit.ptEndLane.x;

	obj.m_sCameraInfo.fGroundTop = obj.m_sWorldCenterInit.ptStartLane.y;
	obj.m_sCameraInfo.fGroundBottom = obj.m_sWorldCenterInit.ptEndLane.y;
	//cout << obj.m_sCameraInfo.fGroundTop << endl;
	//cout << obj.m_sCameraInfo.fGroundBottom << endl;

	for (unsigned int i = 0; i < AutoCalibLane.size(); i++){
		line(obj.m_imgResizeOrigin,
			Point((int)AutoCalibLane[i].ptStartLine.x, (int)AutoCalibLane[i].ptStartLine.y),
			Point((int)AutoCalibLane[i].ptEndLine.x, (int)AutoCalibLane[i].ptEndLine.y),
			Scalar(0, 255, 0), 2);
	}// [LYW_0724] : 화면에 라인 2개 그려주기

	imshow("origin", obj.m_imgResizeOrigin);
	waitKey(0);
	obj.m_lanesResult[AUTOCALIB].clear();

	// AUTOCALIB
	//obj.m_sRoiInfo[AUTOCALIB].nLeft = 216 - 20;
	//obj.m_sRoiInfo[AUTOCALIB].nRight = 433 + 20;
	//obj.m_sRoiInfo[AUTOCALIB].nTop = 185 - 20;
	//obj.m_sRoiInfo[AUTOCALIB].nBottom = 220 + 0;
	//obj.m_sRoiInfo[AUTOCALIB].sizeRoi.width = obj.m_sRoiInfo[AUTOCALIB].nRight - obj.m_sRoiInfo[AUTOCALIB].nLeft;
	//obj.m_sRoiInfo[AUTOCALIB].sizeRoi.height = obj.m_sRoiInfo[AUTOCALIB].nBottom - obj.m_sRoiInfo[AUTOCALIB].nTop;
	//obj.m_sRoiInfo[AUTOCALIB].ptRoi.x = obj.m_sRoiInfo[AUTOCALIB].nLeft;
	//obj.m_sRoiInfo[AUTOCALIB].ptRoi.y = obj.m_sRoiInfo[AUTOCALIB].nTop;
	//obj.m_sRoiInfo[AUTOCALIB].ptRoiEnd.x = obj.m_sRoiInfo[AUTOCALIB].ptRoi.x + obj.m_sRoiInfo[AUTOCALIB].sizeRoi.width;
	//obj.m_sRoiInfo[AUTOCALIB].ptRoiEnd.y = obj.m_sRoiInfo[AUTOCALIB].ptRoi.y + obj.m_sRoiInfo[AUTOCALIB].sizeRoi.height;

	Rect_<int> rectLeftTop, rectLeftBottom;
	Rect_<int> rectRightTop, rectRightBottom;

	vector<Point> vecLeft;
	vector<Point> vecRight;
	int nDivNum = 5;
	//Auto calibration과정에서 검출한 결과 차선인 AutoCalibLane을 vector<Point>로 균등 분할 함수 (4등분)
	if (AutoCalibLane[0].ptStartLine.x < AutoCalibLane[1].ptStartLine.x){
		vecLeft = LineDivNum(AutoCalibLane[0].ptStartLine, AutoCalibLane[0].ptEndLine, nDivNum);
		vecRight = LineDivNum(AutoCalibLane[1].ptStartLine, AutoCalibLane[1].ptEndLine, nDivNum);
	}
	else{
		vecRight = LineDivNum(AutoCalibLane[0].ptStartLine, AutoCalibLane[0].ptEndLine, nDivNum);
		vecLeft = LineDivNum(AutoCalibLane[1].ptStartLine, AutoCalibLane[1].ptEndLine, nDivNum);
	}
	//[LYW_0724] : 4등분하는 점 5개 그려주기
	for (int i = 0; i < vecLeft.size(); i++){
		circle(obj.m_imgResizeOrigin, vecLeft[i], 2, Scalar(0, 0, 255), 2);
	}
	for (int i = 0; i < vecRight.size(); i++){
		circle(obj.m_imgResizeOrigin, vecRight[i], 2, Scalar(0, 0, 255), 2);
	}
	//auto calibration결과 차선 두개를 중심으로 초기 ROI adaptive하게 설정
	rectLeftTop.width = vecRight[1].x - vecLeft[1].x;		//left차선의 상단과 right차선의 상단의 x좌표 차이를 left top ROI의 width로 지정
	rectLeftTop.height = vecLeft[2].y - vecLeft[0].y;		//left top ROI의 height는 검출 차선 height의 1/2로 균등하게 지정
	rectLeftTop.x = vecLeft[1].x;							//left top point.x
	rectLeftTop.y = vecLeft[1].y;							//left top point.y

	rectLeftBottom.width = vecRight[3].x - vecLeft[3].x;
	rectLeftBottom.height = vecLeft[4].y - vecLeft[2].y;
	rectLeftBottom.x = vecLeft[3].x;
	rectLeftBottom.y = vecLeft[3].y;

	rectRightTop.width = vecRight[1].x - vecLeft[1].x;
	rectRightTop.height = vecRight[2].y - vecRight[0].y;
	rectRightTop.x = vecRight[1].x;
	rectRightTop.y = vecRight[1].y;

	rectRightBottom.width = vecRight[3].x - vecLeft[3].x;
	rectRightBottom.height = vecRight[4].y - vecRight[2].y;
	rectRightBottom.x = vecRight[3].x;
	rectRightBottom.y = vecRight[3].y;

	rectangle(obj.m_imgResizeOrigin, Rect(rectLeftTop.x - rectLeftTop.width / 2, rectLeftTop.y - rectLeftTop.height / 2, rectLeftTop.width, rectLeftTop.height), Scalar(255, 0, 0), 2);
	rectangle(obj.m_imgResizeOrigin, Rect(rectLeftBottom.x - rectLeftBottom.width / 2, rectLeftBottom.y - rectLeftBottom.height / 2, rectLeftBottom.width, rectLeftBottom.height), Scalar(255, 0, 0), 2);
	rectangle(obj.m_imgResizeOrigin, Rect(rectRightTop.x - rectRightTop.width / 2, rectRightTop.y - rectRightTop.height / 2, rectRightTop.width, rectRightTop.height), Scalar(255, 0, 0), 2);
	rectangle(obj.m_imgResizeOrigin, Rect(rectRightBottom.x - rectRightBottom.width / 2, rectRightBottom.y - rectRightBottom.height / 2, rectRightBottom.width, rectRightBottom.height), Scalar(255, 0, 0), 2);


	imshow("origin", obj.m_imgResizeOrigin);
	waitKey(0);
	float fWidthScale = IPM_WIDTH_SCALE; //[LYW_0724] : 계산해야할 IPM이미지의 사이즈를 변경해주는 역할. 임베디드에서 계산양을 줄이기 위해 조절할 수도 있음
	float fHeightScale = IPM_HEIGHT_SCALE;


	//////////////////////////end of Auto Calibration & adaptive ROI setting //////////////////////////////////////////////////////////////


	//[LYW_0724] : obj에 본격적으로 ROI정보 넣어주기
	//////////////////////////////////////////////////////////////////////////

	//LEFT_ROI2
	obj.m_sRoiInfo[LEFT_ROI2].nLeft = rectLeftTop.x - rectLeftTop.width / 2;
	obj.m_sRoiInfo[LEFT_ROI2].nRight = rectLeftTop.x + rectLeftTop.width / 2;
	obj.m_sRoiInfo[LEFT_ROI2].nTop = rectLeftTop.y - rectLeftTop.height / 2;
	obj.m_sRoiInfo[LEFT_ROI2].nBottom = rectLeftTop.y + rectLeftTop.height / 2;
	obj.m_sRoiInfo[LEFT_ROI2].sizeRoi.width = obj.m_sRoiInfo[LEFT_ROI2].nRight - obj.m_sRoiInfo[LEFT_ROI2].nLeft;
	obj.m_sRoiInfo[LEFT_ROI2].sizeRoi.height = obj.m_sRoiInfo[LEFT_ROI2].nBottom - obj.m_sRoiInfo[LEFT_ROI2].nTop;
	obj.m_sRoiInfo[LEFT_ROI2].ptRoi.x = obj.m_sRoiInfo[LEFT_ROI2].nLeft;
	obj.m_sRoiInfo[LEFT_ROI2].ptRoi.y = obj.m_sRoiInfo[LEFT_ROI2].nTop;
	obj.m_sRoiInfo[LEFT_ROI2].ptRoiEnd.x = obj.m_sRoiInfo[LEFT_ROI2].ptRoi.x + obj.m_sRoiInfo[LEFT_ROI2].sizeRoi.width;
	obj.m_sRoiInfo[LEFT_ROI2].ptRoiEnd.y = obj.m_sRoiInfo[LEFT_ROI2].ptRoi.y + obj.m_sRoiInfo[LEFT_ROI2].sizeRoi.height;
	obj.m_sRoiInfo[LEFT_ROI2].sizeIPM.width =
		(obj.m_sRoiInfo[LEFT_ROI2].nRight - obj.m_sRoiInfo[LEFT_ROI2].nLeft)*fWidthScale;
	obj.m_sRoiInfo[LEFT_ROI2].sizeIPM.height =
		(obj.m_sRoiInfo[LEFT_ROI2].nBottom - obj.m_sRoiInfo[LEFT_ROI2].nTop)*fHeightScale;

	//LEFT_ROI3
	obj.m_sRoiInfo[LEFT_ROI3].nLeft = rectLeftBottom.x - rectLeftBottom.width / 2;
	obj.m_sRoiInfo[LEFT_ROI3].nRight = rectLeftBottom.x + rectLeftBottom.width / 2;
	obj.m_sRoiInfo[LEFT_ROI3].nTop = rectLeftBottom.y - rectLeftBottom.height / 2;
	obj.m_sRoiInfo[LEFT_ROI3].nBottom = rectLeftBottom.y + rectLeftBottom.height / 2;
	obj.m_sRoiInfo[LEFT_ROI3].sizeRoi.width = obj.m_sRoiInfo[LEFT_ROI3].nRight - obj.m_sRoiInfo[LEFT_ROI3].nLeft;
	obj.m_sRoiInfo[LEFT_ROI3].sizeRoi.height = obj.m_sRoiInfo[LEFT_ROI3].nBottom - obj.m_sRoiInfo[LEFT_ROI3].nTop;
	obj.m_sRoiInfo[LEFT_ROI3].ptRoi.x = obj.m_sRoiInfo[LEFT_ROI3].nLeft;
	obj.m_sRoiInfo[LEFT_ROI3].ptRoi.y = obj.m_sRoiInfo[LEFT_ROI3].nTop;
	obj.m_sRoiInfo[LEFT_ROI3].ptRoiEnd.x = obj.m_sRoiInfo[LEFT_ROI3].ptRoi.x + obj.m_sRoiInfo[LEFT_ROI3].sizeRoi.width;
	obj.m_sRoiInfo[LEFT_ROI3].ptRoiEnd.y = obj.m_sRoiInfo[LEFT_ROI3].ptRoi.y + obj.m_sRoiInfo[LEFT_ROI3].sizeRoi.height;
	obj.m_sRoiInfo[LEFT_ROI3].sizeIPM.width =
		(obj.m_sRoiInfo[LEFT_ROI3].nRight - obj.m_sRoiInfo[LEFT_ROI3].nLeft)*fWidthScale;
	obj.m_sRoiInfo[LEFT_ROI3].sizeIPM.height =
		(obj.m_sRoiInfo[LEFT_ROI3].nBottom - obj.m_sRoiInfo[LEFT_ROI3].nTop)*fHeightScale;


	//[LYW_0815]: ROI추가 시도

	//LEFT_ROI0
	obj.m_sRoiInfo[LEFT_ROI0].nLeft = rectLeftTop.x - rectLeftTop.width / 2;
	obj.m_sRoiInfo[LEFT_ROI0].nRight = rectLeftTop.x + rectLeftTop.width / 2;
	obj.m_sRoiInfo[LEFT_ROI0].nTop = rectLeftTop.y - rectLeftTop.height / 2;
	obj.m_sRoiInfo[LEFT_ROI0].nBottom = rectLeftTop.y + rectLeftTop.height / 2;
	obj.m_sRoiInfo[LEFT_ROI0].sizeRoi.width = obj.m_sRoiInfo[LEFT_ROI0].nRight - obj.m_sRoiInfo[LEFT_ROI0].nLeft;
	obj.m_sRoiInfo[LEFT_ROI0].sizeRoi.height = obj.m_sRoiInfo[LEFT_ROI0].nBottom - obj.m_sRoiInfo[LEFT_ROI0].nTop;
	obj.m_sRoiInfo[LEFT_ROI0].ptRoi.x = obj.m_sRoiInfo[LEFT_ROI2].ptRoi.x - obj.m_sRoiInfo[LEFT_ROI0].sizeRoi.width;
	obj.m_sRoiInfo[LEFT_ROI0].ptRoi.y = obj.m_sRoiInfo[LEFT_ROI0].nTop;
	obj.m_sRoiInfo[LEFT_ROI0].ptRoiEnd.x = obj.m_sRoiInfo[LEFT_ROI0].ptRoi.x + obj.m_sRoiInfo[LEFT_ROI0].sizeRoi.width;
	obj.m_sRoiInfo[LEFT_ROI0].ptRoiEnd.y = obj.m_sRoiInfo[LEFT_ROI0].ptRoi.y + obj.m_sRoiInfo[LEFT_ROI0].sizeRoi.height;
	obj.m_sRoiInfo[LEFT_ROI0].sizeIPM.width =
		(obj.m_sRoiInfo[LEFT_ROI0].nRight - obj.m_sRoiInfo[LEFT_ROI0].nLeft)*fWidthScale;
	obj.m_sRoiInfo[LEFT_ROI0].sizeIPM.height =
		(obj.m_sRoiInfo[LEFT_ROI0].nBottom - obj.m_sRoiInfo[LEFT_ROI0].nTop)*fHeightScale;




	//RIGHT_ROI2
	obj.m_sRoiInfo[RIGHT_ROI2].nLeft = rectRightTop.x - rectRightTop.width / 2;
	obj.m_sRoiInfo[RIGHT_ROI2].nRight = rectRightTop.x + rectRightTop.width / 2;
	obj.m_sRoiInfo[RIGHT_ROI2].nTop = rectRightTop.y - rectRightTop.height / 2;
	obj.m_sRoiInfo[RIGHT_ROI2].nBottom = rectRightTop.y + rectRightTop.height / 2;
	obj.m_sRoiInfo[RIGHT_ROI2].sizeRoi.width = obj.m_sRoiInfo[RIGHT_ROI2].nRight - obj.m_sRoiInfo[RIGHT_ROI2].nLeft;
	obj.m_sRoiInfo[RIGHT_ROI2].sizeRoi.height = obj.m_sRoiInfo[RIGHT_ROI2].nBottom - obj.m_sRoiInfo[RIGHT_ROI2].nTop;
	obj.m_sRoiInfo[RIGHT_ROI2].ptRoi.x = obj.m_sRoiInfo[RIGHT_ROI2].nLeft;
	obj.m_sRoiInfo[RIGHT_ROI2].ptRoi.y = obj.m_sRoiInfo[RIGHT_ROI2].nTop;
	obj.m_sRoiInfo[RIGHT_ROI2].ptRoiEnd.x = obj.m_sRoiInfo[RIGHT_ROI2].ptRoi.x + obj.m_sRoiInfo[RIGHT_ROI2].sizeRoi.width;
	obj.m_sRoiInfo[RIGHT_ROI2].ptRoiEnd.y = obj.m_sRoiInfo[RIGHT_ROI2].ptRoi.y + obj.m_sRoiInfo[RIGHT_ROI2].sizeRoi.height;
	obj.m_sRoiInfo[RIGHT_ROI2].sizeIPM.width =
		(obj.m_sRoiInfo[RIGHT_ROI2].nRight - obj.m_sRoiInfo[RIGHT_ROI2].nLeft)*fWidthScale;
	obj.m_sRoiInfo[RIGHT_ROI2].sizeIPM.height =
		(obj.m_sRoiInfo[RIGHT_ROI2].nBottom - obj.m_sRoiInfo[RIGHT_ROI2].nTop)*fHeightScale;

	//RIGHT_ROI3
	obj.m_sRoiInfo[RIGHT_ROI3].nLeft = rectRightBottom.x - rectRightBottom.width / 2;
	obj.m_sRoiInfo[RIGHT_ROI3].nRight = rectRightBottom.x + rectRightBottom.width / 2;
	obj.m_sRoiInfo[RIGHT_ROI3].nTop = rectRightBottom.y - rectRightBottom.height / 2;
	obj.m_sRoiInfo[RIGHT_ROI3].nBottom = rectRightBottom.y + rectRightBottom.height / 2;
	obj.m_sRoiInfo[RIGHT_ROI3].sizeRoi.width = obj.m_sRoiInfo[RIGHT_ROI3].nRight - obj.m_sRoiInfo[RIGHT_ROI3].nLeft;
	obj.m_sRoiInfo[RIGHT_ROI3].sizeRoi.height = obj.m_sRoiInfo[RIGHT_ROI3].nBottom - obj.m_sRoiInfo[RIGHT_ROI3].nTop;
	obj.m_sRoiInfo[RIGHT_ROI3].ptRoi.x = obj.m_sRoiInfo[RIGHT_ROI3].nLeft;
	obj.m_sRoiInfo[RIGHT_ROI3].ptRoi.y = obj.m_sRoiInfo[RIGHT_ROI3].nTop;
	obj.m_sRoiInfo[RIGHT_ROI3].ptRoiEnd.x = obj.m_sRoiInfo[RIGHT_ROI3].ptRoi.x + obj.m_sRoiInfo[RIGHT_ROI3].sizeRoi.width;
	obj.m_sRoiInfo[RIGHT_ROI3].ptRoiEnd.y = obj.m_sRoiInfo[RIGHT_ROI3].ptRoi.y + obj.m_sRoiInfo[RIGHT_ROI3].sizeRoi.height;
	obj.m_sRoiInfo[RIGHT_ROI3].sizeIPM.width =
		(obj.m_sRoiInfo[RIGHT_ROI3].nRight - obj.m_sRoiInfo[RIGHT_ROI3].nLeft)*fWidthScale;
	obj.m_sRoiInfo[RIGHT_ROI3].sizeIPM.height =
		(obj.m_sRoiInfo[RIGHT_ROI3].nBottom - obj.m_sRoiInfo[RIGHT_ROI3].nTop)*fHeightScale;




	//[LYW_0724] : LUT만들기
	obj.SetRoiIpmCofig(LEFT_ROI2);
	obj.SetRoiIpmCofig(LEFT_ROI3);
	obj.SetRoiIpmCofig(LEFT_ROI0);//[LYW_0815] : ROI추가
	obj.SetRoiIpmCofig(RIGHT_ROI2);
	obj.SetRoiIpmCofig(RIGHT_ROI3);
	Point ptVanSt, ptVanEnd;
	ptVanSt.x = 0;
	ptVanSt.y = obj.m_sCameraInfo.ptVanishingPoint.y;
	ptVanEnd.x = obj.m_imgResizeOrigin.cols - 1;
	ptVanEnd.y = obj.m_sCameraInfo.ptVanishingPoint.y;

	//GROUND 승준이가 테스트하기 위해 만들었데. ROI하나 더 만들어봐서 테스트해보싶었데. 사실 별로 상관 없는 영역.
	obj.m_sRoiInfo[GROUND].nLeft = 0;
	obj.m_sRoiInfo[GROUND].nRight = 640;
	//obj.m_sRoiInfo[GROUND].nTop = rectRightTop.y - rectRightTop.height / 2-20;
	//obj.m_sRoiInfo[GROUND].nBottom = 360;
	obj.m_sRoiInfo[GROUND].nTop = ptVanSt.y;
	obj.m_sRoiInfo[GROUND].nBottom = 360;
	obj.m_sRoiInfo[GROUND].sizeRoi.width = obj.m_sRoiInfo[GROUND].nRight - obj.m_sRoiInfo[GROUND].nLeft;
	obj.m_sRoiInfo[GROUND].sizeRoi.height = obj.m_sRoiInfo[GROUND].nBottom - obj.m_sRoiInfo[GROUND].nTop;
	obj.m_sRoiInfo[GROUND].ptRoi.x = obj.m_sRoiInfo[GROUND].nLeft;
	obj.m_sRoiInfo[GROUND].ptRoi.y = obj.m_sRoiInfo[GROUND].nTop;
	obj.m_sRoiInfo[GROUND].ptRoiEnd.x = obj.m_sRoiInfo[GROUND].ptRoi.x + obj.m_sRoiInfo[GROUND].sizeRoi.width;
	obj.m_sRoiInfo[GROUND].ptRoiEnd.y = obj.m_sRoiInfo[GROUND].ptRoi.y + obj.m_sRoiInfo[GROUND].sizeRoi.height;
	obj.m_sRoiInfo[GROUND].sizeIPM.width =
		(obj.m_sRoiInfo[GROUND].nRight - obj.m_sRoiInfo[GROUND].nLeft) / 3;
	obj.m_sRoiInfo[GROUND].sizeIPM.height =
		(obj.m_sRoiInfo[GROUND].nBottom - obj.m_sRoiInfo[GROUND].nTop) / 3;


	obj.SetRoiIpmCofig(GROUND);
	//for (int i = 0; i<originImg.size(); i++){


	FILE *fp = fopen(szAnnotationSaveFile, "rt");
	SEvaluation structEvaluation;

	//////////////////////////////////////////////////////////////////////////
	//input DB change

	obj.m_bLeftDraw = false;
	obj.m_bRightDraw = false;


	//tracking 모듈 초기화
	obj.nCnt[2] = 0;//[LYW_0815]:roi추가
	obj.nCnt[0] = 0;
	obj.nCnt[1] = 0;

	obj.m_bDraw[2] = false; ////[LYW_0815]:roi추가
	obj.m_bDraw[0] = false;
	obj.m_bDraw[1] = false;
	//tracking 모듈 초기화

	int jMax = 5;
	if (DB_NUM == PRESCAN || DB_ROADINFO == EXPRESSWAY){
		jMax = 0;
	}


	for (int j = 0; j <= jMax; j++){
		char szEnvironment[20];
		if (j == 10){
			strcpy(szEnvironment, "Purity, Urban road");
			strcpy(szTestDir, "./[DB]FreeScaleDemo/Purity/Urban/Straight_1/2015-04-13-09h-07m-32s_F_normal_");  // PUS 평가 완료
		}
		else if (j == 1){
			strcpy(szEnvironment, "Cloudy, Urban road");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_1/2015-04-13-14h-20m-45s_straight_");
			strcpy(szTestDir, "./[DB]FreeScaleDemo/Cloudy/Urban/Straight_2/2015-04-13-14h-20m-45s_straight_2_");  // for demo
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_3/2015-04-13-14h-20m-45s_straight_3_");		// CUS 평가 완료
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Cloudy/Urban/Straight_4/2015-04-13-14h-20m-45s_straight_4_");
		}
		else if (j == 2){
			strcpy(szEnvironment, "BackLight, Urban road");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/BackLight/Urban/Straight_1/2015-03-02-09h-40-00s_");//3월 //평가불가
			strcpy(szTestDir, "./[DB]FreeScaleDemo/BackLight/Urban/Straight_2/2015-04-23-09h-17m-10s_F_event_");//4월		// BUS 평가 완료
		}
		else if (j == 3){
			strcpy(szEnvironment, "Night, Urban road");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Night/Urban/Straight_1/2015-04-17-20h-26m-47s_night_straight_1_");	// NUS1 평가 완료
			strcpy(szTestDir, "./[DB]FreeScaleDemo/Night/Urban/Straight_2/2015-04-17-20h-29m-51s_night_straight_2_");		// NUS2 평가 완료
		}
		else if (j == 4){
			strcpy(szEnvironment, "Rainy, Urban road");
			//strcpy(szTestDir, "H:/[DB]CVLAB_Lane/Rainy/Urban/Straight_1/2015-04-13-17h-37m-00s_");		// RUS 평가 완료
			strcpy(szTestDir, "./[DB]FreeScaleDemo/Rainy/Urban/Straight_2/2015-04-13-18h-43m-00s_");		// RUS 평가 완료
		}
		else if (j == 5){
			strcpy(szEnvironment, "Rainy, Urban road");
			strcpy(szTestDir, "./[DB]FreeScaleDemo/Rainy/Urban/Straight_3/2015-04-13-17h-37m-00s_");		// RUS 평가 완료
		}
		strcpy(obj.m_sPreScanDB.szDataDir, szTestDir);
		if (j != 2)
			for (int i = FIRSTFRAMENUM;; i++){
				/*obj.m_imgResizeScaleGray = originImg[i];
				obj.m_imgResizeOrigin = originImgClr[i];*/
				SetFrameName(obj.m_sPreScanDB.szDataName, obj.m_sPreScanDB.szDataDir, i);
				obj.m_imgOrigin = imread(string(obj.m_sPreScanDB.szDataName));
				if (obj.m_imgOrigin.empty())
					break;


				obj.InitialResizeFunction(sizeResizeImg);

				double dStartTick = (double)getTickCount();

				//detection start
				//[LYW]드디어 시작!!!
				obj.StartLanedetection(LEFT_ROI0); //[LYW_0815] : ROI추가
				obj.StartLanedetection(LEFT_ROI2);
				obj.StartLanedetection(LEFT_ROI3);
				obj.StartLanedetection(RIGHT_ROI2);
				obj.StartLanedetection(RIGHT_ROI3);

				//////////////////////////////////////////////////////////////////////////
				double dTrackingSt = (double)getTickCount();

				//Clear result Uv Coordinate

				////tkm before
				//tracking module
				//	obj.m_sTrakingLane[0].ptUvStartLine.x = EMPTY;
				//	obj.m_sTrakingLane[1].ptUvStartLine.x = EMPTY;

				//Left,Right 검출 결과 추적 맴버 변수로 변환


				////tracking module
				//////////////////////////////////////////////////////////////////////////
				//LEFT_ROI2와 LEFT_ROI3를 트렉킹 모듈 0에 , RIGHT_ROI2와 RIGHT_ROI3를 트렉킹 모듈 1에 적용하여 시작한다
				// 모듈을 추가하고 싶을 경우 아래와 같이 트렉킹 모듈을 추가시키면 된다.
				//////////////////////////////////////////////////////////////////////////
				
				obj.TrackingStageGround(LEFT_ROI0, 2); //[LYW_0815] : ROI추가(1)
				obj.TrackingStageGround(LEFT_ROI2, 0);
				obj.TrackingStageGround(LEFT_ROI3, 0);
				obj.TrackingStageGround(RIGHT_ROI2, 1);
				obj.TrackingStageGround(RIGHT_ROI3, 1);

				////tkm before		//Tracking continue 판별식
				/*obj.TrackingContinue();*/

				////tracking module
				obj.TrackingContinue(0);
				obj.TrackingContinue(1);
				obj.TrackingContinue(2); //[LYW_0815] : ROI추가(2)

				////tracking module
				if (obj.m_bTrackingFlag[obj.m_sTracking[LEFT_ROI0].nTargetTracker] == false){ 
					obj.m_sTracking[LEFT_ROI0].bTracking = false;
				}//[LYW_0815] : ROI추가(3)
				if (obj.m_bTrackingFlag[obj.m_sTracking[LEFT_ROI2].nTargetTracker] == false){
					obj.m_sTracking[LEFT_ROI2].bTracking = false;
				}
				if (obj.m_bTrackingFlag[obj.m_sTracking[LEFT_ROI3].nTargetTracker] == false){
					obj.m_sTracking[LEFT_ROI3].bTracking = false;
				}
				if (obj.m_bTrackingFlag[obj.m_sTracking[RIGHT_ROI2].nTargetTracker] == false){
					obj.m_sTracking[RIGHT_ROI2].bTracking = false;
				}
				if (obj.m_bTrackingFlag[obj.m_sTracking[RIGHT_ROI3].nTargetTracker] == false){
					obj.m_sTracking[RIGHT_ROI3].bTracking = false;
				}


				////tracking module
				obj.KalmanTrackingStage(0);
				obj.KalmanTrackingStage(1);
				obj.KalmanTrackingStage(2); //[LYW_0815] : ROI추가(4)


				//tracking module
				//[LYW_0815] : 이건 예외처리한 것 같음. 좌우 차선의 간격이 좁아지면 제거
				if ((obj.m_bDraw[0] == true) && (obj.m_bDraw[1] == true)){
					float fRightGround = obj.m_sTrakingLane[1].fXcenter / 1000;
					float fLeftGround = obj.m_sTrakingLane[0].fXcenter / 1000;
					if ((fRightGround - fLeftGround) < MIN_WORLD_WIDTH)
					{
						obj.ClearDetectionResult(0);
						obj.ClearDetectionResult(1);
						obj.ClearDetectionResult(2); //[LYW_0815] : ROI추가(5)
					}

				}

				//###########end of Image processing & Tracking processing################
				//////////////////////////////////////////////////////////////////////////
				double dEndTick = (double)getTickCount();
				double dTrackingEnd = (double)getTickCount();
				double dCurrentProcessTime = (dEndTick - dStartTick) / getTickFrequency()*1000.0;

				//	cout << "processing time  " << dCurrentProcessTime << " msec" << endl;
				//cout << "	Tracking time " << (dTrackingEnd - dTrackingSt) / getTickFrequency()*1000.0 << " msec" << endl;
				d_totalProcessTime += (dEndTick - dStartTick) / getTickFrequency()*1000.0;
				nCntProcess++;
				//////////////////////////////////////////////////////////////////////////
				//###########Result Draw##################################################

				//vanishing line draw
				line(obj.m_imgResizeOrigin, ptVanSt, ptVanEnd, Scalar(0, 255, 0), 2);
				
				//each roi result draw
				ShowResults(obj, LEFT_ROI0);//[LYW_0815] : ROI추가(6)
				ShowResults(obj, LEFT_ROI2);
				ShowResults(obj, LEFT_ROI3);
				ShowResults(obj, RIGHT_ROI2);
				ShowResults(obj, RIGHT_ROI3);

				obj.ClearResultVector(LEFT_ROI0); //[LYW_0815] : ROI추가(7)
				obj.ClearResultVector(LEFT_ROI2);
				obj.ClearResultVector(LEFT_ROI3);
				obj.ClearResultVector(RIGHT_ROI2);
				obj.ClearResultVector(RIGHT_ROI3);
				stringstream ssLeft, ssRight;
				stringstream ssLeft2;//[LYW_0815] : ROI추가(8)

				float fLeftGround, fRightGround;
				float fLeftGround2;//[LYW_0815] : ROI추가(9)


				////tracking module
				if (obj.m_bTrackingFlag[0]){

					int ssTemp = obj.m_sTrakingLane[0].fXcenter / 1000 * 100;
					fLeftGround = float(ssTemp) / 100;
					ssLeft << fLeftGround;
				}
				if (obj.m_bTrackingFlag[1]){
					int ssTemp = obj.m_sTrakingLane[1].fXcenter / 1000 * 100;
					fRightGround = float(ssTemp) / 100;
					ssRight << fRightGround;
				}
				if (obj.m_bTrackingFlag[2]){  //[LYW_0815] : ROI추가(10)
					int ssTemp = obj.m_sTrakingLane[2].fXcenter / 1000 * 100;
					fLeftGround2 = float(ssTemp) / 100;
					ssLeft2 << fLeftGround2;
				}
				//Lane Draw & Lateral Distance Draw
				if ((obj.m_bDraw[0] == true) && (obj.m_bDraw[1] == true)){
				//if ((obj.m_bDraw[0] == true) && (obj.m_bDraw[1] == true) && (obj.m_bDraw[2]==true)){
					if (obj.m_bTrackingFlag[0]){
						line(obj.m_imgResizeOrigin, obj.m_sTrakingLane[0].ptUvStartLine,
							obj.m_sTrakingLane[0].ptUvEndLine, Scalar(0, 0, 255), 2);
						putText(obj.m_imgResizeOrigin, ssLeft.str(), obj.m_sTrakingLane[0].ptUvEndLine,
							FONT_HERSHEY_COMPLEX, 1, Scalar(50, 50, 200), 2, 8, false);
					}
					if (obj.m_bTrackingFlag[1]){
						line(obj.m_imgResizeOrigin, obj.m_sTrakingLane[1].ptUvStartLine,
							obj.m_sTrakingLane[1].ptUvEndLine, Scalar(0, 0, 255), 2);
						putText(obj.m_imgResizeOrigin, ssRight.str(), obj.m_sTrakingLane[1].ptUvEndLine,
							FONT_HERSHEY_COMPLEX, 1, Scalar(50, 50, 200), 2, 8, false);
					}
					if (obj.m_bTrackingFlag[2]){ //[LYW_0815] : ROI추가(11)
						line(obj.m_imgResizeOrigin, obj.m_sTrakingLane[2].ptUvStartLine,
							obj.m_sTrakingLane[2].ptUvEndLine, Scalar(0, 0, 255), 2);
						putText(obj.m_imgResizeOrigin, ssLeft2.str(), obj.m_sTrakingLane[2].ptUvEndLine,
							FONT_HERSHEY_COMPLEX, 1, Scalar(50, 50, 200), 2, 8, false);
					}
				}


				//line(obj.m_imgResizeOrigin, obj.m_sImgCenter.ptStartLine, obj.m_sImgCenter.ptEndLine, Scalar(255, 0, 255),2);
				stringstream ssCenter;
				int ssTemp = (obj.m_sWorldCenterInit.fXcenter) / 1000 / 2 * 100;
				ssCenter << float(ssTemp) / 100;
				/*putText(obj.m_imgResizeOrigin, ssCenter.str(), obj.m_sImgCenter.ptEndLine,
				FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 0, 255), 1, 8, false);*/
				////////////////////////////////////////////////////////////////////////////
				////test

				//Mat matMat = Mat(2, 1, CV_32FC1);
				//float* pfMat = (float*)matMat.data;
				//pfMat[0] = obj.m_sLeftTrakingLane.fXcenter + obj.m_sLeftTrakingLane.fXderiv / 2+1000;
				//pfMat[1] = 20000;
				//obj.TransformGround2Image(matMat, matMat);
				//Point ptTemp = Point(pfMat[0], pfMat[1]);
				////circle(obj.m_imgResizeOrigin, ptTemp, 2, Scalar(255, 0, 0), 2);

				//pfMat[0] = obj.m_sLeftTrakingLane.fXcenter + obj.m_sLeftTrakingLane.fXderiv / 2 + 2000;
				//pfMat[1] = 20000;
				//obj.TransformGround2Image(matMat, matMat);
				//ptTemp = Point(pfMat[0], pfMat[1]);
				////circle(obj.m_imgResizeOrigin, ptTemp, 2, Scalar(0, 0, 255), 2);

				////test end
				////////////////////////////////////////////////////////////////////////////

				////anotation show
				//SWorldLane GroundLeft;
				//SWorldLane GroundRight;
				//if (EVALUATION == true){
				//	LoadExtractedPoint(fp, GroundLeft.ptUvStartLine, GroundLeft.ptUvEndLine, GroundRight.ptUvStartLine, GroundRight.ptUvEndLine);

				//	if ((GroundLeft.ptUvStartLine.x != EMPTY) && (GroundLeft.ptUvEndLine.x != EMPTY)){
				//		line(obj.m_imgResizeOrigin, GroundLeft.ptUvStartLine, GroundLeft.ptUvEndLine, Scalar(255, 0, 0), 3);
				//	}
				//	else{
				//		//cout << "	Left GroundTruth		 lane Detect EMPTY" << endl;
				//	}
				//		
				//	if ((GroundRight.ptUvStartLine.x != EMPTY) && (GroundRight.ptUvEndLine.x != EMPTY)){
				//		line(obj.m_imgResizeOrigin, GroundRight.ptUvStartLine, GroundRight.ptUvEndLine, Scalar(255, 0, 0), 3);
				//	}
				//	else{
				//		//cout << "	Right GroundTruth		 lane Detect EMPTY" << endl;
				//	}
				//		
				//	if (obj.m_sLeftTrakingLane.ptUvStartLine.x == EMPTY){
				//		//cout << "	Left		 lane Detect EMPTY" << endl;
				//	}
				//	if (obj.m_sRightTrakingLane.ptUvStartLine.x == EMPTY){
				//		//cout << "	Right		 lane Detect EMPTY" << endl;
				//	}
				//	EvaluationFunc(obj, structEvaluation, GroundLeft, GroundRight, obj.m_sLeftTrakingLane, obj.m_sRightTrakingLane);

				//}

				//cout << endl;
				//

				////////////////////////////////////////////////////////////////////////////

				putText(obj.m_imgResizeOrigin, szEnvironment, Point(10, 50),
					FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, 8, false);


				stringstream ssTime;
				char szProcTime[20] = "FPS : ";
				char szMs[10] = "ms";
				ssTime << szProcTime;
				ssTime << (int)(1000 / dCurrentProcessTime);
				//ssTime << szMs;
				putText(obj.m_imgResizeOrigin, ssTime.str(), Point(10, 90),
					FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 50), 2, 8, false);

				////TrackingStage(obj, LEFT_ROI2);
				////imshow("original_gray", obj.m_imgResizeScaleGray);
				imshow("Result Img", obj.m_imgResizeOrigin);
				////obj.GetIPM(GROUND);
				////imshow("IPM_GROUND", obj.m_imgIPM[GROUND]);
				//ShowResults(obj,AUTOCALIB);
				////waitKey(20);
				char cTemp = waitKey(1);
				if (cTemp == 'q'){
					break;
					i -= 2;
				}
			}// end of 하나의 frame에서 detection & tracking 다
		//obj.ClearDetectionResult();

		////tracking module
		obj.ClearDetectionResult(0);
		obj.ClearDetectionResult(1);
		obj.ClearDetectionResult(2);//[LYW_0815] : ROI추가(12)


	}
	cout << "Average processing time  : " << d_totalProcessTime / nCntProcess << endl;
	/*cout << "Total Frames : " << structEvaluation.nTotalFrame << endl;
	cout << "Left Evaluation" << endl;
	cout << "TP : " << structEvaluation.LeftTP << endl;
	cout << "FP : " << structEvaluation.LeftFP << endl;
	cout << "FN : " << structEvaluation.LeftFN << endl;
	cout << "TN : " << structEvaluation.LeftTN << endl;

	cout << "Right Evaluation" << endl;
	cout << "TP : " << structEvaluation.RightTP << endl;
	cout << "FP : " << structEvaluation.RightFP << endl;
	cout << "FN : " << structEvaluation.RightFN << endl;
	cout << "TN : " << structEvaluation.RightTN << endl;

	cout << "Total Evaluation" << endl;

	cout << "TP : " << structEvaluation.LeftTP + structEvaluation.RightTP << endl;
	cout << "FP : " << structEvaluation.LeftFP + structEvaluation.RightFP << endl;
	cout << "FN : " << structEvaluation.LeftFN + structEvaluation.RightFN << endl;
	cout << "TN : " << structEvaluation.LeftTN + structEvaluation.RightTN << endl;

	cout << "Total Ground Truth: " << structEvaluation.nLeftGroundTruth + structEvaluation.nRightGroundTruth << endl;*/
	waitKey(0);
	///end
	cout << "-------------------------------" << endl;
	cout << "process END" << endl;
	//for(int i=0; i<vecIpmFiltered.size();i++){
	//	imshow("pushed vecter mat",vecIpmFiltered[i]);
	//	//printf("%d\n",i);
	//	waitKey(1);
	//	//ShowImageNormalize("aa",vecIpmFiltered[i]);
	//	//cout<<vecIpmFiltered.size()<<endl;
	//	//printf("%d\n",i);
	//	//waitKey(0);
	//}
	//obj.GetCameraPose(AUTOCALIB, vecIpmFiltered);
	//ShowResults(obj,AUTOCALIB);
	//waitKey(0);
	return 0;
}


//my function
