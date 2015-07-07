// hello git world

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <fstream>
#include <math.h>
//#include "PedestrianDetector.h"

using namespace std;
using namespace cv;

#define BASELINE 0.25 // unit : m
#define FOCAL 1200 // unit : pixels   //240x10^-5:0.012=240:f

#define NUM_DISP 48
#define PI 3.141592
#define BOUND_DIST 20 //unit : m

int dispTohist(Mat *img, Mat *imghist,int nDisp, double *dDistance);

int calDist_stereo(Mat &imgLeftInput, Mat &imgRightInput, Rect_<int> &rRoi, Mat* disp8, Mat* imgHist, double* dDistance, int alg=0);
int calDist_mono(Mat &imgLeftInput, Rect_<int> &rRoi, double* dDistance, double dPitch=0);

int main()
{
	//////////////////////////////////// file load ////////////////////////////////////////////
	string strDBFilePath = "./공인DB/documentation/GroundTruth/"; // *.db file loading
	string strDBFileName = "GroundTruth2D_part1_stereo.db";
	string str3dDBFileName = "GroundTruth3D_part1_stereo.db";

	//string strLeftImagePath = "./공인DB/TestData_c0_part2/TestData/c0/"; // image file path
	//string strRightImagePath = "./공인DB/TestData_c1_part2/TestData/c1/"; // image file path
	string strLeftImagePath = "./공인DB/TestData_c0_part1/TestData/c0/"; // image file path
	string strRightImagePath = "./공인DB/TestData_c1_part1/TestData/c1/";

	FILE *fp = fopen((strDBFilePath+strDBFileName).c_str(),"rt");
	FILE *fp3D = fopen((strDBFilePath+str3dDBFileName).c_str(),"rt");
	FILE *fpDistErr = fopen("DaimlerPart1_dist.txt","wt");

	char temp[100];
	char temp3d[1000];
	for (int i=0; i<5; i++)              // first to fifth line will be ignore 
	{
		fscanf(fp, "%s", &temp);
		fscanf(fp3D,"%s",&temp);
	}

	//////////////////////////////////// variable /////////////////////////////////////////////
	Mat imgLeftInput, imgRightInput;
	Mat imgDisplay, imgDisparity, imgLeftGT;
	int ww, hh;
	int zero;

	enum { STEREO_BM=0, STEREO_SGBM=1 };
	bool alg = STEREO_BM;
	int SADWindowSize = 0;
	const int numberOfDisparities = NUM_DISP;
	bool no_display = false;
	
	double dtime,dtime_aver=0;
	int nObjFrameCnt = 0;

	int color_mode = alg == STEREO_BM ? 0 : -1;
	
	//////////////////////////////////////processing////////////////////////////////////////////
	////////////////////////////////////// off line ////////////////////////////////////////////
	
	
	//////////////////////////////////// on line /////////////////////////////////////////////

	int cntframe=0;
	while(1)
	{
	////////////////////////////////DB read///////////////////////////////////////////////////////////
		cntframe++;
		fscanf(fp, "%s", &temp);
		fscanf(fp3D,"%s",&temp3d);
		imgLeftInput = imread(strLeftImagePath+(string)temp,CV_LOAD_IMAGE_GRAYSCALE);
		imgRightInput = imread(strRightImagePath+(string)temp,CV_LOAD_IMAGE_GRAYSCALE);
		cout << (string)temp << endl;
		if(imgLeftInput.empty() || imgRightInput.empty())
		{	
			cout << "no images" << endl;
			break;
		}
		
		//int Count_Frame = 0;	
		//Count_Frame++;
		//printf("%d", Count_Frame);
		
		cvtColor(imgLeftInput,imgLeftGT,CV_GRAY2BGR);
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
		
		for (int i=0; i<numOfObj; i++)
		{
			char sharp[20];
			char ques[20];
			int objectClass;int objectClass3d;
			int objectID;int objectID3d;
			int uniqueID;int uniqueID3d;
			double confi;double confi3d;

			fscanf(fp, "%s", &sharp);
			fscanf(fp3D,"%s",&ques);
			if(ques[0]=='?') {
				//cout << "goodgood" << endl;
				fscanf(fp3D,"%d",&objectID3d);
				fscanf(fp3D,"%d",&uniqueID3d);
				fscanf(fp3D,"%lf",&confi3d);
				double dDepthGT=0;
				double dtemp=0;
				fscanf(fp3D,"%lf",&dtemp);
				fscanf(fp3D,"%lf",&dtemp);
				fscanf(fp3D,"%lf",&dDepthGT);
				fscanf(fp3D,"%lf",&dtemp);
				fscanf(fp3D,"%lf",&dtemp);
				fscanf(fp3D,"%lf",&dtemp);

				fscanf(fp, "%d", &objectClass);
				fscanf(fp, "%d", &objectID);
				fscanf(fp, "%d", &uniqueID);
				fscanf(fp, "%lf", &confi);
				Rect_<int> rect;Rect_<int> rect3d;
				int x2, y2;int x2_3d, y2_3d;
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
			fscanf(fp3D,"%d",&objectClass3d);
			
			fscanf(fp, "%d", &objectID);
			fscanf(fp3D,"%d",&objectID3d);

			fscanf(fp, "%d", &uniqueID);
			fscanf(fp3D,"%d",&uniqueID3d);

			fscanf(fp, "%lf", &confi);
			fscanf(fp3D,"%lf",&confi3d);

			Rect_<int> rect;Rect_<int> rect3d;
			int x2, y2;int x2_3d, y2_3d;
			fscanf(fp, "%d", &rect.x);
			fscanf(fp, "%d", &rect.y);
			fscanf(fp, "%d", &x2);
			fscanf(fp, "%d", &y2);
			fscanf(fp3D,"%d",&rect3d.x);fscanf(fp3D,"%d",&rect3d.y);fscanf(fp3D,"%d",&x2_3d);fscanf(fp3D,"%d",&y2_3d);

			rect.width = x2 - rect.x;
			rect.height = y2 - rect.y;

		//	if (objectClass == 0)
			fscanf(fp, "%d", &zero);
			fscanf(fp3D,"%d",&zero);	
		}
		fscanf(fp, "%s",&temp);
		fscanf(fp3D, "%s",&temp3d);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////// stereo matching process ///////////////////////////////////////
		imgLeftGT.copyTo(imgDisplay);
		
		int64 t = getTickCount();
		
		Mat disp8;
		Mat imgHist(256,256,CV_8U,Scalar(255));
		
		for(int i=0;i<vecRectGT.size();i++){
			rectangle(imgDisplay, vecRectGT[i],CV_RGB(0,255,0),2);
			
			//double dDistance_stereo=0, dDistance_mono=0;
			double dDistance_result=0;
			
			//calDist_mono(imgLeftInput,vecRectGT[i],&dDistance_result);
			//if(dDistance_result > BOUND_DIST) {
			//	calDist_stereo(imgLeftInput, imgRightInput, vecRectGT[i], &disp8, &imgHist, &dDistance_result);
			//	if(vecRectGT[i].x<NUM_DISP) dDistance_result=vecdRoiDistGT[i];
			//}
			calDist_stereo(imgLeftInput, imgRightInput, vecRectGT[i], &disp8, &imgHist, &dDistance_result);
			if(dDistance_result > 50) dDistance_result=50;
			vecdRoiDistance.push_back(dDistance_result);
			/*if(dDistance_result > 20){
				imshow("hist",imgHist);
				imshow("disp8",disp8);
			}*/
		}
				
		t = getTickCount() - t;
		dtime=t*1000/getTickFrequency();
		printf("image , Time elapsed: %fms\n", dtime);
		if(vecdRoiDistance.size()!=0) {dtime_aver+=dtime;nObjFrameCnt++;}

		for(int i=0;i<vecdRoiDistance.size();i++)
		{
			cout << "object #" << i << " distance : " << vecdRoiDistance[i] << "m" << endl;
			cout << "object #" << i << " GT dist  : " << vecdRoiDistGT[i] <<"m"<< endl;
			fprintf(fpDistErr,"%lf %lf %lf\n",vecdRoiDistance[i],vecdRoiDistGT[i],vecdRoiDistance[i]-vecdRoiDistGT[i]);
		}

		if(!no_display)
		{
			//char strRoiName[5]="Roi";
			//char strRoiSeq[4]="12";
			//for(int i=0; i<vecImgRoiDisp8.size(); i++){
			//	itoa(i,strRoiSeq,10);
			//	strcat(strRoiName, strRoiSeq);
			//	namedWindow(strRoiName,WINDOW_AUTOSIZE);
			//	imshow(strRoiName,vecImgRoiDisp8[i]);
			//	//cout << i << " size : " << vecImgRoiDisp8[i].rows << endl;
			//}
			//imshow("Left",imgLeftInput);
			//imshow("Right",imgRightInput);
			imshow("gt",imgDisplay);
			if(vecRectGT.size()!=0) {
				if(waitKey(0)=='t') ;
			}
			if(waitKey(1)==27) break;
			//if(cntframe==1) waitKey(0);
		}
		//cout << "frame : " << cntframe << endl;
		if(cntframe > 4000) {
			break;
		}
	}
	dtime_aver/=nObjFrameCnt;
	cout << "time aver : " << dtime_aver <<"ms"<< endl;
	fclose(fpDistErr);
	fclose(fp3D);
	fclose(fp);
	return 0;
}
int dispTohist(Mat *img,			// input image : disparity map
			   Mat *imghist,		// output image: histogram
			   int nDisp,			// input number of disparity
			   double *dDistance	// output
			   )
{
	/// Establish the number of bins
		int histSize = 256;

		/// Set the ranges ( for B,G,R) )
		float range[] = { 0, 256 }; 
		const float* histRange = { range };

		bool uniform = true; bool accumulate = false;

		Mat hist;
		float mean=0;
		float sum=0;

		/// Compute the histograms:
		calcHist( img, 1,0,Mat(),hist,1,&histSize,&histRange,uniform,accumulate);
		hist.at<float>(0) = 0;
		double maxVal=0, minVal=0;
		minMaxLoc(hist,&minVal,&maxVal,0,0);
		Mat histImg(256,256,CV_8U,Scalar(255));

		int hpt = static_cast<int>(0.9*256);//(5*256);//(10*256);
		int max = 0; 
		int disp_of_object = 0;
		float nhist_aver = 0;

		for(int h=0; h<208; h++)// 208 : 5m
		{
			if(h<4) hist.at<float>(h)=0;//|| h>numberOfDisparities
			float binVal = hist.at<float>(h);
			nhist_aver+=hist.at<float>(h)/208;
			if(h>2 && binVal>=max) {max=binVal;disp_of_object=h;}
			int intensity = static_cast<int>(binVal*hpt/maxVal);
			line(histImg,Point(h,255),Point(h,255-intensity),Scalar::all(0));
		}
		//cout << "max : " << max << ", aver : " << nhist_aver << endl;
	
		if((double)max/((double)nhist_aver-(double)max/153 ) > 4.)
		{
			//cout << "h : " << disp_of_object << endl;
			//cout << "disparity : " << (double)disp_of_object*(double)nDisp/255+5 << endl;
			*dDistance=(double)(BASELINE*FOCAL/((double)(disp_of_object)*(double)nDisp/255));
			//cout << "distance of object : " << *dDistance << "m" << endl;
			//fprintf(f_distance,"%lf\n",(double)(BASELINE*FOCAL/((double)disp_of_object*(double)numberOfDisparities/255+7)));
		}
		else
		{
			cout << "can't find a object " << endl;
			*dDistance=0;
			//fprintf(f_distance,"%d\n", 0);
		}
		*imghist=histImg;
	return 0;
}

int calDist_stereo(Mat &imgLeftInput, Mat &imgRightInput, Rect_<int> &rRoi, // input
				   Mat* disp8, Mat* imgHist,							 // output image
				   double* dDistance,									 // output
				   int alg												 // algorithm : BM=0 SGBM=1
				   )
{
	enum { STEREO_BM=0, STEREO_SGBM=1 };
	const int numberOfDisparities = NUM_DISP;
	int SADWindowSize = 0;
	Rect_<int> rectRoiTemp(rRoi);

	StereoBM bm;
	
	if(alg == STEREO_BM){
		if(rRoi.x < numberOfDisparities){
			rRoi.x=0;
			rRoi.width=rRoi.width+numberOfDisparities;
		}
		else{
			rRoi.x=rRoi.x-numberOfDisparities;
			rRoi.width=rRoi.width+numberOfDisparities;
		}
		//bm.state->roi1 = roi1;
		//bm.state->roi2 = roi2;
		bm.state->preFilterCap = 31;
		bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 11;
		bm.state->minDisparity = 1;
		bm.state->numberOfDisparities = numberOfDisparities;
		bm.state->textureThreshold = 10;
		bm.state->uniquenessRatio = 15;
		bm.state->speckleWindowSize = 25;//9;
		bm.state->speckleRange = 32;//4;
		bm.state->disp12MaxDiff = 1;
	}

	Mat disp, imgDisp8_temp;
	Mat imgRoiLeft, imgRoiRight;
	
	imgRoiLeft = imgLeftInput(rRoi).clone();
	imgRoiRight = imgRightInput(rRoi).clone();

	
	
	cout << rRoi.size() << " "<<imgRoiLeft.size() <<" " <<imgRoiRight.size()<<endl;
	if( alg == STEREO_BM )
		bm(imgRoiLeft,imgRoiRight,disp,CV_16S);//bm(imgRoiLeft, imgRoiRight, disp, CV_16S);//bm(imgLeftInput, imgRightInput, disp, CV_16S);//
	disp.convertTo(imgDisp8_temp, CV_8U, 255/(numberOfDisparities*16.));
	
	Rect rResetRoi(numberOfDisparities,0,rectRoiTemp.width,rectRoiTemp.height);
	
	*disp8=imgDisp8_temp(rResetRoi).clone();
	bm.state.release();
	
	//////////////////////////////////////////////////////////////////////////////////////////////

	dispTohist(disp8,imgHist,numberOfDisparities, dDistance);
	
	//temp : 20150702
	Scalar temp;
	Mat imgDiff=imgRoiLeft-imgRoiRight;
	temp = mean(imgDiff);
	//cout << "mean : " << temp << endl;
	threshold(imgDiff, imgDiff, temp(0), 255, CV_THRESH_BINARY);
	imshow("diff", imgDiff);


	imshow("roi left",imgRoiLeft);
	imshow("roi Right", imgRoiRight);
	imshow("roi disparity", imgDisp8_temp);
	waitKey(1);

	return 0;
}

int calDist_mono(Mat &imgLeftInput, Rect_<int> &rRoi, //input
				 double* dDistance,					//output
				 double dPitch						//input Pitch
				 )
{
	*dDistance = 1.17*tan((76.8+0.047125*(imgLeftInput.rows-(rRoi.y+rRoi.height)))*PI/180);

	return 0;
}






//////////////////stereo online process.///////////
			//StereoBM bm;
			//if(alg == STEREO_BM){
			//	if(vecRectGT[i].x < numberOfDisparities){
			//		vecRectGT[i].x=0;
			//		vecRectGT[i].width=vecRectGT[i].width+numberOfDisparities;
			//	}
			//	else{
			//		vecRectGT[i].x=vecRectGT[i].x-numberOfDisparities;
			//		vecRectGT[i].width=vecRectGT[i].width+numberOfDisparities;
			//	}
			//	//bm.state->roi1 = roi1;
			//	//bm.state->roi2 = roi2;
			//	bm.state->preFilterCap = 31;
			//	bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 11;
			//	bm.state->minDisparity = 1;
			//	bm.state->numberOfDisparities = numberOfDisparities;
			//	bm.state->textureThreshold = 10;
			//	bm.state->uniquenessRatio = 15;
			//	bm.state->speckleWindowSize = 25;//9;
			//	bm.state->speckleRange = 32;//4;
			//	bm.state->disp12MaxDiff = 1;
			//}

			//Mat disp;
			//Mat imgRoiLeft, imgRoiRight;
			////imgLeftInput(vecRectGT[i]).copyTo(imgRoiLeft);
			////imgRightInput(vecRectGT[i]).copyTo(imgRoiRight);
			//imgRoiLeft = imgLeftInput(vecRectGT[i]).clone();
			//imgRoiRight = imgRightInput(vecRectGT[i]).clone();
			//
			//imshow("roi left",imgRoiLeft);
			//imshow("roi Right", imgRoiRight);
			////waitKey(1);
			//
			//cout << vecRectGT[i].size() << " "<<imgRoiLeft.size() <<" " <<imgRoiRight.size()<<endl;
			//if( alg == STEREO_BM )
   //				bm(imgRoiLeft,imgRoiRight,disp,CV_16S);//bm(imgRoiLeft, imgRoiRight, disp, CV_16S);//bm(imgLeftInput, imgRightInput, disp, CV_16S);//
			//disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));

			////rectangle(disp8, vecRectGT[i], CV_RGB(255, 255, 255), 2);
			//Mat temp;
			//temp=disp8;//disp8(vecRectGT[i]);
			//vecImgRoiDisp8.push_back(temp);

			//bm.state.release();
			////////////////////////////////////////////////////////////////////////////////////////////////
			//Mat histImg(256,256,CV_8U,Scalar(255));

			//double dtemp;
			//dispTohist(&vecImgRoiDisp8[i],&histImg,numberOfDisparities, &dtemp);
			//vecdRoiDistance.push_back(dtemp);
			//cout << "object #" << i << " distance : " << dtemp << "m" << endl;

			//cout << "object #" << i << " GT dist  : " << vecdRoiDistGT[i] <<"m"<< endl;
			//fprintf(fpDistErr,"%lf %lf %lf\n",vecdRoiDistance[i],vecdRoiDistGT[i],vecdRoiDistance[i]-vecdRoiDistGT[i]);
