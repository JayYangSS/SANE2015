//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#include <iostream>

using namespace cv; // OpenCV API is in the C++ "cv" namespace
using namespace std;
#include <stdio.h>
#include <io.h>
static const int MAX_CLASS = 4;	// 클래스 수
#define ATD at<int>

void readImg(vector<Mat> &, Mat &, int );
void HistVec(Mat& , vector<vector<int>>& );
static struct sSampleParam {
	int no_sample;				// 셈플 데이터의 수
	double mean_x, mean_y;		// x, y의 평균
	double stdev_x, stdev_y;	// x, y의 표준 편차
	CvScalar color_pt;			// 셈플 색
	CvScalar color_bg;			// 배경 색
} sample_param[MAX_CLASS] = {
	{ 500,  500, 200,  60,  30, CV_RGB(180, 0, 0), CV_RGB(255, 0, 0), },
	{ 1500, 200, 500, 100,  80, CV_RGB(0, 180, 0), CV_RGB(0, 255, 0), },
	{ 1000, 400, 700,  60, 100, CV_RGB(0, 0, 180), CV_RGB(0, 0, 255), },
};
void rotate(cv::Mat& src, double angle, cv::Mat& dst)
{

	int len = max(src.cols, src.rows);
	cv::Point2f pt(len/2., len/2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);

	cv::warpAffine(src, dst, r, cv::Size(len, len));
}


//void readImg(vector<Mat> &x, Mat &y, int number_of_images)
//{
//	y = Mat::zeros(1, number_of_images, CV_32SC1);
//	char num[10]={0};
//	char iFname[100]={0};
//	char iFpath[100]={0};
//	int cnt=0;
//	for(int ci=0; ci<MAX_CLASS; ci++){
//
//
//		sprintf(num, "%d", ci);
//		string inpath = "training\\train";
//		strcpy(iFname,inpath.c_str());
//		strncat(iFname,num,strlen(num));
//		strcpy(iFpath,iFname);
//		strcat(iFname,"\\*.*");
//		strcat(iFpath,"\\");
//		char s_buf[100];
//		char **filelist = NULL;
//		struct _finddata_t cfile;
//		long hFile;
//		sprintf_s(s_buf, iFname);
//		hFile= (long)_findfirst(s_buf, &cfile);	
//		_findnext(hFile, &cfile);
//
//		while(_findnext(hFile, &cfile)==0) //
//		{
//			cnt++;
//			sprintf_s(iFpath, "training\\train%s\\%s", num,cfile.name);
//			Mat img=imread(iFpath,CV_LOAD_IMAGE_GRAYSCALE);
//			x.push_back(img);
//			y.ATD(0, cnt-1) = (int)(ci);
//		}
//
//
//	}
//
//}


void HistVec(Mat& imghistor, Mat &matHist)
{
	Mat dst; 
	Mat imghist;
	imghistor.copyTo(imghist);
	imghist = 0;
	imghistor.copyTo(dst);
	vector<int> vecHist;
	//	cout << "dst.cols : "<<dst.cols <<", dst.rows : "<<dst.rows<<endl;
	for(int i=2 ; i<dst.cols; i=i+5 )
	{
		for(int j=0; j < dst.rows; j++)
		{
			if(dst.at<uchar>(j,i)!=0)
			{
				vecHist.push_back(j);
				//cout <<j<<" ";
				break;
			}
			else if(j == dst.rows-1){
				vecHist.push_back(j);
				//cout <<j<<" ";
			}
		}

	}

	//imshow("imghistor",imghistor);
	rotate(imghistor, -90, dst);
	//	cout <<endl<< "dst.cols : "<<dst.cols <<", dst.rows : "<<dst.rows<<endl;
	for(int i=2 ; i<dst.cols; i=i+5 )
	{
		for(int j=0; j < dst.rows; j++)
		{
			if(dst.at<uchar>(j,i)!=0)
			{
				vecHist.push_back(j);
				//cout <<j<<" ";
				break;
			}
			else if(j == dst.rows-1){
				vecHist.push_back(j);
				//cout <<j<<" ";
			}
		}

	}
	//	imshow("imghistor90",dst);
	rotate(dst, -90, dst);
	//	cout <<endl<< "dst.cols : "<<dst.cols <<", dst.rows : "<<dst.rows<<endl;
	for(int i=2 ; i<dst.cols; i=i+5 )
	{
		for(int j=0; j < dst.rows; j++)
		{
			if(dst.at<uchar>(j,i)!=0)
			{
				vecHist.push_back(j);
				//cout <<j<<" ";
				break;
			}
			else if(j == dst.rows-1){
				vecHist.push_back(j);
				//cout <<j<<" ";
			}
		}

	}
	//	imshow("imghistor180",dst);
	rotate(dst, -90, dst);
	//	cout <<endl<< "dst.cols : "<<dst.cols <<", dst.rows : "<<dst.rows<<endl;
	for(int i=2 ; i<dst.cols; i=i+5 )
	{
		for(int j=0; j < dst.rows; j++)
		{
			if(dst.at<uchar>(j,i)!=0)
			{
				vecHist.push_back(j);
				//cout <<j<<" ";
				break;
			}
			else if(j == dst.rows-1){
				vecHist.push_back(j);
				//cout <<j<<" ";

			}
		}

	}





	Mat sample(1, vecHist.size(), CV_32FC1);	
	for(int i=0; i<vecHist.size(); i++)
		sample.at<float>(i) = vecHist[i];

	matHist = sample;
	////imshow("imghistor270",dst);
	////cout <<endl;
	////cout <<"imghist.cols : " <<imghist.cols<<endl;
	//for(int i=0 ; i<vecHist.size(); i++ )
	//{
	//	for(int j=imghist.rows-1; j >=0; j--)
	//	{
	//		imghist.at<uchar>(j,i) = 255;
	//		if((imghist.rows-1-j)==vecHist[i])
	//		{
	//			//cout << (imghist.rows-1-j)<<" ";
	//			break;
	//		}
	//	}
	//}

	//VecHist.push_back(vecHist);
	////cout <<endl;
	////rotate(imghist, 180, imghist);
	//imshow("imghist",imghist);
	//cout <<endl<< "vecHist size : "<<vecHist.size()<<endl;;
}



void HistVecNormalize(Mat& imghistor, Mat &matHist)
{
	Mat dst; 
	Mat imghist;
	imghistor.copyTo(imghist);
	imghist = 0;
	imghistor.copyTo(dst);
	vector<int> vecHist;
	//	cout << "dst.cols : "<<dst.cols <<", dst.rows : "<<dst.rows<<endl;
	for(int i=2 ; i<dst.cols; i=i+5 )
	{
		for(int j=0; j < dst.rows; j++)
		{
			if(dst.at<uchar>(j,i)!=0)
			{
				vecHist.push_back(j);
				//cout <<j<<" ";
				break;
			}
			else if(j == dst.rows-1){
				vecHist.push_back(j);
				//cout <<j<<" ";
			}
		}

	}

	//imshow("imghistor",imghistor);
	rotate(imghistor, -90, dst);
	//	cout <<endl<< "dst.cols : "<<dst.cols <<", dst.rows : "<<dst.rows<<endl;
	for(int i=2 ; i<dst.cols; i=i+5 )
	{
		for(int j=0; j < dst.rows; j++)
		{
			if(dst.at<uchar>(j,i)!=0)
			{
				vecHist.push_back(j);
				//cout <<j<<" ";
				break;
			}
			else if(j == dst.rows-1){
				vecHist.push_back(j);
				//cout <<j<<" ";
			}
		}

	}
	//	imshow("imghistor90",dst);
	rotate(dst, -90, dst);
	//	cout <<endl<< "dst.cols : "<<dst.cols <<", dst.rows : "<<dst.rows<<endl;
	for(int i=2 ; i<dst.cols; i=i+5 )
	{
		for(int j=0; j < dst.rows; j++)
		{
			if(dst.at<uchar>(j,i)!=0)
			{
				vecHist.push_back(j);
				//cout <<j<<" ";
				break;
			}
			else if(j == dst.rows-1){
				vecHist.push_back(j);
				//cout <<j<<" ";
			}
		}

	}
	//	imshow("imghistor180",dst);
	rotate(dst, -90, dst);
	//	cout <<endl<< "dst.cols : "<<dst.cols <<", dst.rows : "<<dst.rows<<endl;
	for(int i=2 ; i<dst.cols; i=i+5 )
	{
		for(int j=0; j < dst.rows; j++)
		{
			if(dst.at<uchar>(j,i)!=0)
			{
				vecHist.push_back(j);
				//cout <<j<<" ";
				break;
			}
			else if(j == dst.rows-1){
				vecHist.push_back(j);
				//cout <<j<<" ";

			}
		}

	}





	Mat sample(1, vecHist.size(), CV_32FC1);	
	for(int i=0; i<vecHist.size(); i++)
		sample.at<float>(i) = vecHist[i];

	matHist = sample;
	////imshow("imghistor270",dst);
	////cout <<endl;
	////cout <<"imghist.cols : " <<imghist.cols<<endl;
	//for(int i=0 ; i<vecHist.size(); i++ )
	//{
	//	for(int j=imghist.rows-1; j >=0; j--)
	//	{
	//		imghist.at<uchar>(j,i) = 255;
	//		if((imghist.rows-1-j)==vecHist[i])
	//		{
	//			//cout << (imghist.rows-1-j)<<" ";
	//			break;
	//		}
	//	}
	//}

	//VecHist.push_back(vecHist);
	////cout <<endl;
	////rotate(imghist, 180, imghist);
	//imshow("imghist",imghist);
	//cout <<endl<< "vecHist size : "<<vecHist.size()<<endl;;
}
//int BayesianClassifier (Mat & imghistor)
//{
//	cv::RNG rng;
//	Mat matHist;
//
//	double timeC_s = (double)getTickCount();
//
//	HistVec(imghistor, matHist);
//	double timeC1_e = (double)getTickCount();
//	CvNormalBayesClassifier *bayes = new CvNormalBayesClassifier;
//	//CvNormalBayesClassifier bayes (trainData, trainY, 0, 0);
//	//CvNormalBayesClassifier bayes;
//	//bayes->train(trainData, trainY);
//
//	//bayes->save("trainDtBs.xml");
//
//	bayes->load("trainDtBs.xml");
//
//
//	//Mat sample(1, featureSZ, CV_32FC	
//
//	int response = bayes->predict(matHist, 0);
//
//	//cout << response <<" ";
//	//printf("%d ",response);
//
//	//trainX.clear();
//	//testX.clear();
//	//vecHist.clear();
//	// 키를 누르면 종료
//	//cvWaitKey (0);
//	return response;
//
//}

int BayesianClassifier(CvNormalBayesClassifier *bayes, Mat & imghistor)
{
	Mat matHist;
	HistVec(imghistor, matHist);
	int response = bayes->predict(matHist, 0);
	return response;
}
