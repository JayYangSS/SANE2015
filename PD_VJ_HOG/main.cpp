#include <highgui.h>
#include <iostream>
#include <math.h>
#include "PedestrianDetector.h"
#include <fstream>
#include "new_KF.h"


using std::cout;
using std::endl;

using namespace std;
using namespace cv;

struct vecTracking{
	vector<Rect> vecBefore;
	vector<Rect> vecCandidate;
	vector<int> vecCount;
	vector<Point> vecCountPush;
};

void kalmanTrackingStart(SKalman&, Rect&);
void kalmanMultiTarget(Mat&, vector<Rect>&, vecTracking&, vector<SKalman>&, float, int, int, int, int&, bool&, bool&, Rect&, vector<Rect>&);

int main(){

	//vector<SKalman> MultiKF;
	int cntframe = 0;
	int numframe = 0;
	bool bOnOff = false;
	bool bOnOffcandidate = false;
	vecTracking High;

	char szFileName[32];			// image file name
	Mat imgSrc, imgRst, imgSrcRsz;

	/* Set Image ROI */
	CPedestrianDetector PDHaar(0, 0, 640, 360);		// constructor (x, y, width, height) <- ROI
	//CPedestrianDetector PDHaar(150, 320, 1000, 300);		// constructor (x, y, width, height) <- ROI
	//CPedestrianDetector PDHaarHOG(100, 140, 600, 200);

	/* Set MinMax Detector Size */
	PDHaar.SetMinSize(24, 48);			// Minimum size of detection
	PDHaar.SetMaxSize(90, 180);			// Maximum size of detection
	//PDHaarHOG.setMinSize(32, 64);
	//PDHaarHOG.setMaxSize(100, 200);

	/*  Make Colors */
	CvScalar yellow = CV_RGB(255, 255, 0), white = CV_RGB(255, 255, 255), red = CV_RGB(255, 0, 0);
	CvScalar green = CV_RGB(0, 255, 0), blue = CV_RGB(0, 0, 255), black = CV_RGB(0, 0, 0);
	
	/* Load Video */
	cv::VideoCapture vcap("../CVLAB_dataset/data/2015-04-29-14h-56m-42s_F_normal.avi");
	if (!vcap.isOpened()){
		cout << "Could not load the video file." << endl;
		return -1;
	}
	cout << "Successfully loaded the video file." << endl;


	string strFilePath = "../CVLAB_dataset/GT/";
	string strFileName = "2015-04-29-14h-56m-42s_F_normal_gt.txt";
	FILE *fp = fopen((strFilePath+strFileName).c_str(), "rt");

	/* Load Classifier */
	CascadeClassifier cascadeHOG, cascadeHaar;
	//if (!cascadeHaar.load("haarcascade_fullbody.xml")){				// Default for OpenCV
	if (!cascadeHaar.load("HaarCascade_body24x48_30stgs.xml")){
		cout << "xml file load error" << endl;
		return -1;
	}
	if (!cascadeHOG.load("hogCascade_pedestrians_newMinFAR04.xml")){
	//if(!cascadeHOG.load("hogCascade_pedestrians.xml")){		// Default for OpenCV
		cout << "xml file load error" << endl;
		return -1;
	}

	double tt = 0.0, t = 0.0;
	unsigned int nFrame = 0, i = 0;
	char temp[1];

	int totalFrames = 0;
	int totalGTs = 0;
	int totalDTs = 0;

	double totalTime = 0;
	int totalTP = 0;
	int totalFP = 0;
	int totalFN = 0;

	while ( true ){
		/* Capture Image Frame */
		vcap >> imgSrc;				// Bind video frame to image matrix
		if (imgSrc.empty()) break;

		int dtemp;
		fscanf(fp, "%d", &dtemp);
		printf("%d %d\n", dtemp,nFrame);
		int numOfObj = 0;
		fscanf(fp, "%d", &numOfObj);
		vector<Rect_<int> >vecRectGT;
		Rect rtemp = Rect(1280 / 4, 720 / 4, 1280 / 2, 720 / 2);

		for (int i = 0; i < numOfObj; i++)
		{
			Rect_<int> rect;
			int x2, y2;
			fscanf(fp, "%d", &rect.x);
			fscanf(fp, "%d", &rect.y);
			fscanf(fp, "%d", &rect.width);
			fscanf(fp, "%d", &rect.height);

// 			rect.x = rect.x / 2;
// 			rect.y = rect.y / 2;
// 			rect.width = x2 / 2;
// 			rect.height = y2 / 2;

			rect.x -= rtemp.x;
			rect.y -= rtemp.y;

			vecRectGT.push_back(rect);

		}

		///////////////////////////////////// draw GT rectangle /////////////////////////////////////////////
		//for (int i = 0; i < vecRectGT.size(); i++)
		//{
		//	rectangle(imgRst, vecRectGT[i], CV_RGB(0, 255, 0), 2);
		//}
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		/* Load Images */
		/*sprintf(szFileName, "version3/src%03d.jpg", i++);
		cout << szFileName;
		imgSrc = cv::imread(szFileName, CV_LOAD_IMAGE_UNCHANGED);*/

		//resize(imgSrc, imgSrcRsz, Size2i(640, 360));
		imgSrcRsz = imgSrc(rtemp).clone();


		vector<Rect> vecRectFound;
		vector<Rect> vecRectFoundUnfiltered;
		t = (double)cv::getTickCount();
		
		PDHaar.PedestrianDetectorHaarHOG(imgSrcRsz, cascadeHaar, cascadeHOG, vecRectFoundUnfiltered);
		//PDHaar.PedestrianDetectorHaar(imgRst, cascadeHaar, vecRectFound, 1.1f);

		t = ((double)getTickCount() - t) / getTickFrequency();
		totalTime += t;

		totalDTs += vecRectFoundUnfiltered.size();

		PDHaar.ClusteringAndRectangle(imgSrcRsz, vecRectFoundUnfiltered, yellow);
		

		//cout << nFrame << " : Detection time = " << t*1000. / cv::getTickFrequency() << "ms" << endl;

		//imgSrcRsz.copyTo(imgRst);
		//cv::rectangle(imgRst, PDHaar.GetROI(), white);		// Draw ROI

		//PDHaarHOG.pedestrianDetectorHaarHOG(imgSrc, cascadeHaar, cascadeHOG, vecRectFoundUnfiltered);
		//PDHaar.clusteringAndRectangle(imgRst, vecRectFoundUnfiltered, red);


		/////////////////////////////////// draw GT rectangle /////////////////////////////////////////////
		for (int i = 0; i < vecRectGT.size(); i++)
		{
			rectangle(imgSrcRsz, vecRectGT[i], CV_RGB(0, 255, 0), 2);
		}
		totalGTs += vecRectGT.size();
		for (int i = 0; i < vecRectFoundUnfiltered.size(); i++)
		{
			int chk = 0;
			for (int j = 0; j < vecRectGT.size(); j++)
			{
				int cap = (vecRectGT[j] & vecRectFoundUnfiltered[i]).area();
				int cup = (vecRectGT[j] | vecRectFoundUnfiltered[i]).area();

				double oRatio = (double)cap / (double)cup;
				if (oRatio >= 0.5)
				{
					chk = 1;
					vecRectGT.erase(vecRectGT.begin() + j);
					//j--;
					break;
				}
			}

			if (chk == 0)
				totalFP++;
			else
				totalTP++;
		}

		totalFN += vecRectGT.size();

		///////////////////////////////////////////////////////////////////////////////////////////////////////

		//========================Tracking========================//
		//vector<Rect> vecValidRec;
		//Point pCurrent;
		//float scaledist = 0.4;
		//int cntCandidate = 1;
		//int cntBefore = 4;
		//int frameCandidate = 3;

		//Rect ROIset = Rect_<int>(0, 0, imgRst.cols, imgRst.rows);

		//cntframe = PDHaar.nFrame;
		//kalmanMultiTarget(imgRst, vecRectFoundUnfiltered, High, MultiKF, scaledist, cntCandidate, cntBefore, frameCandidate, cntframe, bOnOff, bOnOffcandidate, ROIset, vecValidRec);      //kalmen tracking 

		//=======================================================//



		//resize(imgRst, imgRst, Size(imgRst.size().width / 2, imgRst.size().height / 2));
		imshow("People Detection", imgSrcRsz);

		totalFrames++;


		char c = cv::waitKey(30);
		if (c == 27) break;

		nFrame++;
		tt += t;
	}
	

	printf("fps = %.2lf\n", 1 / (totalTime / totalFrames));
	printf("totalFrames = %d\n", totalFrames);
	printf("totalGTs = %d\n", totalGTs);
	printf("totalDTs = %d\n", totalDTs);

	printf("totalTP = %d\n", totalTP);
	printf("totalFP = %d\n", totalFP);
	printf("totalFN = %d\n", totalFN);

	fclose(fp);

	return 0;
}

void kalmanMultiTarget(Mat& srcImage, vector<Rect>& vecRectTracking, vecTracking& Set, vector<SKalman>& MultiKF, float scaledist, int cntCandidate, int cntBefore, int frameCandiate, int& cntframe, bool& bOnOff, bool& bOnOffcandidate, Rect& ROIset, vector<Rect>& vecValidRec)
{
	Point pCurrent;

	//if (bOnOff == true)
	//{
	for (int i = 0; i<Set.vecBefore.size(); i++)
	{
		pCurrent.x = srcImage.cols;
		pCurrent.y = srcImage.rows;
		int num = -1;
		//cout <<"scaledist*Set.vecBefore.size() : "<<scaledist*Set.vecBefore[i].area()<<endl;
		for (int j = 0; j< vecRectTracking.size(); j++)
		{
			//cout << "MultiKF[i].ptPredict : "<<MultiKF[i].ptPredict.x <<"MultiKF[i].ptEstimate : "<< MultiKF[i].ptEstimate<<endl;

			//int dist = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x) + abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y);
			//if(dist <pCurrent.x+pCurrent.y  && dist <scaledist*Set.vecBefore[i].area()&&vecRectTracking[j].y+vecRectTracking[j].height/2-(Set.vecBefore[i].y+(Set.vecBefore[i].height)/2)<=0)
			//if(dist < pCurrent.x+pCurrent.y && dist <scaledist)
			float dist = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x)*abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x)
				+ abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y)*abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y);
			dist = sqrt(dist);
			float pCurrnetXY = pCurrent.x*pCurrent.x + pCurrent.y*pCurrent.y;
			pCurrnetXY = sqrt(pCurrnetXY);
			if (dist < pCurrent.x + pCurrent.y && dist <scaledist *Set.vecBefore[i].width)
			{
				pCurrent.x = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - MultiKF[i].ptEstimate.x);
				pCurrent.y = abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - MultiKF[i].ptEstimate.y);
				num = j;
			}
		}

		if (num == -1 && MultiKF.size()>0)
		{

			MultiKF[i].ptCenter.x = MultiKF[i].ptPredict.x;
			MultiKF[i].ptCenter.y = MultiKF[i].ptPredict.y;
			Set.vecBefore[i].x = MultiKF[i].ptPredict.x - (Set.vecBefore[i].width) / 2;
			Set.vecBefore[i].y = MultiKF[i].ptPredict.y - (Set.vecBefore[i].height) / 2;

		}
		else if (Set.vecBefore.size() != 0)
		{
			MultiKF[i].speedX = vecRectTracking[num].x + (vecRectTracking[num].width) / 2 - (Set.vecBefore[i].x + (Set.vecBefore[i].width) / 2);
			MultiKF[i].speedY = vecRectTracking[num].y + (vecRectTracking[num].height) / 2 - (Set.vecBefore[i].y + (Set.vecBefore[i].height) / 2);
			//cout << "vecRectTracking XY : " <<vecRectTracking[num].x+(vecRectTracking[num].width)/2<<" "<<vecRectTracking[num].y+(vecRectTracking[num].height)/2<<endl;
			//cout << "Set.vecBefore XY : " <<(Set.vecBefore[i].x+(Set.vecBefore[i].width)/2)<<" "<<Set.vecBefore[i].y+(Set.vecBefore[i].height)/2<<endl;
			//cout << "MultiKF[i].es : "<<MultiKF[i].ptEstimate.x<<" " << MultiKF[i].ptEstimate.y<<endl;
			Set.vecBefore[i] = vecRectTracking[num];
			vecRectTracking.erase(vecRectTracking.begin() + num);
			MultiKF[i].ptCenter.x = Set.vecBefore[i].x + (Set.vecBefore[i].width) / 2;
			MultiKF[i].ptCenter.y = Set.vecBefore[i].y + (Set.vecBefore[i].height) / 2;
			MultiKF[i].width = Set.vecBefore[i].width;
			MultiKF[i].height = Set.vecBefore[i].height;
			Set.vecCount[i] = cntframe; //vecCount 갱신
		}
		cout << "Set.vecBefore[i].y : " << Set.vecBefore[i].y << " ROIset : " << ROIset.y << endl;
		//if((Set.vecBefore[i].y<=(srcImage.rows/4-srcImage.rows/16+4)||(Set.vecBefore[i].x+Set.vecBefore[i].width>=srcImage.cols*1/3 + srcImage.cols*3/8-4)||(cntframe - Set.vecCount[i])>cntBefore) && Set.vecBefore.size() != 0&& MultiKF.size() != 0){
		if ((Set.vecBefore[i].y <= (ROIset.y) || (Set.vecBefore[i].x + Set.vecBefore[i].width <= ROIset.x) || (Set.vecBefore[i].x + Set.vecBefore[i].width >= ROIset.x + ROIset.width) || (cntframe - Set.vecCount[i])>cntBefore) && Set.vecBefore.size() != 0 && MultiKF.size() != 0){
			Set.vecBefore.erase(Set.vecBefore.begin() + i);
			MultiKF.erase(MultiKF.begin() + i);
			Set.vecCount.erase(Set.vecCount.begin() + i);
			i--;
		}
	}

	for (int i = 0; i<Set.vecBefore.size(); i++)
	{
		//MultiKF[i].ptCenter.x = Set.vecBefore[i].x+(Set.vecBefore[i].width)/2;
		//MultiKF[i].ptCenter.y = Set.vecBefore[i].y+(Set.vecBefore[i].height)/2;

		//kalmanfilter(srcImage,MultiKF[i].ptCenter,MultiKF[i].KF,MultiKF[i].ptEstimate,MultiKF[i].smeasurement,Set.vecBefore[i],MultiKF[i].rgb,MultiKF[i].width,MultiKF[i].height,MultiKF[i].speedX,MultiKF[i].speedY,MultiKF[i].ptPredict);
		bool bfinish = kalmanfilter(srcImage, MultiKF[i], Set.vecBefore[i], ROIset, vecValidRec);
		if (bfinish == false)
		{
			Set.vecBefore.erase(Set.vecBefore.begin() + i);
			MultiKF.erase(MultiKF.begin() + i);
			Set.vecCount.erase(Set.vecCount.begin() + i);
			i--;
			bfinish = true;
		}
		vecValidRec = Set.vecBefore;

	}

	//// initial
	//if (bOnOffcandidate == false && vecRectTracking.size()>0)
	//{
	//	bOnOffcandidate = true;
	//	for (int i = 0; i<vecRectTracking.size(); i++)
	//	{
	//		Set.vecCountPush.push_back(Point(0, cntframe));
	//		Set.vecCandidate.push_back(vecRectTracking[i]);
	//	}
	//}
	//else
	//{

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
			//int dist = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2)) + abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2));
			/*if(dist < CurrentX+CurrentY && dist <scaledist*Set.vecCandidate[i].area()&&vecRectTracking[j].y+(vecRectTracking[j].height)/2-(Set.vecCandidate[i].y+(Set.vecCandidate[i].height)/2)<=0)*/
			//if (dist < pCurrent.x + pCurrent.y && dist <scaledist * 2)
			if (dist < pCurrnetXY && dist <scaledist * 2 * Set.vecCandidate[i].width)
			{
				pCurrent.x = abs(vecRectTracking[j].x + (vecRectTracking[j].width) / 2 - (Set.vecCandidate[i].x + (Set.vecCandidate[i].width) / 2));
				pCurrent.y = abs(vecRectTracking[j].y + (vecRectTracking[j].height) / 2 - (Set.vecCandidate[i].y + (Set.vecCandidate[i].height) / 2));
				num = j;
				//Before = j;
			}
		}
		int speedtempX = 0;
		int speedtempY = 0;
		if (num > -1 && vecRectTracking.size()>0)
		{
			Set.vecCountPush[i].x++;			// 찾으면 +1
			Set.vecCountPush[i].y = cntframe;	// 찾았을때의 frame 갱신
			speedtempX = vecRectTracking[num].x - Set.vecCandidate[i].x;
			speedtempY = vecRectTracking[num].y - Set.vecCandidate[i].y;
			Set.vecCandidate[i] = vecRectTracking[num];
			vecRectTracking.erase(vecRectTracking.begin() + num);
		}

		if (Set.vecCountPush[i].x >= cntCandidate) // cntCandidate넘게 찾으면 새로운 target으로 인식
		{
			////
			Set.vecBefore.push_back(Set.vecCandidate[i]);
			SKalman temKalman;
			temKalman.speedX = speedtempX;
			temKalman.speedY = speedtempY;
			kalmanTrackingStart(temKalman, Set.vecCandidate[i]);
			MultiKF.push_back(temKalman);
			Set.vecCount.push_back(0);
			Set.vecCandidate.erase(Set.vecCandidate.begin() + i);
			Set.vecCountPush.erase(Set.vecCountPush.begin() + i);
			i--;
		}
	}
	for (int i = 0; i<vecRectTracking.size(); i++)
	{
		Set.vecCandidate.push_back(vecRectTracking[i]);
		Set.vecCountPush.push_back(Point(0, cntframe));
	}
	for (int i = 0; i<Set.vecCountPush.size(); i++)
	{
		if (cntframe - Set.vecCountPush[i].y >= frameCandiate) //frameCandidate이상 연속으로 못찾으면 제거
		{
			Set.vecCandidate.erase(Set.vecCandidate.begin() + i);
			Set.vecCountPush.erase(Set.vecCountPush.begin() + i);
			i--;
		}
	}
	//	}
	//}
	/////////////// initial tracking ///////////////
	//if (vecRectTracking.size()>0 && bOnOff == false)
	//{
	//	bOnOff = true;
	//	Set.vecBefore = vecRectTracking;
	//	for (int i = 0; i< Set.vecBefore.size(); i++)
	//	{
	//		SKalman temKalman;
	//		kalmanTrackingStart(temKalman, Set.vecBefore[i]);
	//		MultiKF.push_back(temKalman);
	//		Set.vecCount.push_back(0);
	//	}
	//}
}
void kalmanTrackingStart(SKalman& temKalman, Rect& recStart)
{
	temKalman.KF.statePost.at<float>(0) = recStart.x + (recStart.width) / 2;
	temKalman.KF.statePost.at<float>(1) = recStart.y + (recStart.height) / 2;
	temKalman.KF.statePost.at<float>(2) = temKalman.speedX;
	temKalman.KF.statePost.at<float>(3) = temKalman.speedY;
	temKalman.KF.statePre.at<float>(0) = recStart.x + (recStart.width) / 2;
	temKalman.KF.statePre.at<float>(1) = recStart.y + (recStart.height) / 2;
	temKalman.KF.statePre.at<float>(2) = temKalman.speedX;
	temKalman.KF.statePre.at<float>(3) = temKalman.speedY;
	kalmansetting(temKalman.KF, temKalman.smeasurement);
	temKalman.smeasurement.at<float>(0) = recStart.x + (recStart.width) / 2;
	temKalman.smeasurement.at<float>(1) = recStart.y + (recStart.height) / 2;
	temKalman.ptCenter.x = recStart.x + (recStart.width) / 2;
	temKalman.ptCenter.y = recStart.y + (recStart.height) / 2;
	temKalman.ptEstimate.x = recStart.x + (recStart.width) / 2;
	temKalman.ptEstimate.y = recStart.y + (recStart.height) / 2;
	temKalman.ptPredict.x = temKalman.ptEstimate.x + temKalman.speedX;
	temKalman.ptPredict.y = temKalman.ptEstimate.y + temKalman.speedY;
	//temKalman.rgb = Scalar(CV_RGB(rand() % 255, rand() % 255, rand() % 255));
	temKalman.rgb = Scalar(CV_RGB(255, 0, 255));
}