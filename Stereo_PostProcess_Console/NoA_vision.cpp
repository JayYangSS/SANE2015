#define _CRT_SECURE_NO_WARNINGS

#include<iostream>
#include<opencv2/opencv.hpp>
#include<windows.h>

using namespace std;
using namespace cv;

static bool readStringList(const string& filename, vector<string>& l);

// 3D 좌표 구하고 Z 좌표 중 1.5 밑, 50 위인거 0으로 

Mat compute3DAndRemove(Mat disp){
	float thresMax = 50;
	float thresMin = 3.5;
	//float u0 = 258;
	float v0 = 156;
	float au = 410;
	float av = au;
	float b = 0.24;
	float z0 = 7;
	for (int u = 0; u<disp.rows; u++){
		unsigned char *input = disp.ptr<unsigned char>(u);
		for (int v = 0; v<disp.cols; v++) {
			int d = input[v] / 16;
			//float x = (u-u0)*b/d - (b/2);
			//float y = au*b/d;
			float z = z0 - (((u - v0)*b) / (d));
			if (z<thresMin || z>thresMax){
				input[v] = 0;
			}
		}
	}
	return disp;
}

//occupancy grid 시도
Mat occupancygrid(Mat img){
	int maxDisp = 50;
	int mind = 130;
	int minz = 10;
	float au = 410;
	float b = 0.24;
	Mat occupancy(img.cols, maxDisp * 10, CV_8U, Scalar(0));	// 세로 640 가로 500 까만 화면

	//u는 행. max640	v는 max480
	for (int u = 0; u<img.cols; u++) {
		for (int v = 0; v<img.rows; v++) {
			int dd = img.at<unsigned char>(v, u);
			float y = au*b / dd;
			if (y * 10<20)
				occupancy.at<unsigned char>(u, y * 100) = 255;
			//{mind=y*100;}  	
		}
		//occupancy.at<unsigned char>(u,mind) = 255;	
	}
	return occupancy;
}

int main()
{
	const char* img1_filename = "kv30l_0.bmp";
	const char* img2_filename = "kv30r_0.bmp";
	const char* intrinsic_filename = "intrinsics.yml";
	const char* extrinsic_filename = "extrinsics.yml";

	//image list load	
	string Left_imglist_filename = "./imgLeft_list.xml";//"./calibration/스테레오카메라 Left BMP/imgLeft_list.xml";
	string Right_imglist_filename = "./imgRight_list.xml";;//"./calibration/스테레오카메라 Right BMP/imgRight_list.xml"
	string Left_imglist_path = "./스테레오카메라 Left BMP/";//"./calibration/스테레오카메라 Left BMP/";
	string Right_imglist_path = "./스테레오카메라 Right BMP/";//"./calibration/스테레오카메라 Right BMP/";
	vector<string> vLeftimglist, vRightimglist;
	int nimages;

	bool ok = readStringList(Left_imglist_filename, vLeftimglist);
	if (ok == false) { cout << "left file error" << endl; return -1; }
	ok = readStringList(Right_imglist_filename, vRightimglist);
	if (ok == false) { cout << "Right file error" << endl; return -1; }
	if (vLeftimglist.size() != vRightimglist.size()) { cout << "image file error" << endl; return -1; }

	else nimages = vLeftimglist.size();	// nimages는 이미지 개수

	enum { STEREO_BM = 0, STEREO_SGBM = 1 };
	int alg = STEREO_BM;
	int SADWindowSize = 9, numberOfDisparities = 80;
	bool no_display = false;
	float scale = 1.f;

	StereoBM bm;
	StereoSGBM sgbm;

	if (!img1_filename || !img2_filename)
	{
		printf("Command-line parameter error: both left and right images must be specified\n");
		return -1;
	}

	if ((intrinsic_filename != 0) ^ (extrinsic_filename != 0))
	{
		printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
		return -1;
	}

	int color_mode = alg == STEREO_BM ? 0 : -1;
	Mat img1 = imread(img1_filename, color_mode);
	Mat img2 = imread(img2_filename, color_mode);

	if (scale != 1.f)
	{
		Mat temp1, temp2;
		int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
		resize(img1, temp1, Size(), scale, scale, method);
		img1 = temp1;
		resize(img2, temp2, Size(), scale, scale, method);
		img2 = temp2;
	}

	Size img_size = img1.size();

	Rect roi1, roi2;
	Mat Q;

	Mat img1r, img2r;
	Mat map11, map12, map21, map22;

	if (intrinsic_filename)
	{
		// reading intrinsic parameters
		FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename);
			return -1;
		}

		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		//cout << M1 << endl << M2 << endl << D1 << endl << D2 << endl;

		M1 *= scale;
		M2 *= scale;

		fs.open(extrinsic_filename, CV_STORAGE_READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", extrinsic_filename);
			return -1;
		}

		Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;

		//cout << R << endl << T << endl;

		stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);


		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		//imshow("temp_left", img1r);
		//imshow("temp_right", img2r);

		img1 = img1r;
		img2 = img2r;
	}

	//numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 9;//100;
	bm.state->speckleRange = 4;//32;
	bm.state->disp12MaxDiff = 1;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3; // sgbm size : 5~9 odd 

	int cn = img1.channels();

	sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = bm.state->speckleWindowSize;
	sgbm.speckleRange = bm.state->speckleRange;
	sgbm.disp12MaxDiff = 1;
	///////////////////////////////////////////////////////////////////////////////////
	Mat disp, disp8;
	/////////////////////////////////////on-line////////////////////////////////////////

	for (int i = 250; i<nimages; i++){

		img1 = imread((Left_imglist_path + vLeftimglist[i]).c_str(), color_mode);	//c_str() 는 char*로의 타입 변환 함수
		img2 = imread((Right_imglist_path + vRightimglist[i]).c_str(), color_mode);


		// img1 = imread(vLeftimglist[i], color_mode);
		// img2 = imread(vRightimglist[i], color_mode);

		int64 t = getTickCount();

		//Mat imgLeftNew = imread(leftfile name, color_mode);
		//Mat imgRightNew = imread(right also, color_mode);

		remap(img1, img1r, map11, map12, INTER_LINEAR); // use carefully about (map11, 12, 21, 22)
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		img1 = img1r;
		img2 = img2r;

		if (alg == STEREO_BM)
			bm(img1, img2, disp);
		else if (alg == STEREO_SGBM)
			sgbm(img1, img2, disp);

		t = getTickCount() - t;
		printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*11.));

		if (!no_display)
		{
			namedWindow("left", 1);
			imshow("left", img1);
			namedWindow("right", 1);
			imshow("right", img2);
			//namedWindow("disparity", 0);
			//imshow("disparity", disp8);
			Mat imgThres, imgThresre, disp88, imgThres2;
			Mat imgRoad = compute3DAndRemove(disp8);

			threshold(imgRoad, imgThres, 80, 255, CV_THRESH_BINARY);
			threshold(imgRoad, imgThresre, 80, 255, CV_THRESH_BINARY_INV);
			bitwise_and(imgRoad, imgThresre, disp88);
			threshold(disp88, imgThres2, 50, 255, CV_THRESH_BINARY);

			//printf("press any key to continue...");
			//imshow("img", imgThres);

			vector<vector<Point>> contours;
			// 외곽선 벡터 , 외부 외곽선 검색, 내부 공백 무시,  각 외곽선의 모든 화소 탐색
			findContours(imgThres, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			int cmin = 1000;  // 최소 외곽선 길이
			int cmax = 10000; // 최대 외곽선 길이

			vector<vector<Point>>::const_iterator itc = contours.begin();

			while (itc != contours.end()) {
				if (itc->size() < cmin || itc->size() > cmax)
					itc = contours.erase(itc);
				else
					++itc;
			}
			// -1 = 모든 외곽선 그리기, scalar(255)= 하얗게, 두께 2로
			drawContours(imgThres, contours, -1, Scalar(255), 2);

			Rect k;


			for (int a = 0; a<contours.size(); a++)
			{
				k = boundingRect(Mat(contours[a]));

				rectangle(imgRoad, k, Scalar(255), 2);	//disparity에 그 사각형 집어넣기

			}

			findContours(imgThres2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			cmin = 1000;  // 최소 외곽선 길이
			cmax = 5000; // 최대 외곽선 길이

			vector<vector<Point>>::const_iterator itc2 = contours.begin();

			while (itc2 != contours.end()) {
				if (itc2->size() < cmin || itc2->size() > cmax)
					itc2 = contours.erase(itc2);
				else
					++itc2;
			}
			// 원본 영상 내 외곽선 그리기

			Mat original2 = imgThres2;
			// 모든 외곽선 그리기, 하얗게, 두께를 2로
			drawContours(original2, contours, -1, Scalar(255), 2);

			Rect r, u;	//rect 도 vector로 하면 안됌 이미 contours 가 vector니까 할필요 ㄴㄴ
			for (int i = 0; i<contours.size(); i++)
			{
				r = boundingRect(Mat(contours[i]));
				if (r.height>30)
				{
					rectangle(imgRoad, r, Scalar(125), 2);	//disparity에 그 사각형 집어넣기

				}
			}

			imshow("no road", imgRoad);

			Mat element = getStructuringElement(CV_SHAPE_RECT, Size(15, 15));	// 커널 생성

			Mat temp;
			morphologyEx(imgRoad, temp, CV_MOP_OPEN, element);	// 모폴로지 열기

			//	imshow("no road-mo", temp);


			Mat occupancy = occupancygrid(temp);
			imshow("occupancy grid", occupancy);

			//imshow("disparity with contours",disp8);
			waitKey(1);
			printf("\n");
		}
	} //for문 end;
	return 0;
}

static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}