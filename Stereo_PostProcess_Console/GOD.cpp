/*
*  General obstacle detection.cpp
*  Make sense obstacle 3D position & ground constraint
*
*  Created by T.K.Woo on July/23/2015.
*  Copyright 2015 CVLAB at Inha. All rights reserved.
*
*/

// #include "disparity.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <stdio.h>

#include "tp_util.h"

#define BASELINE 0.25		//unit : meter
#define FOCAL_LENGTH 1200	//unit : pixel

#define WIDTH 640
#define HEIGHT 480

using namespace cv;
using namespace std;

unsigned char g_pseudoColorLUT[256][3]; // RGB

struct stixel_t
{
	int nGround;
	int nHeight;
	uchar chDistance;
	stixel_t(){
		nGround = -1;
		nHeight = -1;
		chDistance = 0;
	}
};
/**
@brief	make Pseudo-Color LUT (Look up table)
@param	-
@return	-
*/
void makePseudoColorLUT()
{
	int b = 125;
	int g = 0;
	int r = 0;

	int idx = 0;

	int mode = 0;
	// mode = 0 : increasing 'b'
	// mode = 1 : increasing 'g'
	// mode = 2 : decreasing 'b'
	// mode = 3 : increasing 'r'
	// mode = 4 : decreasing 'g'
	// mode = 5 : decreasing 'r'

	while (1)
	{
		g_pseudoColorLUT[idx][0] = b;
		g_pseudoColorLUT[idx][1] = g;
		g_pseudoColorLUT[idx][2] = r;

		if (b == 255 && g == 0 && r == 0)
			mode = 1;
		else if (b == 255 && g == 255 && r == 0)
			mode = 2;
		else if (b == 0 && g == 255 && r == 0)
			mode = 3;
		else if (b == 0 && g == 255 && r == 255)
			mode = 4;
		else if (b == 0 && g == 0 && r == 255)
			mode = 5;

		switch (mode)
		{
		case 0: b += 5; break;
		case 1: g += 5; break;
		case 2: b -= 5; break;
		case 3: r += 5; break;
		case 4: g -= 5; break;
		case 5: r -= 5; break;
		default: break;
		}

		if (idx == 255)
			break;

		idx++;
	}
}

/**
@brief	convert Pseudo-Color Image
@param	srcGray: 입력 gray 영상, dstColor: 출력 color 영상
@return	-
*/
void cvtPseudoColorImage(Mat srcGray, Mat& dstColor)
{
	for (int i = 0; i<srcGray.rows; i++)
	{
		for (int j = 0; j<srcGray.cols; j++)
		{
			unsigned char val = srcGray.data[i*srcGray.cols + j];
			if (val == 0) continue;
			dstColor.data[(i*srcGray.cols + j) * 3 + 0] = g_pseudoColorLUT[val][0];
			dstColor.data[(i*srcGray.cols + j) * 3 + 1] = g_pseudoColorLUT[val][1];
			dstColor.data[(i*srcGray.cols + j) * 3 + 2] = g_pseudoColorLUT[val][2];
		}
	}
}


int DrawStixel(Mat& imgColorDisp, stixel_t* objStixels){
	for (int u = 0; u < imgColorDisp.cols; u++){
		line(imgColorDisp,
			Point(u, objStixels[u].nGround),
			Point(u, objStixels[u].nHeight),
			Scalar(0, 255 - objStixels[u].chDistance, objStixels[u].chDistance));
	}
	return 0;
}
int DrawStixel_gray(Mat& imgGrayDisp, stixel_t* objStixels){
	for (int u = 0; u < imgGrayDisp.cols; u++){
		line(imgGrayDisp,
			Point(u, objStixels[u].nGround),
			Point(u, objStixels[u].nHeight),
			Scalar(objStixels[u].chDistance));
	}
	return 0;
}

int StixelEstimation_col(Mat& imgDispRm, int col, stixel_t& objStixel)
{
	int nIter = imgDispRm.rows / 2;
	uchar chDisp;

	for (int v = 1; v < nIter; v++){
		chDisp = imgDispRm.at<uchar>(imgDispRm.rows - v, col);
		
		if (imgDispRm.at<uchar>(v, col)>0 && objStixel.nHeight == -1){ objStixel.nHeight = v; nIter = imgDispRm.rows - v; }
		if (chDisp > 0 && objStixel.nGround == -1){
			objStixel.nGround = imgDispRm.rows - v +10; //10 is manually
			objStixel.chDistance = chDisp; // 2015.08.11 have to fix
			nIter = imgDispRm.rows - v;
		}
	}
	//cout << col << " : " << objStixel.nGround << ", " << objStixel.nHeight << endl;
	return 0;
}
int StixelEstimation_img(Mat& imgDispRm, stixel_t* objStixels)
{
	for (int u = 0; u < imgDispRm.cols; u++){
		if (u < 30) { //manually
			objStixels[u].chDistance =  0;
			objStixels[u].nGround = 0;
			objStixels[u].nHeight = 0;
		}
		else StixelEstimation_col(imgDispRm, u, objStixels[u]);
	}
	return 0;
}

// calcul the disparity map between two images
int calculDisp(Mat& im1, Mat& im2, Mat& imgDisp16){
	
	StereoBM bm;
	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = 13;
	bm.state->minDisparity = 1;
	bm.state->numberOfDisparities = 48;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 25;//9;
	bm.state->speckleRange = 32;//4;
	bm.state->disp12MaxDiff = 1;

	//StereoSGBM sgbm (0, 48, 5, 8 * 5 * 5, 32 * 5 * 5, 1, 0, 5, 100, 32, false);
	//sgbm = StereoSGBM(0, 32, 5, 8 * 5 * 5, 8 * 5 * 5, 1, 5, 10, 9, 4, false);
	
	bm(im1, im2, imgDisp16, CV_16S);
	//sgbm(im1, im2, imgDisp16);
	//disp.convertTo(disp8, CV_8U);
	return 0;
}

int PostProcess(Mat& imgDisp8, int nNumOfDisp)
{
	uchar chTempCur = 0;
	uchar chTempPrev = 0;
	for (int v = 0; v < imgDisp8.rows; v++){
		for (int u = nNumOfDisp; u < imgDisp8.cols; u++){
			chTempCur = imgDisp8.at<uchar>(v, u);
			if (chTempCur == 0) imgDisp8.at<uchar>(v, u)=chTempPrev;
			else chTempPrev = chTempCur;
		}
	}
	
	return 0;
}

// Computation of the 3D coordinates and remove all the pixels with a Z coodinate higher or lower than a threshold
Mat compute3DAndRemove(Mat disp){
	Mat res(disp.rows, disp.cols, CV_8U, Scalar(0));
	float thresMax = 2.5;
	float thresMin = -0.6;
	//float u0 = 258; //unused here
	float v0 = 156;
	float au = 410;
	float av = 410;
	float b = 0.22;
	float z0 = 1.28;
	for (int v = 0; v<disp.rows; v++){
		for (int u = 0; u<disp.cols; u++) {
			int d = disp.at<unsigned char>(v, u);
			//float x = (u-u0)*b/d - (b/2); //unused here
			//float y = au*b/d;             //unused here
			float z = z0 - (((v - v0)*au*b) / (av*d / 16));
			if (z<thresMin || z>thresMax){
				res.at<unsigned char>(v, u) = 0;
			}
			else{
				res.at<unsigned char>(v, u) = d;
			}
		}
	}
	return res;
}

// Compute the v-disparity image histogram of disparity matrix
Mat computeVDisparity(Mat img){
	int maxDisp = 255;
	Mat vDisp(img.rows, maxDisp, CV_8U, Scalar(0));
	for (int u = 0; u<img.rows; u++){
		if (u < 200) continue; // we are finding ground. therefore we check pixels below vanishing point 
		for (int v = 0; v<img.cols; v++){
			int disp = (img.at<uchar>(u, v));// / 8;
			//if(disp>0 && disp < maxDisp){
			if (disp>6 && disp < maxDisp - 2){ //We remove pixels of sky and car to compute the roadline
				vDisp.at<unsigned char>(u, disp) += 1;
			} 
		}
	}
	return vDisp;
}

// Removing noise from v disparity map
Mat removeNoise(Mat img){
	Mat res = img.clone();
	int thresh = 50;
	//threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
	threshold(img, res, thresh, 255, 3); // 255 > max binary value, 4 > zero inverted (to 0 under thresh)
	return res;
}

// Storing pixel of a map into an std::vector
std::vector<Point2f> storeRemainingPoint(Mat img){
	std::vector<Point2f> res;
	res.clear();
	for (int u = 200; u<img.rows; u++){
		for (int v = 0; v<img.cols; v++){
			int value = img.at<unsigned char>(u, v);
			if (value > 0){
				res.push_back(Point2f(u, v));
			}
		}
	}
	return res;
}

Mat filterRansac(Vec4f line, Mat& img){
	//Mat res(img.rows, img.cols, CV_8U, Scalar(0));
	double slope = line[0] / line[1];
	double orig = line[2] - slope*line[3];
	printf("v=%lf * d + %lf\n", slope, orig);
	//slope = -0.7531;
	//orig = 200.;
	for (int u = 200; u<img.rows; u++){
		for (int v = 0; v<img.cols; v++){
			int value = img.at<unsigned char>(u, v);
			double test = orig + slope*value - u;
			if (test > 15){
				img.at<unsigned char>(u, v) = value;
				//res.at<unsigned char>(u, v) = value;
			}
			else{
				img.at<unsigned char>(u, v) = 0;
				//res.at<unsigned char>(u, v) = 0;
			}
		}
	}
	return img;
}
Mat FilterHeight3m(double slope, double orig, Mat& img)
{
	for (int u = 0; u<img.rows; u++){
		for (int v = 0; v<img.cols; v++){
			int value = img.at<unsigned char>(u, v);
			//double test = orig + slope*value - u;
			if (u < (orig+slope*value)){
				//img.at<unsigned char>(u, v) = value;
				img.at<unsigned char>(u, v) = 0;
				//res.at<unsigned char>(u, v) = value;
			}
			//else{
			//	img.at<unsigned char>(u, v) = 0;
			//	//res.at<unsigned char>(u, v) = 0;
			//}
		}
	}
	return img;
}

int main()
{
	double dtime = 0;
	int64 t = getTickCount();
	int64 tp = getTickCount();

	// Open image from input file in grayscale
	Mat img1 = imread("Left_923730u.pgm", 0);
	Mat img2 = imread("Right_923730u.pgm", 0);
	Mat disp, disp8;

	// Displaying left and right loaded imgs
	imshow("left image", img1);
	//imshow("right image", img2);
	// Disparity map between the two images using SGBM
	
	calculDisp(img1, img2, disp);
	disp.convertTo(disp8, CV_8U, 255 / (48*16.));
	imshow("diparity map", disp8);
	

	

	//Mat imgDisp8Temp = disp8.clone();
	//PostProcess(imgDisp8Temp, 48);
	PostProcess(disp8, 48);
	//imshow("post process", imgDisp8Temp);

	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("disparity Time elapsed: %fms\n", dtime);

	Mat imgColorDisp8;
	cvtColor(disp8, imgColorDisp8, CV_GRAY2BGR);

	makePseudoColorLUT();
	cvtPseudoColorImage(disp8, imgColorDisp8);
	
	Mat imgtemp;
	cvtColor(img1, imgtemp, CV_GRAY2BGR);
	addWeighted(imgColorDisp8, 0.5, imgtemp, 0.5, 0.0, imgColorDisp8);
	/*for (int v = 0; v < imgColorDisp8.rows; v++){
		for (int u = 0; u < imgColorDisp8.cols; u++){
			if (disp8.at<uchar>(v, u) == 0) continue;
			imgColorDisp8.at<Vec3b>(v, u)[0] = 0;
			imgColorDisp8.at<Vec3b>(v, u)[1] = 255 - disp8.at<uchar>(v, u);
			imgColorDisp8.at<Vec3b>(v, u)[2] = disp8.at<uchar>(v, u);
		}
	}*/
	imshow("color disp", imgColorDisp8);
	
	t = getTickCount();
	//Mat dispFiltered = compute3DAndRemove(disp8);
	//imshow("Disparity map with height filter (2.2.2)", dispFiltered);

	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("3D remove Time elapsed: %fms\n", dtime);

	// Compute VDisparity
	t = getTickCount();
	Mat vDispNoisy = computeVDisparity(disp8);
	
	imshow("VDisparity method", vDispNoisy);

	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("Vdisparity Time elapsed: %fms\n", dtime);
	

	/* We measure a line y = (?) d + (?)
	* where y is the heigh in pixel image and d the luminosity between 0 and 255 of disparity map(disp8)
	* The luminosity depends on the depth x=255 -> depth = 0 and x=0 -> depth = +infiny
	* But this correlation is not linear and the y is not linked with y in real world
	*/
	t = getTickCount();
	//Removing noise from v disparity map
	Mat vDisp = removeNoise(vDispNoisy);
	imshow("VDisparity map no noise", vDisp);
	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("Vdisparity-remove noise Time elapsed: %fms\n", dtime);
	//if (waitKey(0) == 27) return 0;
	//imwrite("vdisparity.bmp", vDisp);

	t = getTickCount();
	// extracting the remaining points and removing the floor
	std::vector<Point2f> tempVec = storeRemainingPoint(vDisp);
	//cout << tempVec << endl;
	Vec4f roadLine;
	fitLineRansac(tempVec, roadLine);
	//roadLine = Vec4f(100, 100, 340, 100);
	std::cout << "road line : " << roadLine << std::endl;
	// removing pixels under the road according to the line of the road
	Mat dispFiltered2 = filterRansac(roadLine, disp8);
	imshow("Final Disparity filtered", dispFiltered2);
	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("fitRansac Ground remove Time elapsed: %fms\n", dtime);
	if (waitKey(0) == 27) return 0;
	//return 0;

	t = getTickCount() - t;

	Mat imgDispfilter3 = FilterHeight3m(-1.842016, 220.22857, dispFiltered2);// 1m
	imshow("remove sky", imgDispfilter3);

	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("sky remove Time elapsed: %fms\n", dtime);

	t = getTickCount() - t;

	stixel_t objStixels[WIDTH];
	StixelEstimation_img(imgDispfilter3, objStixels);
	//cout << objStixels << endl;

	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("Stixel estimation Time elapsed: %fms\n", dtime);

	tp = getTickCount() - tp;
	dtime = tp * 1000 / getTickFrequency();
	printf("Total Time elapsed: %fms\n", dtime);

	/*Mat imgSobel;
	Sobel(imgDispfilter3, imgSobel, -1, 0, 2);
	imshow("sobel", imgSobel);
	imwrite("sobel.bmp", imgSobel);*/

	threshold(dispFiltered2, dispFiltered2, 1, 255, CV_THRESH_BINARY);
	Mat imgFiltered;
	bitwise_and(dispFiltered2, img1, imgFiltered);

	Mat imgColorDisp, imgMask;
	cvtColor(imgFiltered, imgColorDisp, CV_GRAY2BGR);
	imgMask = imgColorDisp.clone();
	//DrawStixel(imgMask, objStixels);
	DrawStixel_gray(imgFiltered, objStixels);
	//imshow("n", imgFiltered);
	cvtPseudoColorImage(imgFiltered, imgMask);
	addWeighted(imgtemp, 0.5, imgMask, 0.5, 0.0, imgColorDisp);
	imshow("color", imgColorDisp);
	
	//imshow("ground remove", imgFiltered);
	if (waitKey(0) == 27) return 0;

	//Mat morph1, morph2;
	//// morphological erosion and dilatation :
	//int transf_type;
	////transf_type = MORPH_RECT;
	////transf_type = MORPH_CROSS;
	//transf_type = MORPH_ELLIPSE;
	//int transf_size = 8;

	//Mat element = getStructuringElement(transf_type,
	//	Size(2 * transf_size + 1, 2 * transf_size + 1),
	//	Point(transf_size, transf_size));
	//erode(dispFiltered, morph1, element);
	//dilate(morph1, morph2, element);
	

	//Mat morph3, morph4;
	//erode(dispFiltered2, morph3, element);
	//dilate(morph3, morph4, element);


	//// extraction of components using segmentDisparity
	//Mat segmentedDisparity;
	//segmentDisparity(morph2, segmentedDisparity);
	//imshow("Segmented disparity1", segmentedDisparity * 16);

	//t = getTickCount() - t;
	//dtime = t * 1000 / getTickFrequency();
	//printf("image , Time elapsed: %fms\n", dtime);

	// Display images and wait for a key press
	waitKey();
	return 0;
}




