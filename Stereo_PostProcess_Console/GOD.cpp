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

int StixelEstimation_col(Mat& imgDispRm, int col, stixel_t& objStixel)
{
	int nIter = imgDispRm.rows / 2;
	uchar chDisp;

	for (int v = 1; v < nIter; v++){
		chDisp = imgDispRm.at<uchar>(imgDispRm.rows - v, col);
		
		if (imgDispRm.at<uchar>(v, col)>0 && objStixel.nHeight == -1){ objStixel.nHeight = v; nIter = imgDispRm.rows - v; }
		if (chDisp > 0 && objStixel.nGround == -1){
			objStixel.nGround = imgDispRm.rows - v;
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
		if (u < 30) { 
			objStixels[u].chDistance =  0;
			objStixels[u].nGround = 0;
			objStixels[u].nHeight = 0;
		}
		else StixelEstimation_col(imgDispRm, u, objStixels[u]);
	}
	return 0;
}

// calcul the disparity map between two images
Mat calculDisp(Mat im1, Mat im2){
	Mat disp, disp8;
	StereoSGBM sgbm (0, 48, 5, 8 * 5 * 5, 32 * 5 * 5, 1, 0, 5, 100, 32, false);
	//sgbm = StereoSGBM(0, 32, 5, 8 * 5 * 5, 8 * 5 * 5, 1, 5, 10, 9, 4, false);
	sgbm(im1, im2, disp);
	//disp.convertTo(disp8, CV_8U);
	return disp;
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
			if (test > 10){
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
			if (u > (orig+slope*value)){
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

int main()
{
	// Open image from input file in grayscale
	Mat img1 = imread("Left_923730u.pgm", 0);
	Mat img2 = imread("Right_923730u.pgm", 0);
	// Displaying left and right loaded imgs
	imshow("left image", img1);
	//imshow("right image", img2);
	double dtime = 0;
	int64 t = getTickCount();
	// Disparity map between the two images using SGBM
	Mat disp, disp8;
	disp = calculDisp(img1, img2);
	disp.convertTo(disp8, CV_8U, 255 / (48*16.));
	imshow("diparity map", disp8);

	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("disparity Time elapsed: %fms\n", dtime);

	
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
	

	/* We measure a line y = 3.36x +143,56
	* where y is the heigh in pixel image and x the luminosity between 0 and 32 of disparity map
	* The luminosity depends on the depth x=32 -> depth = 0 and x=0 -> depth = +infiny
	* But this correlation is not linear and the y is not linked with y in real world
	* so computing the height of road and it's slope seems difficult there..
	*/
	t = getTickCount();
	//Removing noise from v disparity map
	Mat vDisp = removeNoise(vDispNoisy);
	imshow("VDisparity map no noise", vDisp);
	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("Vdisparity-remove noise Time elapsed: %fms\n", dtime);
	//if (waitKey(0) == 27) return 0;
	imwrite("vdisparity.bmp", vDisp);

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
	printf("fitRansac Time elapsed: %fms\n", dtime);
	
	Mat imgDispfilter3 = FilterHeight3m(-3.042016, 248.22857, dispFiltered2);// 1m
	imshow("remove sky", imgDispfilter3);

	stixel_t objStixels[WIDTH];
	StixelEstimation_img(imgDispfilter3, objStixels);
	//cout << objStixels << endl;

	/*Mat imgSobel;
	Sobel(imgDispfilter3, imgSobel, -1, 0, 2);
	imshow("sobel", imgSobel);
	imwrite("sobel.bmp", imgSobel);*/

	threshold(dispFiltered2, dispFiltered2, 1, 255, CV_THRESH_BINARY);
	Mat imgFiltered;
	bitwise_and(dispFiltered2, img1, imgFiltered);
	
	imshow("ground remove", imgFiltered);
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




