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

using namespace cv;
using namespace std;

//2.1 calcul the disparity map between two images
Mat calculDisp(Mat im1, Mat im2){
	Mat disp, disp8;
	cv::StereoSGBM sgbm(0, 32, 7, 8 * 7 * 7, 32 * 7 * 7, 2, 0, 5, 100, 32, true);
	sgbm(im1, im2, disp);
	disp.convertTo(disp8, CV_8U);
	return disp;
}


//2.2 Computation of the 3D coordinates and remove all the pixels with a Z coodinate higher or lower than a threshold
Mat compute3DAndRemove(Mat disp){
	Mat res(disp.rows, disp.cols, CV_8U, Scalar(0));
	float thresMax = 2.5;
	float thresMin = 0.2;
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

//2.3.1 Compute the v-disparity image histogram of disparity matrix
Mat computeVDisparity(Mat img){
	int maxDisp = 32;
	Mat vDisp(img.rows, maxDisp, CV_8U, Scalar(0));
	for (int u = 0; u<img.rows; u++){
		for (int v = 0; v<img.cols; v++){
			int disp = (img.at<unsigned char>(u, v)) / 8;
			//if(disp>0 && disp < maxDisp){
			if (disp>6 && disp < maxDisp - 2){ //We remove pixels of sky and car to compute the roadline
				vDisp.at<unsigned char>(u, disp) += 1;
			}
		}
	}
	return vDisp;
}

//2.3.3 Removing noise from v disparity map
Mat removeNoise(Mat img){
	Mat res = img.clone();
	int thresh = 30;
	//threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
	threshold(img, res, thresh, 255, 3); // 255 > max binary value, 4 > zero inverted (to 0 under thresh)
	return res;
}

//3.3.3 Storing pixel of a map into an std::vector
std::vector<Point2f> storeRemainingPoint(Mat img){
	std::vector<Point2f> res;
	res.clear();
	for (int u = 0; u<img.rows; u++){
		for (int v = 0; v<img.cols; v++){
			int value = img.at<unsigned char>(u, v);
			if (value > 0){
				res.push_back(Point2f(u, v));
			}
		}
	}
	return res;
}


Mat filterRansac(Vec4f line, Mat img){
	Mat res(img.rows, img.cols, CV_8U, Scalar(0));
	double slope = line[0] / line[1];
	double orig = line[2] - slope*line[3];
	for (int u = 0; u<img.rows; u++){
		for (int v = 0; v<img.cols; v++){
			int value = img.at<unsigned char>(u, v);
			double test = orig + slope*value / 16 - u;
			if (abs(test) < 30){
				res.at<unsigned char>(u, v) = value;
			}
			else{
				res.at<unsigned char>(u, v) = 0;
			}
		}
	}
	return res;
}

int main()
{
	// Open image from input file in grayscale
	Mat img1 = imread("Left_923730u.pgm", 0);
	Mat img2 = imread("Right_923730u.pgm", 0);
	//2.1.1 Displaying left and right loaded imgs
	//imshow("left image", img1);
	//imshow("right image", img2);
	double dtime = 0;
	int64 t = getTickCount();
	// 2.1.2 Disparity map between the two images using SGBM
	Mat disp, disp8;
	disp = calculDisp(img1, img2);
	disp.convertTo(disp8, CV_8U);
	imshow("diparity map (2.1.2)", disp8);

	//2.2.1, 2.2.2 remove the road (z<0.2m) and upper pixels(z>2.5m)
	Mat dispFiltered = compute3DAndRemove(disp8);
	imshow("Disparity map with height filter (2.2.2)", dispFiltered);

	//2.3.1 Compute VDisparity
	Mat vDispNoisy = computeVDisparity(disp);
	imshow("VDisparity method (2.3.1)", vDispNoisy);

	//2.3.2 manually computation of ho & po :
	/* We measure a line y = 3.36x +143,56
	* where y is the heigh in pixel image and x the luminosity between 0 and 32 of disparity map
	* The luminosity depends on the depth x=32 -> depth = 0 and x=0 -> depth = +infiny
	* But this correlation is not linear and the y is not linked with y in real world
	* so computing the height of road and it's slope seems difficult there..
	*/

	//2.3.3 Removing noise from v disparity map
	Mat vDisp = removeNoise(vDispNoisy);
	imshow("VDisparity map no noise (2.3.3)", vDisp);

	//2.3.3 extracting the remaining points and removing the floor
	std::vector<Point2f> tempVec = storeRemainingPoint(vDisp);
	Vec4f roadLine;
	fitLineRansac(tempVec, roadLine);
	std::cout << "2.3.3, road line : " << roadLine << std::endl;
	//2.3.4 removing pixels under the road according to the line of the road
	Mat dispFiltered2 = filterRansac(roadLine, disp8);
	imshow("Final Disparity filtered (2.3.4)", dispFiltered2);

	Mat morph1, morph2;
	//2.4.1 morphological erosion and dilatation :
	int transf_type;
	//transf_type = MORPH_RECT;
	//transf_type = MORPH_CROSS;
	transf_type = MORPH_ELLIPSE;
	int transf_size = 8;

	Mat element = getStructuringElement(transf_type,
		Size(2 * transf_size + 1, 2 * transf_size + 1),
		Point(transf_size, transf_size));
	erode(dispFiltered, morph1, element);
	//imshow("Erosion disparity (2.4.1)", morph1);
	dilate(morph1, morph2, element);
	imshow("Eroded then dilated disparity1 (2.4.1)", morph2);

	Mat morph3, morph4;
	erode(dispFiltered2, morph3, element);
	//imshow("Erosion disparity (2.4.1)", morph3);
	dilate(morph3, morph4, element);
	imshow("Eroded then dilated disparity2 (2.4.1)", morph4);


	//2.4.2 extraction of components using segmentDisparity
	Mat segmentedDisparity;
	segmentDisparity(morph2, segmentedDisparity);
	//résultas un peu louches ici puisque les pincipaux obstacles sont enlevés...
	//Les cluster paraissaient plus évident sur l'image erodée et dilatée.
	//Aussi, on est censé avoir 1 couleur par objet ce qui n'est pas du tout le cas.
	imshow("Segmented disparity1 (2.4.2)", segmentedDisparity * 16);

	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("image , Time elapsed: %fms\n", dtime);

	// Display images and wait for a key press
	waitKey();
	return 0;
}




