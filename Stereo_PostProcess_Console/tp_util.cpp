/*
*  tp_util.cpp
*  Make sense obstacle 3D position & ground constraint
*  using ransac line fitting and disparity segmentation
*  Created by T.K.Woo on July/23/2015.
*  Copyright 2015 CVLAB at Inha. All rights reserved.
*
*/
#include "tp_util.h"

#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void fitLineRansac(const std::vector<Point2f> points,
	Vec4f &line,
	int iterations,
	double sigma,
	double a_max)
{
	int n = points.size();
	//cout <<"point size : "<< n << endl;
	if (n<2)
	{
		return;
	}

	RNG rng;
	double bestScore = -1.;
	for (int k = 0; k<iterations; k++)
	{
		int i1 = 0, i2 = 0;
		double dx = 0;
		while (i1 == i2)
		{
			i1 = rng(n);
			i2 = rng(n);
		}
		Point2f p1 = points[i1];
		Point2f p2 = points[i2];

		Point2f dp = p2 - p1;
		dp *= 1. / norm(dp);
		double score = 0;

		if (fabs(dp.x/1.e-5f) && fabs(dp.y / dp.x) <= a_max)
		{
			for (int i = 0; i<n; i++)
			{
				Point2f v = points[i] - p1;
				double d = v.y*dp.x - v.x*dp.y;
				score += exp(-0.5*d*d / (sigma*sigma));
			}
		}
		if (score > bestScore)
		{
			line = Vec4f(dp.x, dp.y, p1.x, p1.y);
			bestScore = score;
		}
	}
}

unsigned int segmentDisparity(const Mat &disparity, Mat &output)
{
	output = Mat::zeros(disparity.size(), CV_32SC1);
	Mat tmp = Mat::zeros(disparity.size(), CV_8UC1);
	Mat tmp2 = Mat::zeros(disparity.size(), CV_32SC1);

	disparity.convertTo(tmp, CV_8UC1);
	Mat mask = Mat::zeros(disparity.size().height + 2,
		disparity.size().width + 2,
		CV_8UC1);

	Mat mask2 = mask(Rect(1, 1, disparity.size().width, disparity.size().height));
	//threshold(tmp, mask2, 0, 255, CV_THRESH_BINARY_INV);

	unsigned int k = 1;

	for (int i = 0; i<tmp.rows; i++)
	{
		const unsigned char* buf = tmp.ptr<unsigned char>(i);
		for (int j = 0; j<tmp.cols; j++)
		{
			unsigned char d = buf[j];
			if (d>0)
			{
				if (floodFill(tmp,
					mask,
					Point(j, i),
					Scalar(1),
					NULL,
					cvScalarAll(0),
					cvScalarAll(0),
					8 + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY)>0)
				{
					mask2.convertTo(tmp2, CV_32SC1, k);
					output += tmp2;
					tmp.setTo(0, mask2);
					mask.setTo(0);
					k++;
				}
			}
		}
	}
	return k;
}

