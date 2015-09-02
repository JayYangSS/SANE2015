#include"StixelEstimation.h"


int main()
{
	double dtime = 0;

	//Declaration
	Mat imgLeftInput, imgRightInput;
	CStixelEstimation objStixelEstimation;

	//off-line param setting
	objStixelEstimation.SetParam(CStixelEstimation::Daimler);
	objStixelEstimation.m_flgDisplay = true;
	objStixelEstimation.m_flgVideo = false;

	//on-line
	imgLeftInput = imread("Left_923730u.pgm", 0);
	imgRightInput = imread("Right_923730u.pgm", 0);

	int64 t = getTickCount();
	
	objStixelEstimation.SetImage(imgLeftInput, imgRightInput);
	objStixelEstimation.CreateDisparity();
	//objStixelEstimation.ImproveDisparity(); // image hole removal
	
	t = getTickCount() - t;
	dtime = t * 1000 / getTickFrequency();
	printf("disparity Time elapsed: %fms\n", dtime);


	objStixelEstimation.Display();

	return 0;
}