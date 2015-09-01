#include"StixelEstimation.h"


int main()
{
	//Declaration
	Mat imgLeftInput, imgRightInput;
	CStixelEstimation objStixelEstimation;

	//off-line param setting
	objStixelEstimation.SetParam(CStixelEstimation::Daimler);
	objStixelEstimation.m_flgDisplay = true;
	objStixelEstimation.m_flgVideo = false;

	//on-line
	//imgLeftInput = imread();
	//imgRightInput = imread();
	objStixelEstimation.SetImage(imgLeftInput, imgRightInput);
	objStixelEstimation.CreateDisparity();



	return 0;
}