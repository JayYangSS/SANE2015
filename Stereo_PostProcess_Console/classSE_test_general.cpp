#include"StixelEstimation.h"


int main()
{
	double dtime = 0;//time var
	int nError;		//error check var
	int cnt = 0;	//frame cnt

	//Declaration
	Mat imgLeftInput, imgRightInput;
	CStixelEstimation objStixelEstimation;

	//off-line param setting
	objStixelEstimation.SetParam(CStixelEstimation::Daimler);
	objStixelEstimation.SetStixelWidth(4);
	objStixelEstimation.m_flgDisplay = true;
	objStixelEstimation.m_flgVideo = true;
	objStixelEstimation.m_flgColor = false;

	//on-line
	while (1){
		cnt++;
		
		imgLeftInput = imread("Left_923730u.pgm", 0);
		imgRightInput = imread("Right_923730u.pgm", 0);

		int64 t = getTickCount();

		//case 1: if you want to see the stixel using one line
		objStixelEstimation.StixelEstimation(imgLeftInput, imgRightInput, true);

		////case 2: you can also control each function
		//objStixelEstimation.SetImage(imgLeftInput, imgRightInput);
		////objStixelEstimation.CreateDisparity();
		//objStixelEstimation.CreateDisparity(CStixelEstimation::COLOR,true);
		////objStixelEstimation.ImproveDisparity(); // image hole removal
		//nError = objStixelEstimation.GroundEstimation();
		//if (nError == -1){
		//	printf("Error!!\n");
		//	return -1;
		//}
		//objStixelEstimation.HeightEstimation();
		//objStixelEstimation.StixelDistanceEstimation();
		//objStixelEstimation.DrawStixelsColor();
		
		////case 3: if you want fastest stixel method. It can not draw stixel on image.
		//objStixelEstimation.CreateStixels(imgLeftInput, imgRightInput, true);


		t = getTickCount() - t;
		dtime = t * 1000 / getTickFrequency();
		printf("disparity Time elapsed: %fms\n", dtime);

		if (nError != -1) objStixelEstimation.Display();

		cout << "count : " << cnt << endl;
		if (objStixelEstimation.m_flgVideo == false) break;
	}
	return 0;
}