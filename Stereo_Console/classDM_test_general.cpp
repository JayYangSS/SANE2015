#include"DistMeasure.h"

int main()
{
	Mat imgLeftInput, imgRightInput;

	////////////////////////////////////////////////off-line//////////////////////////////////////////////////
	// class param setting
	CDistMeasure objDistMeasure;
	objDistMeasure.SetParam(CDistMeasure::Daimler);//if you use KITTI, write "KITTI" instead of Daimler
	objDistMeasure.m_flgVideo = false;	// video or image
	objDistMeasure.m_flgDisplay = false;// image show or not
		//objDistMeasure.SetParam(BASELINE, FOCAL, 0, -1.8907);
	//objDistMeasure.m_nDistAlg = CDistMeasure::STEREOBM;
	//objDistMeasure.m_flgVideo = false; // video or image

	/////////////////////////////////////////////////on-line//////////////////////////////////////////////////
	//image read
	imgLeftInput = imread("Left_923730u.pgm", 0);
	imgRightInput = imread("Right_923730u.pgm", 0);
	
	vector<Rect_<int> > vecRectROI;

	//detect object
	Rect rectROI(306, 136, 126, 202); // a example(GT : 10.71m )
	
	vecRectROI.push_back(rectROI);

	//Calculate disparity & distance
	objDistMeasure.SetImage(imgLeftInput, imgRightInput, vecRectROI);
	objDistMeasure.CalcDistImg(CDistMeasure::FVLM);//Parameter seq : FVLM, mono, stereoBM, stereoSGBM

	//result : distance
	for (int i = 0; i < vecRectROI.size(); i++){
		cout << objDistMeasure.m_vecdDistance[i] <<"m" << endl;
	}
	
	rectangle(imgLeftInput, rectROI, Scalar(255));
	imshow("Left input", imgLeftInput);
	waitKey(0);


	return 0;
}