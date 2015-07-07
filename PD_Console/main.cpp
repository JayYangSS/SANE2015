#include "commons.h"

int main()
{
	//SDetector sACF;
	//LoadDetector("file.txt", sACF);

	VideoCapture vcap("PD_sunny.mp4");
	if (!vcap.isOpened())
		return -1;

	Mat imgInput;
	int delayms = 1;

	while (1)
	{
		vcap >> imgInput;
		if (imgInput.empty()) break;
		imshow("asdf", imgInput);
		int key = waitKey(delayms);
		if (key == 27) break;
		else if (key == 32) delayms = 1-delayms;
		else if (key == 'f') { delayms = 0; }
	}

	//acfDetect(imgInput, sACF);

	return 0;
}

void LoadDetector(string strFileName, SDetector& detector)
{

}

Rect_<int> acfDetect(Mat& img, SDetector detector)
{

	return Rect_<int>(0, 0, 0, 0);
}

void chnsPyramid(Mat& img, SDetector::opts::pPyramid pyramid)
{
	
}

void chnsCompute();