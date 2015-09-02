#include "PedestrianDetection.h"

int main()
{
	VideoCapture vcap("../CVLAB_dataset/data/$$2015-03-18-09h-59m-40s_F_normal.mp4");
	if (!vcap.isOpened())
		return -1;

	CPedestrianDetection objPD;
	objPD.LoadClassifier("acf_classifier.txt");

	Mat imgInput;
	Mat imgDisp;
	int nDelayms = 1;
	int cntFrames = 0;

	while (1)
	{
		vcap >> imgInput;
		if (imgInput.empty()) break;

		Mat imgInputResz;
		Rect rectROI = Rect(1280 / 4, 720 / 4, 1280 / 2, 720 / 2);
		imgInputResz = imgInput(rectROI).clone();
		imgDisp = imgInputResz.clone();

		double t = (double)getTickCount();
		
		objPD.Detect(imgInputResz);
		
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("%.2lf ms  %.1lf fps\n", t * 1000, 1 / t);

		objPD.DrawBoundingBox(imgDisp, CV_RGB(0, 0, 255));

		imshow("Display", imgDisp);
		cntFrames++;

		int nKey = waitKey(nDelayms);
		if (nKey == 27) break;
		else if (nKey == 32) nDelayms = 1 - nDelayms;
		else if (nKey == 'f') { nDelayms = 0; }
	}

	return 0;
}