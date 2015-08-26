#include "PedestrianDetection.h"

int main()
{
	VideoCapture vcap("PD_sunny.mp4");
	if (!vcap.isOpened())
		return -1;

	CPedestrianDetection objPD;
	objPD.LoadClassifier("acf_classifier.txt");

	Mat imgInput;
	Mat imgDisp;
	int nDelayms = 1;

	while (1)
	{
		vcap >> imgInput;
		if (imgInput.empty()) break;

		imgDisp = imgInput.clone();
		Mat imgInput2;
		resize(imgInput, imgInput2, Size(640, 360));

		double t = (double)getTickCount();
		
		objPD.Detect(imgInput2);
		
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("%.2lf\n", t * 1000);
		
		objPD.DrawBoundingBox(imgInput2);
		imshow("asdf", imgInput2);

		int nKey = waitKey(nDelayms);
		if (nKey == 27) break;
		else if (nKey == 32) nDelayms = 1 - nDelayms;
		else if (nKey == 'f') { nDelayms = 0; }
	}

	return 0;
}
