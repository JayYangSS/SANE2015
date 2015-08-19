#include "PedestrianDetection.h"

int main()
{
	VideoCapture vcap("PD_sunny.mp4");
	if (!vcap.isOpened())
		return -1;

	CPedestrianDetection objPD;

	Mat imgInput;
	Mat imgDisp;
	int nDelayms = 1;

	while (1)
	{
		vcap >> imgInput;
		if (imgInput.empty()) break;

		imgDisp = imgInput.clone();

		objPD.Detect(imgInput);
		objPD.DrawBoundingBox(imgDisp);

		imshow("asdf", imgDisp);

		int nKey = waitKey(nDelayms);
		if (nKey == 27) break;
		else if (nKey == 32) nDelayms = 1 - nDelayms;
		else if (nKey == 'f') { nDelayms = 0; }
	}

	return 0;
}
