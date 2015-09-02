#include "PedestrianDetection.h"

int main()
{
	VideoCapture vcap("../CVLAB_dataset/data/$$2015-03-18-09h-59m-40s_F_normal.mp4"); // y=0.7513x+337.75
	//VideoCapture vcap("../CVLAB_dataset/data/2015-02-24-17h-52m-23s_F_event.avi"); // y=0.9037x+359.41
	if (!vcap.isOpened())
		return -1;

	Rect rectROI = Rect(1280 / 4, 720 / 4, 1280 / 2, 720 / 2);

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
		imgInputResz = imgInput(rectROI).clone();
		imgDisp = imgInputResz.clone();

		double t = (double)getTickCount();
		
		objPD.Detect(imgInputResz);
		
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("%.2lf ms  %.1lf fps\n", t * 1000, 1 / t);

		vector<Rect_<int> > bb = objPD.m_vecrectDetectedBB;
		for (int i = 0; i < (int)bb.size(); i++)
		{
			bb[i].x += rectROI.x;
			bb[i].y += rectROI.y;
		}

		for (int i = 0; i < (int)bb.size(); i++)
		{
			double est = (double)bb[i].height*0.7513 + 337.75;
			if (abs((double)bb[i].br().y - est) > 10)
			{
				objPD.m_vecrectDetectedBB.erase(objPD.m_vecrectDetectedBB.begin() + i);
				bb.erase(bb.begin() + i);
				i--;
			}
		}

		objPD.DrawBoundingBox(imgDisp, CV_RGB(255, 255, 255));

		imshow("Display", imgDisp);
		cntFrames++;

		int nKey = waitKey(nDelayms);
		if (nKey == 27) break;
		else if (nKey == 32) nDelayms = 1 - nDelayms;
		else if (nKey == 'f') { nDelayms = 0; }
	}

	return 0;
}