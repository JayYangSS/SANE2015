#include "PedestrianDetector.h"

CPedestrianDetector::CPedestrianDetector(){
	SetROI(0, 0, 0, 0);
	SetMinSize(0, 0);
	SetMaxSize(0, 0);
}

CPedestrianDetector::CPedestrianDetector(int _x, int _y, int _width, int _height){
	SetROI(_x, _y, _width, _height);
	SetMinSize(32, 64);
	SetMaxSize(_height / 2, _height);
}

void CPedestrianDetector::PedestrianDetectorHaar(const Mat &imgSrc, CascadeClassifier &cascadeHaar, vector<Rect>& vecRectFoundUnfiltered, float scaleFactor, int minNeighbors){
	Mat imgDst = imgSrc(GetROI());

	vector<Rect> vecRectFoundHaar;
	
	cascadeHaar.detectMultiScale(imgDst, vecRectFoundHaar, scaleFactor, minNeighbors, 0, GetMinSize(), GetMaxSize());
	//cascadeHaar.detectMultiScale(imgDst, vecRectFoundHaar, scaleFactor, minNeighbors, CV_HAAR_DO_CANNY_PRUNING, GetMinSize(), GetMaxSize());

	for (unsigned i = 0; i < vecRectFoundHaar.size(); i++){
		Rect rectFound = vecRectFoundHaar[i];

		//rectFound.x = rectFound.x + m_rectROI.x;
		//rectFound.y = rectFound.y + m_rectROI.y;
		
		rectFound.x = rectFound.x + cvRound(rectFound.width*0.05) + m_rectROI.x;
		rectFound.width = cvRound(rectFound.width*0.9);
		rectFound.y = rectFound.y + cvRound(rectFound.height*0.1) + m_rectROI.y;
		rectFound.height = cvRound(rectFound.height*0.8);

		vecRectFoundUnfiltered.push_back(rectFound);
	}
}

void CPedestrianDetector::PedestrianDetectorHOG(const Mat &imgSrc, CascadeClassifier &cascadeHOG, vector<Rect>& vecRectFoundUnfiltered, float scaleFactor, int minNeighbors){
	Mat imgDst = imgSrc(GetROI());

	vector<Rect> vecRectFoundHOG;

	cascadeHOG.detectMultiScale(imgDst, vecRectFoundHOG, scaleFactor, minNeighbors, 0, GetMinSize(), GetMaxSize());

	for (unsigned i = 0; i < vecRectFoundHOG.size(); i++){
		Rect rectFound = vecRectFoundHOG[i];

		rectFound.x = rectFound.x + cvRound(rectFound.width*0.05) + m_rectROI.x;
		rectFound.width = cvRound(rectFound.width*0.9);
		rectFound.y = rectFound.y + cvRound(rectFound.height*0.1) + m_rectROI.y;
		rectFound.height = cvRound(rectFound.height*0.8);

		vecRectFoundUnfiltered.push_back(rectFound);
	}
}

void CPedestrianDetector::PedestrianDetectorHaarHOG(const Mat &imgSrc, CascadeClassifier &cascadeHaar, CascadeClassifier &cascadeHOG,
	vector<Rect>& vecRectFoundUnfiltered, float scaleFactor1st, int minNeighbors1st, float scaleFactor2nd, int minNeighbors2nd){
	Mat imgDst = imgSrc(GetROI());

	vector<Rect> vecRectFoundHaar, vecRectFoundHOG;

	cascadeHaar.detectMultiScale(imgDst, vecRectFoundHaar, scaleFactor1st, minNeighbors1st, 0, GetMinSize(), GetMaxSize());

	for (unsigned i = 0; i < vecRectFoundHaar.size(); i++){
		Rect rectFound = vecRectFoundHaar[i];
		int nOffset = (int)(rectFound.width*0.2f);

		int maxRectX0 = std::max(rectFound.x - nOffset, 0);        // left-side validation
 		int maxRectY0 = std::max(rectFound.y - nOffset * 2, 0);  //

		/*CvRect rectInnerROI = cvRect(maxRectX0, maxRectY0,
			((rectFound.x + rectFound.width + nOffset * 2) >= m_nROIwidth) ? m_nROIwidth - maxRectX0 : rectFound.width + nOffset * 2,
			((rectFound.y + rectFound.height + nOffset * 4) >= m_nROIheight) ? m_nROIheight - maxRectY0 : rectFound.height + nOffset * 4);*/

		CvRect rectInnerROI = cvRect(maxRectX0, maxRectY0,
			((maxRectX0 + rectFound.width + nOffset * 2) >= m_nROIwidth) ? m_nROIwidth - maxRectX0 : rectFound.width + nOffset * 2,
			((maxRectY0 + rectFound.height + nOffset * 4) >= m_nROIheight) ? m_nROIheight - maxRectY0 : rectFound.height + nOffset * 4);

		Mat imgLittleDst = imgDst(rectInnerROI);
		cascadeHOG.detectMultiScale(imgLittleDst, vecRectFoundHOG, scaleFactor2nd, minNeighbors2nd, 0, GetMinSize(), GetMaxSize());

		for (unsigned int j = 0; j < vecRectFoundHOG.size(); j++){
			Rect _rectFound = vecRectFoundHaar[i];

			_rectFound.x = _rectFound.x + cvRound(_rectFound.width*0.05) + m_rectROI.x;
			_rectFound.width = cvRound(_rectFound.width*0.9);
			_rectFound.y = _rectFound.y + cvRound(_rectFound.height*0.1) + m_rectROI.y;
			_rectFound.height = cvRound(_rectFound.height*0.8);

			vecRectFoundUnfiltered.push_back(_rectFound);
		}
	}
}

Mat & CPedestrianDetector::ClusteringAndRectangle(Mat & imgDst, vector<Rect>& vecRectFoundUnfiltered, cv::Scalar color){
	unsigned i, j;
	vector<Rect> vecRectFoundFiltered;

	// Clustering
	for (i = 0; i < vecRectFoundUnfiltered.size(); i++){
		Rect r = vecRectFoundUnfiltered[i];
		for (j = 0; j < vecRectFoundUnfiltered.size(); j++)
			if (j != i && (r & vecRectFoundUnfiltered[j]) == r)
				break;
		if (j == vecRectFoundUnfiltered.size())
			vecRectFoundFiltered.push_back(r);
	}

	// Rectangle
	for (i = 0; i<vecRectFoundFiltered.size(); i++)
	{
		Rect r = vecRectFoundFiltered[i];
		rectangle(imgDst, r.tl(), r.br(), color, 2);
		std::cout << r << std::endl;
	}

	return imgDst;
}

void CPedestrianDetector::SetROI(int _x, int _y, int _width, int _height){
	m_rectROI = cvRect(_x, _y, _width, _height);
	m_nROIwidth = _width;
	m_nROIheight = _height;
}

void CPedestrianDetector::SetMinSize(int _width, int _height){
	m_sizeDetectorS = cvSize(_width, _height);
}

void CPedestrianDetector::SetMaxSize(int _width, int _height){
	m_sizeDetectorL = cvSize(_width, _height);
}

Rect CPedestrianDetector::GetROI() const{
	return m_rectROI;
}

Size CPedestrianDetector::GetMinSize() const{
	return m_sizeDetectorS;
}

Size CPedestrianDetector::GetMaxSize() const{
	return m_sizeDetectorL;
}