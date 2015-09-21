#ifndef PEDESTRIANDETECTOR_H
#define PEDESTRIANDETECTOR_H

#include <cv.h>

using std::vector;
using cv::Rect;
using cv::Mat;
using cv::Size;
using cv::CascadeClassifier;

class CPedestrianDetector{
public:
	CPedestrianDetector();
	CPedestrianDetector(int, int, int, int);

	void PedestrianDetectorHaar(const Mat &, CascadeClassifier &, vector<Rect> &, float = 1.05f, int = 3);
	void PedestrianDetectorHOG(const Mat &, CascadeClassifier &, vector<Rect> &, float = 1.05f, int = 3);
	void PedestrianDetectorHaarHOG(const Mat &, CascadeClassifier &, CascadeClassifier &, vector<Rect> &, float = 1.05f, int = 3, float = 1.025f, int = 2);

	Mat & ClusteringAndRectangle(Mat &, vector<Rect> &, cv::Scalar);

	void SetROI( int, int, int, int );
	void SetMinSize(int, int);
	void SetMaxSize( int, int );
	Rect GetROI() const;
	Size GetMinSize() const;
	Size GetMaxSize() const;

	unsigned int nFrame;

private:
	int m_nROIwidth;
	int m_nROIheight;
	Rect m_rectROI;
	Size m_sizeDetectorS;
	Size m_sizeDetectorL;
};

#endif