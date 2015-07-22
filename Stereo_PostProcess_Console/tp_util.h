
#ifndef __TP_UTIL_HPP__
#define __TP_UTIL_HPP__

#include <opencv2/opencv.hpp>

///
/// \brief Fit a line to a 2D point set using RANSAC
/// \param points Input 2D point set
/// \param line Output line. It is a vector of 4 floats
///        (vx, vy, x0, y0) where (vx, vy) is a normalized
///        vector collinear to the line and (x0, y0) is some
///        point on the line.
/// \param iterations Number of iterations
/// \param sigma Parameter use to compute the fitting score
/// \param a_max Max slope of the line, to avoid to detect vertical lines

void fitLineRansac(const std::vector<cv::Point2f> points,
	cv::Vec4f &line,
	int iterations = 1000,
	double sigma = 1.,
	double a_max = 7.);

/// \brief The segmentDisparity function segment the disparity using cv::floodFill
///        and return a 8bit Grey image where each pixel represent the index of the
///        object (0 corresponds to the background)
/// \param disparity Disparity image
/// \param output Result image
/// \return Number of segmented objects
unsigned int segmentDisparity(const cv::Mat & disparity, cv::Mat &output);

#endif // __TP_UTIL_HPP__