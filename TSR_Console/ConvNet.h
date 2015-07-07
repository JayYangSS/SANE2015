// ConvNet.cpp
// Version 2.0
//
// Author: Eric Yuan
// Blog: http://eric-yuan.me
// You are FREE to use the following code for ANY purpose.
//
// A Convolutional Neural Networks hand writing classifier.
// You can set the amount of Conv Layers and Full Connected Layers.

// Output layer is softmax regression
//
// To run this code, you should have OpenCV in your computer.
// Have fun with it ^v^

// I'm using mac os so if you're using other OS, just change 
// these "#include"s into your style. Make sure you included
// OpenCV stuff, math.h, f/io/s stream, and unordered_map.

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <unordered_map>
#include <time.h>
#include <cv.h>
#include <highgui.h>
#include <io.h>

using namespace cv;
using namespace std;

//#define Iteration 300 //반복 횟수
//#define TrainImg 2597
//#define TestImg 2597
//#define Mode 1 // Mode 0 : training, Mode 1 : test
//#define Batch 300 // 한번에 training 하는 이미지 수
//#define NumClass 12
//
//// Gradient Checking
//#define G_CHECKING 0
//// Conv2 parameter
//#define CONV_FULL 0
//#define CONV_SAME 1
//#define CONV_VALID 2
//// Pooling methods
//#define POOL_MAX 0
//#define POOL_MEAN 1
//#define POOL_STOCHASTIC 2
//// get Key type
//#define KEY_CONV 0
//#define KEY_POOL 1
//#define KEY_DELTA 2
//#define KEY_UP_DELTA 3
//// non-linearity
//#define NL_SIGMOID 0
//#define NL_TANH 1
//#define NL_RELU 2
//
//
//#define ATD at<double>
//#define elif else if
//int NumHiddenNeurons = 256;
//int NumHiddenLayers = 2;
//int nclasses = NumClass;
//int NumConvLayers = 2;
//
//
//vector<int> KernelSize;
//vector<int> KernelAmount;
//vector<int> PoolingDim;
//
//
//int batch=Batch;
//int Pooling_Methed = POOL_MAX;
//int nonlin = NL_RELU;
//

typedef struct ConvKernel{
	Mat W;
	double b;
	Mat Wgrad;
	double bgrad;
}ConvK;


typedef struct ConvLayer{
	vector<ConvK> layer;
	int kernelAmount;
}Cvl;


typedef struct Network{
	Mat W;
	Mat b;
	Mat Wgrad;
	Mat bgrad;
}Ntw;


typedef struct SoftmaxRegression{
	Mat Weight;
	Mat Wgrad;
	Mat b;
	Mat bgrad;
	double cost;
}SMR;
 //int to string
string i2str(int);
// string to int
int str2i(string);
void unconcatenateMat(vector<Mat> &, vector<vector<Mat> > &, int );
Mat concatenateMat(vector<vector<Mat> > &);
Mat concatenateMat(vector<Mat> &, int );
Mat sigmoid(Mat &);
Mat dsigmoid(Mat &);
Mat ReLU(Mat& );
Mat dReLU(Mat& );
Mat Tanh(Mat &);
Mat dTanh(Mat &);
Mat nonLinearity(Mat &);
Mat dnonLinearity(Mat &);
// Mimic rot90() in Matlab/GNU Octave.
Mat rot90(Mat &, int );
// A Matlab/Octave style 2-d convolution function.
// from http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/
Mat conv2(Mat &, Mat &, int );
Point findLoc(Mat &, int );
Mat Pooling(Mat &, int , int , int , vector<Point> &, bool );
void weightRandomInit(ConvK &, int );
void weightRandomInit(Ntw &, int , int );
void weightRandomInit(SMR &, int , int );
void ConvNetInitPrarms(vector<Cvl> &, vector<Ntw> &, SMR &, int );
void convAndPooling(vector<Mat> &, vector<Cvl> &, unordered_map<string, Mat> &);
vector<string> getLayerNKey(vector<Cvl> &, int , int , int );
Mat resultProdict(vector<Mat> &, vector<Cvl> &, vector<Ntw> &, SMR &, double );
void readImg2(vector<Mat> &, Mat &, int );
void readxml(vector<Cvl> &,vector<Ntw> &, SMR &, char *);
void initRead(char * );
int cnnMain(vector<Mat>&, Mat&);

//#endif