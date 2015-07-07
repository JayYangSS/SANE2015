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
#include "ConvNet.h"

using namespace cv;
using namespace std;
#define Iteration 300 //반복 횟수
#define TrainImg 2597
#define TestImg 2597
#define Mode 1 // Mode 0 : training, Mode 1 : test
#define Batch 256 // 한번에 training 하는 이미지 수
#define NumClass 20

// Gradient Checking
#define G_CHECKING 0
// Conv2 parameter
#define CONV_FULL 0
#define CONV_SAME 1
#define CONV_VALID 2
// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1
#define POOL_STOCHASTIC 2
// get Key type
#define KEY_CONV 0
#define KEY_POOL 1
#define KEY_DELTA 2
#define KEY_UP_DELTA 3
// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2


#define ATD at<double>
#define elif else if
int NumHiddenNeurons = 256;
int NumHiddenLayers = 2;
int nclasses = NumClass;
int NumConvLayers = 2;


vector<int> KernelSize;
vector<int> KernelAmount;
vector<int> PoolingDim;


int batch=Batch;
int Pooling_Methed = POOL_MAX;
int nonlin = NL_RELU;


vector<Cvl> ConvLayers; //Cvl 이라는 struct를 벡터로
vector<Ntw> HiddenLayers; // Ntw라는 struct를 벡터로 
SMR smr; // softmax Regression struct

//typedef struct ConvKernel{
//	Mat W;
//	double b;
//	Mat Wgrad;
//	double bgrad;
//}ConvK;
//
//
//typedef struct ConvLayer{
//	vector<ConvK> layer;
//	int kernelAmount;
//}Cvl;
//
//
//typedef struct Network{
//	Mat W;
//	Mat b;
//	Mat Wgrad;
//	Mat bgrad;
//}Ntw;
//
//
//typedef struct SoftmaxRegression{
//	Mat Weight;
//	Mat Wgrad;
//	Mat b;
//	Mat bgrad;
//	double cost;
//}SMR;
//

// int to string
string i2str(int num){
	stringstream ss;
	ss<<num;
	string s=ss.str();
	return s;
}


// string to int
int str2i(string str){
	return atoi(str.c_str());
}


void unconcatenateMat(vector<Mat> &src, vector<vector<Mat> > &dst, int vsize){
	for(int i=0; i<src.size() / vsize; i++){
		vector<Mat> tmp;
		for(int j=0; j<vsize; j++){
			tmp.push_back(src[i * vsize + j]);
		}
		dst.push_back(tmp);
	}
}


Mat concatenateMat(vector<vector<Mat> > &vec){


	int subFeatures = vec[0][0].rows * vec[0][0].cols;
	int height = vec[0].size() * subFeatures;
	int width = vec.size();
	Mat res = Mat::zeros(height, width, CV_64FC1);


	for(int i=0; i<vec.size(); i++){
		for(int j=0; j<vec[i].size(); j++){
			Rect roi = Rect(i, j * subFeatures, 1, subFeatures);
			Mat subView = res(roi);
			Mat ptmat = vec[i][j].reshape(0, subFeatures);
			ptmat.copyTo(subView);
		}
	}
	return res;
}


Mat concatenateMat(vector<Mat> &vec, int matcols){
	vector<vector<Mat> > temp;
	unconcatenateMat(vec, temp, vec.size() / matcols);
	return concatenateMat(temp);
}


Mat sigmoid(Mat &M){
	Mat temp;
	exp(-M, temp);
	return 1.0 / (temp + 1.0);
}


Mat dsigmoid(Mat &a){
	Mat res = 1.0 - a;
	res = res.mul(a);
	return res;
}


Mat ReLU(Mat& M){
	Mat res(M);
	for(int i=0; i<M.rows; i++){
		for(int j=0; j<M.cols; j++){
			if(M.ATD(i, j) < 0.0) res.ATD(i, j) = 0.0;
		}
	}
	return res;
}


Mat dReLU(Mat& M){
	Mat res = Mat::zeros(M.rows, M.cols, CV_64FC1);
	for(int i=0; i<M.rows; i++){
		for(int j=0; j<M.cols; j++){
			if(M.ATD(i, j) > 0.0) res.ATD(i, j) = 1.0;
		}
	}
	return res;
}


Mat Tanh(Mat &M){
	Mat res(M);
	for(int i=0; i<res.rows; i++){
		for(int j=0; j<res.cols; j++){
			res.ATD(i, j) = tanh(M.ATD(i, j));
		}
	}
	return res;
}


Mat dTanh(Mat &M){
	Mat res = Mat::ones(M.rows, M.cols, CV_64FC1);
	Mat temp;
	pow(M, 2.0, temp);
	res -= temp;
	return res;
}


Mat nonLinearity(Mat &M){
	if(nonlin == NL_RELU){
		return ReLU(M);
	}elif(nonlin == NL_TANH){
		return Tanh(M);
	}else{
		return sigmoid(M);
	}
}


Mat dnonLinearity(Mat &M){
	if(nonlin == NL_RELU){
		return dReLU(M);
	}elif(nonlin == NL_TANH){
		return dTanh(M);
	}else{
		return dsigmoid(M);
	}
}


// Mimic rot90() in Matlab/GNU Octave.
Mat rot90(Mat &M, int k){
	Mat res;
	if(k == 0) return M;
	elif(k == 1){
		flip(M.t(), res, 0); //flip x-axis
	}else{
		flip(rot90(M, k - 1).t(), res, 0);
	}
	return res;
}


// A Matlab/Octave style 2-d convolution function.
// from http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/
Mat conv2(Mat &img, Mat &kernel, int convtype) {
	Mat dest;
	Mat source = img;
	if(CONV_FULL == convtype) {
		source = Mat();
		int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
		copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
	}
	Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
	int borderMode = BORDER_CONSTANT;
	Mat fkernal;
	flip(kernel, fkernal, -1);
	filter2D(source, dest, img.depth(), fkernal, anchor, 0, borderMode);


	if(CONV_VALID == convtype) {
		dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
			.rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
	}
	return dest;
}



Point findLoc(Mat &prob, int m){
	Mat temp, idx;
	Point res = Point(0, 0);
	prob.reshape(0, 1).copyTo(temp); 
	sortIdx(temp, idx, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
	int i = idx.at<int>(0, m);
	res.x = i % prob.rows;
	res.y = i / prob.rows;
	cout <<"5"<<endl;
	return res;

}


Mat Pooling(Mat &M, int pVert, int pHori, int poolingMethod, vector<Point> &locat, bool isTest){
	int remX = M.cols % pHori;
	int remY = M.rows % pVert;
	Mat newM;
	if(remX == 0 && remY == 0) M.copyTo(newM);
	else{
		Rect roi = Rect(remX, remY, M.cols - remX, M.rows - remY);
		M(roi).copyTo(newM);
	}
	Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC1);
	for(int i=0; i<res.rows; i++){
		for(int j=0; j<res.cols; j++){
			Mat temp;
			Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);//// 
			newM(roi).copyTo(temp);
			double val;
			// for Max Pooling
			if(POOL_MAX == poolingMethod){ 
				double minVal; 
				double maxVal; 
				Point minLoc; 
				Point maxLoc;
				minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc );
				val = maxVal;

				locat.push_back(Point(maxLoc.x + j * pHori, maxLoc.y + i * pVert));
			}elif(POOL_MEAN == poolingMethod){
				// Mean Pooling
				val = sum(temp)[0] / (pVert * pHori);
			}elif(POOL_STOCHASTIC == poolingMethod){
				// Stochastic Pooling
				double sumval = sum(temp)[0];
				Mat prob = temp / sumval;
				if(isTest){
					val = sum(prob.mul(temp))[0];
				}else{
					int ran = rand() % (temp.rows * temp.cols);
					Point loc = findLoc(prob, ran);
					val = temp.ATD(loc.y, loc.x);
					locat.push_back(Point(loc.x + j * pHori, loc.y + i * pVert));
				}
			}
			res.ATD(i, j) = val;
		}
	}
	return res;
}


void weightRandomInit(ConvK &convk, int width){


	double epsilon = 0.1;
	convk.W = Mat::ones(width, width, CV_64FC1); //W는 mat width는 5,7
	double *pData; 
	for(int i = 0; i<convk.W.rows; i++){///////////// 0 rand rand~~
		pData = convk.W.ptr<double>(i);////////////// 1 rand rand~~
		for(int j=0; j<convk.W.cols; j++){/////////// 2 rand rand~~  이런식
			pData[j] = randu<double>();        
		}
	}
	convk.W = convk.W * (2 * epsilon) - epsilon;// weightd *2*epsilon - epsilon
	convk.b = 0;
	convk.Wgrad = Mat::zeros(width, width, CV_64FC1);
	convk.bgrad = 0;
}


void weightRandomInit(Ntw &ntw, int inputsize, int hiddensize){//288,200?


	double epsilon = sqrt(6.0) / sqrt((double)hiddensize + inputsize + 1);
	double *pData;
	ntw.W = Mat::ones(hiddensize, inputsize, CV_64FC1);//행 hiddensize, 열 inputsize
	for(int i=0; i<hiddensize; i++){
		pData = ntw.W.ptr<double>(i);
		for(int j=0; j<inputsize; j++){
			pData[j] = randu<double>();// 0~1까지 랜덤number 입력
		}
	}
	ntw.W = ntw.W * (2 * epsilon) - epsilon;
	ntw.b = Mat::zeros(hiddensize, 1, CV_64FC1);
	ntw.Wgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
	ntw.bgrad = Mat::zeros(hiddensize, 1, CV_64FC1);
}


void weightRandomInit(SMR &smr, int nclasses, int nfeatures){
	double epsilon = 0.01;
	smr.Weight = Mat::ones(nclasses, nfeatures, CV_64FC1);
	double *pData; 
	for(int i = 0; i<smr.Weight.rows; i++){
		pData = smr.Weight.ptr<double>(i);
		for(int j=0; j<smr.Weight.cols; j++){
			pData[j] = randu<double>();        
		}
	}
	smr.Weight = smr.Weight * (2 * epsilon) - epsilon;
	smr.b = Mat::zeros(nclasses, 1, CV_64FC1);
	smr.cost = 0.0;
	smr.Wgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
	smr.bgrad = Mat::zeros(nclasses, 1, CV_64FC1);
}


void ConvNetInitPrarms(vector<Cvl> &ConvLayers, vector<Ntw> &HiddenLayers, SMR &smr, int imgDim){


	// Init Conv layers
	for(int i=0; i<NumConvLayers; i++){ //2까지
		Cvl tpcvl;
		for(int j=0; j<KernelAmount[i]; j++){ //4,8
			ConvK tmpConvK; // convkernel
			weightRandomInit(tmpConvK, KernelSize[i]);
			tpcvl.layer.push_back(tmpConvK); //tpcv1에 weight 넣어주기
		}
		tpcvl.kernelAmount = KernelAmount[i]; // 
		ConvLayers.push_back(tpcvl); //convLayers에 weight랑 amount 넣어주기
	}
	std::cout<<"size"<< ConvLayers.size()<<endl;

	// Init Hidden layers
	int outDim = imgDim; //28
	for(int i=0; i<NumConvLayers; i++){ //2까지
		outDim = outDim - KernelSize[i] + 1; // kernel 거치면 사이즈 작아짐
		outDim = outDim / PoolingDim[i]; //나누기2
		//	std::cout <<"kernelsize:"<<KernelSize[i]<<" "<<"outdim:"<<outDim<<endl; //5 12/ 7 3
	}
	int hiddenfeatures = pow(outDim, 2.0); //hidden feature 수는 dimension 제곱 9
	//std::cout <<"hf:"<<hiddenfeatures<<endl;
	for(int i=0; i<ConvLayers.size(); i++){ //2
		hiddenfeatures *= ConvLayers[i].kernelAmount; // * 4,8 = 36, 288
		//	std::cout<< "hiddenfeatures" <<hiddenfeatures<<" "<< ConvLayers[i].kernelAmount<<endl;
	}
	std::cout << "hiddenfeatures:"<<hiddenfeatures<<endl;
	Ntw tpntw; //Network 구조체
	weightRandomInit(tpntw, hiddenfeatures, NumHiddenNeurons);// hiddenfeatures:288 ,NumHiddenNeurons:200
	HiddenLayers.push_back(tpntw);
	for(int i=1; i<NumHiddenLayers; i++){
		Ntw tpntw2;
		weightRandomInit(tpntw2, NumHiddenNeurons, NumHiddenNeurons); // input size랑 hidden size랑 같음 200
		HiddenLayers.push_back(tpntw2);
	}
	// Init Softmax layer
	weightRandomInit(smr, nclasses, NumHiddenNeurons); // 10 classes, 200 neurons
}

void convAndPooling(vector<Mat> &x, vector<Cvl> &CLayers, 
	unordered_map<string, Mat> &map){
		// Conv & Pooling
		int nsamples = x.size();
		for(int m = 0; m < nsamples; m ++){
			string s1 = "X" + i2str(m);
			vector<string> vec;
			for(int cl = 0; cl < CLayers.size(); cl ++){
				int pdim = PoolingDim[cl];
				if(cl == 0){
					for(int k = 0; k < CLayers[cl].kernelAmount; k ++){
						string s2 = s1 + "C0K" + i2str(k);
						Mat temp = rot90(CLayers[cl].layer[k].W, 2);
						Mat tmpconv = conv2(x[m], temp, CONV_VALID);
						tmpconv += CLayers[cl].layer[k].b;
						tmpconv = nonLinearity(tmpconv);
						map[s2] = tmpconv;
						vector<Point> PoolingLoc;
						tmpconv = Pooling(tmpconv, pdim, pdim, Pooling_Methed, PoolingLoc, true);
						string s3 = s2 + "P";
						map[s3] = tmpconv;
						vec.push_back(s3);
					}
				}else{
					vector<string> tmpvec;
					for(int tp = 0; tp < vec.size(); tp ++){
						for(int k = 0; k < CLayers[cl].kernelAmount; k ++){
							string s2 = vec[tp] + "C" + i2str(cl) + "K" + i2str(k);
							Mat temp = rot90(CLayers[cl].layer[k].W, 2);
							Mat tmpconv = conv2(map[vec[tp]], temp, CONV_VALID);
							tmpconv += CLayers[cl].layer[k].b;
							tmpconv = nonLinearity(tmpconv);
							map[s2] = tmpconv;
							vector<Point> PoolingLoc;
							tmpconv = Pooling(tmpconv, pdim, pdim, Pooling_Methed, PoolingLoc, true);
							string s3 = s2 + "P";
							map[s3] = tmpconv;
							tmpvec.push_back(s3);
						}
					}
					swap(vec, tmpvec);
					tmpvec.clear();
				}
			}    
			vec.clear();   
		}
	
}


vector<string> getLayerNKey(vector<Cvl> &CLayers, int nsamples, int n, int keyType){
	vector<string> vecstr;
	for(int i=0; i<nsamples; i++){
		string s1 = "X" + i2str(i);
		vecstr.push_back(s1);
	}
	for(int j=0; j<=n; j++){
		vector<string> tmpvecstr;
		for(int i=0; i<vecstr.size(); i++){
			string s2 = vecstr[i] + "C" + i2str(j);
			for(int k=0; k<CLayers[j].kernelAmount; k++){
				string s3 = s2 + "K" + i2str(k);
				if(j != n){
					s3 += "P";
				}else{
					if(keyType == KEY_POOL){
						s3 += "P";
					}elif(keyType == KEY_DELTA){
						s3 += "PD";
					}elif(keyType == KEY_UP_DELTA){
						s3 += "PUD";
					}
				}
				tmpvecstr.push_back(s3);
			}
		}
		swap(vecstr, tmpvecstr);
		tmpvecstr.clear();
	}
	return vecstr;
}


Mat resultProdict(vector<Mat> &x, vector<Cvl> &CLayers, vector<Ntw> &hLayers, SMR &smr, double lambda){


	int nsamples = x.size();
	// Conv & Pooling
	unordered_map<string, Mat> cpmap;
	convAndPooling(x, CLayers, cpmap);


	vector<Mat> P;
	vector<string> vecstr = getLayerNKey(CLayers, nsamples, CLayers.size() - 1, KEY_POOL);
	for(int i=0; i<vecstr.size(); i++){
		P.push_back(cpmap[vecstr[i]]);
	}
	Mat convolvedX = concatenateMat(P, nsamples);
	P.clear();


	// full connected layers
	vector<Mat> acti;
	acti.push_back(convolvedX);
	//	std::cout<<"num"<<NumHiddenLayers<<endl;
	for(int i=1; i<=NumHiddenLayers; i++){
		Mat tmpacti = hLayers[i - 1].W * acti[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
		acti.push_back(sigmoid(tmpacti));
	}


	Mat M = smr.Weight * acti[acti.size() - 1] + repeat(smr.b, 1, nsamples);
	Mat tmp;
	reduce(M, tmp, 0, CV_REDUCE_MAX);
	M -= repeat(tmp, M.rows, 1);
	Mat p;
	exp(M, p);
	reduce(p, tmp, 0, CV_REDUCE_SUM);
	divide(p, repeat(tmp, p.rows, 1), p);
	log(p, tmp);
	//	std::cout <<"tmp : "<<tmp.colRange(200,300)<<endl;

	Mat result = Mat::ones(1, tmp.cols, CV_64FC1);
	for(int i=0; i<tmp.cols; i++){
		double maxele = tmp.ATD(0, i);
		int which = 0;
		for(int j=1; j<tmp.rows; j++){
			if(tmp.ATD(j, i) > maxele){
				maxele = tmp.ATD(j, i);
				which = j;
			}
		}
		result.ATD(0, i) = which;
	}
	//std::cout<<"here3"<<endl;
	// deconstruct
	cpmap.clear();
	acti.clear();
	return result;
}

void readImg2(vector<Mat> &x, Mat &y, int number_of_images)
{
	y = Mat::zeros(1, number_of_images, CV_64FC1);
	char num[10]={0};
	char iFname[100]={0};
	char iFpath[100]={0};
	int cnt=0;
	for(int ci=0; ci<nclasses; ci++){


		sprintf(num, "%d", ci);
		string inpath = "test\\test";
		strcpy(iFname,inpath.c_str());
		strncat(iFname,num,strlen(num));
		strcpy(iFpath,iFname);
		strcat(iFname,"\\*.*");
		strcat(iFpath,"\\");
		char s_buf[100];
		char **filelist = NULL;
		struct _finddata_t cfile;
		long hFile;
		sprintf_s(s_buf, iFname);
		hFile= (long)_findfirst(s_buf, &cfile);	
		_findnext(hFile, &cfile);

		while(_findnext(hFile, &cfile)==0) //
		{
			cnt++;
			sprintf_s(iFpath, "test\\test%s\\%s", num,cfile.name);
			Mat img=imread(iFpath,CV_LOAD_IMAGE_GRAYSCALE);
			resize(img,img,cvSize(28,28),0,0,INTER_CUBIC);
			img.convertTo(img,CV_64FC1);
			x.push_back(img/255.0);
			y.ATD(0, cnt-1) = (double)(ci);
		}
	}
}

void readxml(vector<Cvl> &CLayers,vector<Ntw> &hLayers, SMR &smr, char *xmlfile)
{


	FileStorage fs(xmlfile, FileStorage::READ);
	FileNode hlayers =fs["Hlayers"];
	FileNodeIterator it = hlayers.begin(), it_end = hlayers.end();
	int i = 0;

	for( ; it != it_end; ++it, i++ ){
		std::cout << "hlayers #" << i << ": ";
		//	std::cout << "Wgrad=" << (int)(*it)["Wgrad"] << ", bgrad=" << (int)(*it)["bgrad"] << ", Weight=" << (int)(*it)["Weight"] <<", b=" << (int)(*it)["b"];

		(*it)["Wgrad"] >> hLayers[i].Wgrad;
		(*it)["bgrad"] >> hLayers[i].bgrad;
		(*it)["Weight"] >>hLayers[i].W;
		(*it)["b"] >> hLayers[i].b;
	}

	i=0;
	FileNode clayers =fs["CLayers"];
	it = clayers.begin(), it_end = clayers.end();
	//for( ; it != it_end; ++it, i++ ){
	for(int i = 0; i < CLayers.size(); i++){

		std::cout << "clayers #" << i << ": ";

		for(int j=0; j<CLayers[i].kernelAmount; j++){

			//	std::cout << "Wgrad=" << (int)(*it)["Wgrad"] << ", bgrad=" << (int)(*it)["bgrad"] << ", Weight=" << (int)(*it)["Weight"] <<", b=" << (int)(*it)["b"];
			(*it)["Wgrad"] >> CLayers[i].layer[j].Wgrad;
			(*it)["bgrad"] >> CLayers[i].layer[j].bgrad;
			(*it)["Weight"] >>CLayers[i].layer[j].W;
			(*it)["b"] >> CLayers[i].layer[j].b;
			++it;
		}
	}




	FileNode Smr =fs["SMR"];
	it = Smr.begin(), it_end = Smr.end();

	(*it)["Wgrad"] >>smr.Wgrad;
	it++;
	(*it)["bgrad"] >>smr.bgrad;
	it++;
	(*it)["Weight"] >>smr.Weight;
	it++;
	(*it)["b"] >>smr.b;
	it++;
	(*it)["Cost"]>>smr.cost;

	fs.release();
}
void initRead(char * Xmlfilename)
{
	/*char Xmlfilename[100] = "test12class.xml";*/
	int imgDim = 32; // image dimension은 trainx vector의 rows, 여기선 28
	//int nsamples = testX.size(); // trainx 의 size는 nsample 수

	KernelSize.push_back(7); //global 함수
	KernelSize.push_back(7); //global 함수
	KernelAmount.push_back(5); //global 함수
	KernelAmount.push_back(8); //global 함수
	PoolingDim.push_back(2); //global 함수
	PoolingDim.push_back(2); //global 함수

	ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgDim);

	readxml(ConvLayers,HiddenLayers,smr,Xmlfilename);  // training data xml 불러오기

}
int cnnMain(vector<Mat>& testX,Mat& result)
{
	//int mode = Mode; // 0 for training, 1 for testing

	long start, end;



	//std::cout <<"read finish"<<endl;
	//std::cout<<"Read testX successfully, including "<<testX[0].cols * testX[0].rows<<" features and "<<testX.size()<<" samples."<<endl;
	//std::cout<<"Read testY successfully, including "<<testY.cols<<" samples"<<endl;


	start = clock();
	result = resultProdict(testX, ConvLayers, HiddenLayers, smr, 3e-3);
	end = clock();
	//Mat err(testY);

	//std::cout <<"err,result cols: "<< err.cols <<  " " << result.cols<<endl;

	//err -= result;
	//int correct = err.cols;
	//for(int i=0; i<err.cols; i++){
	//	if(err.ATD(0, i) != 0) --correct;
	//}
	//std::cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
	//std::cout << "result: "<<result<<endl;
	//std::cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;

	return 0;
}


//
//int main(int argc, char** argv)
//{
//	//int mode = Mode; // 0 for training, 1 for testing
//	char Xmlfilename[100] = "test11class.xml";
//
//	long start, end;
//	
//	vector<Mat> trainX;
//	vector<Mat> testX;
//	Mat trainY, testY;
//  	readImg2(testX, testY, TestImg); //read test image
// 
//	int imgDim = testX[0].rows; // image dimension은 trainx vector의 rows, 여기선 28
//	//int nsamples = testX.size(); // trainx 의 size는 nsample 수
//	vector<Cvl> ConvLayers; //Cvl 이라는 struct를 벡터로
//
//	vector<Ntw> HiddenLayers; // Ntw라는 struct를 벡터로 
//	SMR smr; // softmax Regression struct
//	KernelSize.push_back(5); //global 함수
//	KernelSize.push_back(7); //global 함수
//	KernelAmount.push_back(4); //global 함수
//	KernelAmount.push_back(8); //global 함수
//	PoolingDim.push_back(2); //global 함수
//	PoolingDim.push_back(2); //global 함수
//	
//	ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgDim);
//	
//	readxml(ConvLayers,HiddenLayers,smr,Xmlfilename);  // training data xml 불러오기
// 
//	std::cout <<"read finish"<<endl;
//	std::cout<<"Read testX successfully, including "<<testX[0].cols * testX[0].rows<<" features and "<<testX.size()<<" samples."<<endl;
//	std::cout<<"Read testY successfully, including "<<testY.cols<<" samples"<<endl;
//
//	if(!G_CHECKING){
//		std::cout << "Test use test set"<<endl;
//		start = clock();
//		Mat result = resultProdict(testX, ConvLayers, HiddenLayers, smr, 3e-3);
//		end = clock();
//		Mat err(testY);
//
//		std::cout <<"err,result cols: "<< err.cols <<  " " << result.cols<<endl;
//
//		err -= result;
//		int correct = err.cols;
//		for(int i=0; i<err.cols; i++){
//			if(err.ATD(0, i) != 0) --correct;
//		}
//		std::cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
//	}  
//	std::cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
//	
//	return 0;
//}

 