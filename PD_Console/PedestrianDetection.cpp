#include "PedestrianDetection.h"

// ctor
CPedestrianDetection::CPedestrianDetection()
{
	m_lambdas[0] = 0;		m_lambdas[1] = 0;
	m_lambdas[2] = 0;		m_lambdas[3] = 0.11047;
	m_lambdas[4] = 0.10826;	m_lambdas[5] = 0.10826;
	m_lambdas[6] = 0.10826;	m_lambdas[7] = 0.10826;
	m_lambdas[8] = 0.10826;	m_lambdas[9] = 0.10826;
}

// dtor
CPedestrianDetection::~CPedestrianDetection()
{
}

// public
vector<Rect_<int> > CPedestrianDetection::Detect(Mat& imgSrc)
{
	BuildPyramid(imgSrc);

	return vector<Rect_<int> >();
}

// public
void CPedestrianDetection::DrawBoundingBox(Mat& imgDisp)
{
	DrawBoundingBox(imgDisp, m_vecrectDetectedBB);
}

// public
void CPedestrianDetection::DrawBoundingBox(Mat& imgDisp, vector<Rect_<int> >& rectBB)
{

}

//////////////////////////////////////////////////////////////////////////

// private
void CPedestrianDetection::BuildPyramid(Mat& imgSrc)
{
	// step 1 : RGB to LUV
	Mat imgLUV16b;
	RgbConvertTo(imgSrc, imgLUV16b);

	// step 2 : get pyramid scales
	if ((Size_<int>)m_sizeInputImage != imgSrc.size())
	{
		m_sizeInputImage = imgSrc.size();
		GetPyramidScales();
	}
	vector<vector<Mat> > data;
	data.resize(m_nScales);

	// step 3 : channel computation along the real scales
	const double kdShrink = 2;
	const int knAppox = 8;
	for (int i = 0; i < m_nScales; i += knAppox) // "isR" part
	{
		double s = m_vecdScales[i];
		int nDownHeight = (int)(round(m_sizeInputImage.height * s / kdShrink)*kdShrink);
		int nDownWidth = (int)(round(m_sizeInputImage.width* s / kdShrink)*kdShrink);

		Mat imgOneScale;
		if (i == 0)
			imgOneScale = imgLUV16b.clone();
		else
		{
			//resize(imgLUV16b, imgPyramid, Size(tWidth, tHeight));
			ResampleImg(imgLUV16b, imgOneScale, nDownHeight, nDownWidth);
		}

		if (s == 0.5)
		{
			imgLUV16b = imgOneScale.clone();
		}

		ComputeChannelFeature(imgOneScale, data[i]);
	}

	// step 4 : channel computation along the approx. scales
	vector<int> vecRef;
	for (int i = 0; i < m_nScales; i += knAppox)
	{
		if (i+knAppox < m_nScales)
		{
			int j = (int)((i + i + knAppox) / 2);
			vecRef.push_back(j);
		}
	}
	vecRef.push_back(m_nScales - 1);

	for (int i = 0, jcnt = 0; i < m_nScales; i++) // "isA" part
	{
		if (i % knAppox == 0)
			continue;

		if (i <= vecRef[jcnt])
		{
			int iR = jcnt*knAppox;
			int h1 = (int)round(m_sizeInputImage.height*m_vecdScales[i] / kdShrink);
			int w1 = (int)round(m_sizeInputImage.width*m_vecdScales[i] / kdShrink);
			for (int j = 0; j < 10; j++)
			{
				double ratio_ = pow(m_vecdScales[i] / m_vecdScales[iR], -m_lambdas[j]);
				Mat imgReszed;
				ResampleImg(data[iR][j], imgReszed, h1, w1, (float)ratio_);
				data[i].push_back(imgReszed);
			}
		}
		else
		{
			jcnt++;
			i--;
		}
	}
}

// private
void CPedestrianDetection::RgbConvertTo(Mat& imgSrc, Mat& imgDst16b)
{
	//cvtColor(imgSrc, imgDst16b, CV_BGR2Luv);

	int n = imgSrc.rows * imgSrc.cols;
	int d = imgSrc.channels();
	int flag = 2; // rgb2luv

	unsigned char* temp = new unsigned char[n*d];
	int cnt = 0;
	for (int k = d - 1; k >= 0; k--)
	{
		for (int j = 0; j < imgSrc.cols; j++)
		{
			for (int i = 0; i < imgSrc.rows; i++)
			{
				temp[cnt++] = imgSrc.data[(i*imgSrc.cols + j) * d + k];
			}
		}
	}

	void *J = (void*)rgbConvert(temp, n, d, flag, 1.0f / 255);

	imgDst16b = Mat::zeros(imgSrc.rows, imgSrc.cols, CV_32FC3);

	float* temp2 = (float*)imgDst16b.data;
	cnt = 0;
	for (int k = 0; k < d; k++)
	{
		for (int j = 0; j < imgDst16b.cols; j++)
		{
			for (int i = 0; i < imgDst16b.rows; i++)
			{
				temp2[(i*imgDst16b.cols + j) * d + k] = ((float*)J)[cnt++];
			}
		}
	}

	delete temp;
	free(J);
}

// private
void CPedestrianDetection::GetPyramidScales()
{
	const double kdShrink = 2;
	const double kdPerOct = 8;
	const Size_<double> ksizeSmallestPyramid(20.5, 50); // smallest scale of the pyramid

	// ksizeSmallestPyramid 만큼까지 작아지려면 몇 개의 스케일이 필요한가?
	m_nScales = (int)floor(kdPerOct*(log2(min(m_sizeInputImage.height / ksizeSmallestPyramid.height,
		m_sizeInputImage.width / ksizeSmallestPyramid.width))) + 1);

	m_vecdScales.clear();
	for (int i = 0; i < m_nScales; i++)
	{
		// 2^0, 2^(-1/8), 2^(-2/8), ... 하면서 계속 스케일을 작게 만듬.
		double dScale = pow(2, (double)-i / kdPerOct);

		if (i == 0)
		{
			m_vecdScales.push_back(dScale);
			continue;
		}

		double s0 = (round(m_sizeInputImage.height*dScale / kdShrink)*kdShrink - 0.25*kdShrink) / m_sizeInputImage.height;
		double s1 = (round(m_sizeInputImage.height*dScale / kdShrink)*kdShrink + 0.25*kdShrink) / m_sizeInputImage.height;
		double ss[101];
		double es0[101];
		double es1[101];
		double maxEs0Es1[101];
		int minIdx = -1;
		double minVal = 9999;
		for (int j = 0; j <= 100; j++)
		{
			ss[j] = (double)j*0.01*(s1 - s0) + s0;
			es0[j] = m_sizeInputImage.height * ss[j];
			es0[j] = abs(es0[j] - round(es0[j] / kdShrink)*kdShrink);
			es1[j] = m_sizeInputImage.width * ss[j];
			es1[j] = abs(es1[j] - round(es1[j] / kdShrink)*kdShrink);
			maxEs0Es1[j] = (es0[j] > es1[j]) ? es0[j] : es1[j];
			if (maxEs0Es1[j] < minVal)
			{
				minVal = maxEs0Es1[j];
				minIdx = j;
			}
		}
		m_vecdScales.push_back(ss[minIdx]);
	}

	m_vecdScalesHW.clear();
	for (int i = 0; i < m_nScales - 1; i++)
	{
		if (m_vecdScales[i] == m_vecdScales[i + 1])
		{
			m_vecdScales.erase(m_vecdScales.begin() + i);
			i--;
			continue;
		}

		Size_<double> sizeHW;
		sizeHW.height = round(m_sizeInputImage.height*m_vecdScales[i] / kdShrink)*kdShrink / m_sizeInputImage.height;
		sizeHW.width = round(m_sizeInputImage.width*m_vecdScales[i] / kdShrink)*kdShrink / m_sizeInputImage.width;
		m_vecdScalesHW.push_back(sizeHW);
	}
}

// private
void CPedestrianDetection::ResampleImg(Mat& imgSrc, Mat& imgDst, int nReHeight, int nReWidth, float ratioVal)
{
	if (imgSrc.channels() > 1)
	{
		vector<Mat> vecimgOrigScale;
		vector<Mat> vecimgDownScale;
		Mat imgOut;
		if (imgSrc.type() == CV_32FC3)
			imgOut = Mat::zeros(nReHeight, nReWidth, CV_32FC1);
		else if (imgSrc.type() == CV_8UC3)
			imgOut = Mat::zeros(nReHeight, nReWidth, CV_8UC1);

		split(imgSrc, vecimgOrigScale);
		for (int i = 0; i < imgSrc.channels(); i++)
		{
			Mat imgIn = vecimgOrigScale[i];
			if (imgSrc.type() == CV_32FC3)
				resample((float*)imgIn.data, (float*)imgOut.data, imgSrc.cols, nReWidth, imgSrc.rows, nReHeight, 1, (float)ratioVal);
			else if (imgSrc.type() == CV_8UC3)
				resample((unsigned char*)imgIn.data, (unsigned char*)imgOut.data, imgSrc.cols, nReWidth, imgSrc.rows, nReHeight, 1, (unsigned char)ratioVal);

			vecimgDownScale.push_back(imgOut.clone());
		}

		imgDst = Mat::zeros(nReWidth, nReWidth, imgSrc.type());
		merge(vecimgDownScale, imgDst);
	}
	else
	{
		Mat imgOut;
		if (imgSrc.type() == CV_32FC1)
		{
			imgOut = Mat::zeros(nReHeight, nReWidth, CV_32FC1);
			resample((float*)imgSrc.data, (float*)imgOut.data, imgSrc.cols, nReWidth, imgSrc.rows, nReHeight, 1, (float)ratioVal);
		}
		else if (imgSrc.type() == CV_8UC1)
		{
			imgOut = Mat::zeros(nReHeight, nReWidth, CV_8UC1);
			resample((unsigned char*)imgSrc.data, (unsigned char*)imgOut.data, imgSrc.cols, nReWidth, imgSrc.rows, nReHeight, 1, (unsigned char)ratioVal);
		}
		imgDst = imgOut.clone();
	}
}

// private
void CPedestrianDetection::CalculateGradMag(Mat& imgSrc, Mat& imgDstMag, Mat& imgDstOrient)
{
	int full = 0;
	int h = imgSrc.rows;
	int w = imgSrc.cols;
	int d = imgSrc.channels();

	imgDstMag = Mat::zeros(imgSrc.size(), CV_32FC1);
	imgDstOrient = Mat::zeros(imgSrc.size(), CV_32FC1);

	Mat temp = Mat::zeros(w, h, CV_32FC1);
	Mat temp2 = Mat::zeros(w, h, CV_32FC1);

	float* I = new float[h*w*d];
	float* M = (float*)temp.data;//new float[h*w]; //(float*)imgDstMag.data;
	float* O = (float*)temp2.data; //(float*)imgDstOrient.data;

// 	int cnt = 0;
// 	for (int k = 0; k < d; k++)
// 		for (int j = 0; j < imgSrc.cols; j++)
// 			for (int i = 0; i < imgSrc.rows; i++)
// 				I[cnt++] = ind[(i*imgSrc.cols + j) * d + k];


	Mat imgT;
	transpose(imgSrc, imgT);

	vector<Mat> vecimg;
	split(imgT, vecimg);

	float* idx;
	for (int i = 0; i < d; i++)
	{
		idx = (float*)vecimg[i].data;
		memcpy(I + h*w*i, idx, sizeof(float)*h*w);
	}

	gradMag(I, M, O, h, w, d, full > 0);

	transpose(temp, imgDstMag);
	transpose(temp2, imgDstOrient);

	delete I;
	//delete M;
	//delete O;
}

// private
void CPedestrianDetection::ComputeChannelFeature(Mat& imgPyramid, vector<Mat>& vecimgChns)
{
	const int kdShrink = 2;

	// step 0 : crop image
	int h = imgPyramid.rows;
	int w = imgPyramid.cols;

	int cr1 = h%kdShrink;
	int cr2 = w%kdShrink;
	if (cr1 || cr2)
	{
		printf(" ccf 이거 지우면 안될 것 같다!!\n");
		h -= cr1;
		w -= cr2;
		Mat imgCrop = imgPyramid(Range(0, h), Range(0, w));
		imgPyramid = imgCrop.clone();
	}

	h /= kdShrink;
	w /= kdShrink;

	// step 1 : compute color channels
	AddChannel(imgPyramid, vecimgChns, h, w);

	// step 2 : compute gradient magnitude channel
	Mat imgMag, imgOrient, imgMagSmooth, imgMagNorm;
	CalculateGradMag(imgPyramid, imgMag, imgOrient);
	ConvTriangle(imgMag, imgMagSmooth);
	NormalizeGradMag(imgMag, imgMagSmooth, imgMagNorm);
	AddChannel(imgMagNorm, vecimgChns, h, w);

	// step 3 : compute gradient histogram channel
	Mat imgHist;
	CalculateGradHist(imgMagNorm, imgOrient, imgHist);
	AddChannel(imgHist, vecimgChns, h, w);
}

// private
void CPedestrianDetection::AddChannel(Mat& imgSrc, vector<Mat>& vecimgChns, int h, int w)
{
	int h1 = imgSrc.rows;
	int w1 = imgSrc.cols;

	vector<Mat> vecimgSrc;
	if (imgSrc.channels() > 1)
		split(imgSrc, vecimgSrc);
	else
		vecimgSrc.push_back(imgSrc);

	for (int i = 0; i < imgSrc.channels(); i++)
	{
		Mat imgShrink;
		if (h1 != h || w1 != w)
			ResampleImg(vecimgSrc[i], imgShrink, h, w);
		else
			imgShrink = vecimgSrc[i].clone();

		vecimgChns.push_back(imgShrink.clone());
	}
}

void CPedestrianDetection::ConvTriangle(Mat& imgSrc, Mat& imgDst)
{
	int s = 1;
	int h = imgSrc.rows;
	int w = imgSrc.cols;
	int d = imgSrc.channels();
	int r = 5;

	Mat temp1;
	transpose(imgSrc, temp1);
	Mat temp2 = Mat::zeros(w, h, CV_32FC1);

	float* A = (float*)temp1.data;
	float* B = (float*)temp2.data;

	convTri(A, B, h, w, d, r, s);

	transpose(temp2, imgDst);
}

void CPedestrianDetection::NormalizeGradMag(Mat& imgSrcMag, Mat& imgSrcSmooth, Mat& imgDstMag)
{
	int h = imgSrcMag.rows;
	int w = imgSrcMag.cols;
	float norm_ = 0.005f;

	Mat temp1, temp2;
	transpose(imgSrcMag, temp1);
	transpose(imgSrcSmooth, temp2);

	float* M = (float*)temp1.data;
	float* S = (float*)temp2.data;

	gradMagNorm(M, S, h, w, norm_);

	transpose(temp1, imgDstMag);
}

void CPedestrianDetection::CalculateGradHist(Mat& imgSrcMag, Mat& imgSrcOrient, Mat& imgDstHist)
{
	int binSize = 2;
	int nOrients = 6;
	int softBin = 1;
	int useHog = 0;
	float clipHog = 0.2f;
	bool full = false;

	int h = imgSrcMag.rows;
	int w = imgSrcMag.cols;

	int h2 = h / binSize;
	int w2 = w / binSize;

	float* H = new float[h2 * w2 * nOrients];
	memset(H, 0, sizeof(float)*h2 * w2 * nOrients);

	Mat temp1, temp2;
	transpose(imgSrcMag, temp1);
	transpose(imgSrcOrient, temp2);

	float* M = (float*)temp1.data;
	float* O = (float*)temp2.data;

	gradHist(M, O, H, h, w, binSize, nOrients, softBin, full);

	Mat imgDstHistPre;
	Mat dummy = Mat::zeros(w2, h2, CV_32FC1);
	vector<Mat> vecimgHist;
	for (int i = 0; i < nOrients;i++)
	{
		vecimgHist.push_back(dummy.clone());
		float* idx = (float*)vecimgHist[i].data;
		memcpy(idx, H + h2*w2*i, sizeof(float)*h2*w2);
	}
	Mat ch0 = vecimgHist[0];
	Mat ch1 = vecimgHist[1];
	Mat ch2 = vecimgHist[2];
	Mat ch3 = vecimgHist[3];
	Mat ch4 = vecimgHist[4];
	Mat ch5 = vecimgHist[5];

	merge(vecimgHist, imgDstHistPre);

	transpose(imgDstHistPre, imgDstHist);

	delete H;
}

/*void acfDetect(double* chns, int height, int width)
{
const int shrink = 4;
const int modelHt = 64;
const int modelWd = 32;
const int stride = 4;
const double cascThr = -1.0f;
const int treeDepth = 0;
const int nChns = 40;
const int nTreeNodes = 63;
const int nTrees = 4028;

// construct cids array
const int nFtrs = modelHt / shrink * modelWd / shrink * nChns;
unsigned long int cids[nFtrs];
int m = 0;
for (int z = 0; z < nChns; z++)
{
for (int c = 0; c < modelWd / shrink; c++)
{
for (int r = 0; r < modelHt / shrink; r++)
{
cids[m++] = z*width*height + c*height + r;
}
}
}

const int height1 = (int)ceil(double(height*shrink - modelHt + 1) / stride);
const int width1 = (int)ceil(double(width*shrink - modelWd + 1) / stride);

double *thrs;// detector.clf.thrs임 (고정값)
double *hs;// detector.clf.hs임 (고정값)
unsigned long int *fids;// detector.clf.fids임 (고정값)
unsigned long int *child;// detector.clf.child임 (고정값)

// apply classifier to each patch
vector<int> rs;
vector<int> cs;
vector<double> hs1;
for (int c = 0; c < width1; c++)
{
for (int r = 0; r < height1; r++)
{
double h = 0;
double *chns1 = chns + (r*stride / shrink) + (c*stride / shrink)*height; // chns 가 p.data{i} 임. 피라미드가 구성되어있을듯
// general case (variable tree depth)
for (int t = 0; t < nTrees; t++)
{
unsigned long int offset = t*nTreeNodes;
unsigned long int k = offset;
unsigned long int k0 = k;
while (child[k])
{
double ftr = chns1[cids[fids[k]]];
k = (ftr < thrs[k]) ? 1 : 0;
k0 = k = child[k0] - k + offset;
}
h += hs[k];
if (h <= cascThr) break;
}

if (h>cascThr)
{
cs.push_back(c);
rs.push_back(r);
hs1.push_back(h);
}
}
}
}*/