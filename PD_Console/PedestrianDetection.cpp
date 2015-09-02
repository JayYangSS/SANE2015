#include "PedestrianDetection.h"
#include <algorithm>
#include <functional>

// ctor
CPedestrianDetection::CPedestrianDetection()
{
	m_fLambda[0] = 0;			m_fLambda[1] = 0;
	m_fLambda[2] = 0;			m_fLambda[3] = 0.11047f;
	m_fLambda[4] = 0.10826f;	m_fLambda[5] = 0.10826f;
	m_fLambda[6] = 0.10826f;	m_fLambda[7] = 0.10826f;
	m_fLambda[8] = 0.10826f;	m_fLambda[9] = 0.10826f;

	m_fShrink = 2;
	m_fPerOct = 8;
	m_nAppox = 8;
	m_sizeSmallestPyramid = Size_<float>(20.5, 50); // smallest scale of the pyramid
	m_sizePad = Size_<int>(6, 8);
	m_nChns = 10;

	m_sizeModelPad = Size_<int>(32, 64);
	m_nStride = 4;
	m_fCascadeThresh = -1.0f;
	m_nTreeNodes = 63;
	m_nTrees = 4096;
	m_nTreeDepth = 0;

	m_nColorFlag = 2;

	m_bFullOrient = false;
	m_fNormConst = 0.005f;
	m_nBinSize = 2;
	m_nOrients = 6;
	m_nSoftBin = 1;

	m_nRadius = 5;
	m_nDownSampling = 1;

	m_fOverlap = 0.65f;

	m_pThrs = new float[m_nTreeNodes*m_nTrees];
	m_pHs = new float[m_nTreeNodes*m_nTrees];
	m_pFids = new unsigned int[m_nTreeNodes*m_nTrees];
	m_pChild = new unsigned int[m_nTreeNodes*m_nTrees];
}

// dtor
CPedestrianDetection::~CPedestrianDetection()
{
	delete[] m_pThrs;
	delete[] m_pHs;
	delete[] m_pFids;
	delete[] m_pChild;
}

// public
bool CPedestrianDetection::LoadClassifier(string strClassifierFile)
{
	FILE *fp;
	fopen_s(&fp, strClassifierFile.c_str(), "rt");
	if (fp == NULL) return false;

	for (int i = 0; i < m_nTreeNodes*m_nTrees; i++)
		fscanf_s(fp, "%f", m_pThrs + i);

	for (int i = 0; i < m_nTreeNodes*m_nTrees; i++)
		fscanf_s(fp, "%f", m_pHs + i);

	for (int i = 0; i < m_nTreeNodes*m_nTrees; i++)
		fscanf_s(fp, "%d", m_pFids + i);

	for (int i = 0; i < m_nTreeNodes*m_nTrees; i++)
		fscanf_s(fp, "%d", m_pChild + i);

	fclose(fp);

	return true;
}

// public
vector<Rect_<int> > CPedestrianDetection::Detect(Mat& imgSrc)
{
	vector<vector<Mat> > chnsFtrs;
	BuildPyramid(imgSrc, chnsFtrs);

	m_vecrectDetectedBB.clear();

	vector<pair<float, Rect_<float> > > vecRawBB;
	vector<pair<float, Rect_<float> > > vecShiftBB;
	for (int i = 0; i < m_nChns; i++)
	{
		vecRawBB.clear();
		AcfDetect(chnsFtrs[i], vecRawBB);

		for (int j = 0; j < (int)vecRawBB.size(); j++)
		{
			vecRawBB[j].second.x /= m_vecfScalesHW[i].second;
			vecRawBB[j].second.y /= m_vecfScalesHW[i].first;
			vecRawBB[j].second.width = m_sizeSmallestPyramid.width / m_vecfScales[i];
			vecRawBB[j].second.height = m_sizeSmallestPyramid.height / m_vecfScales[i];
			vecShiftBB.push_back(vecRawBB[j]);
		}
	}

	m_vecrectDetectedBB = bbNMS(vecShiftBB);

	return m_vecrectDetectedBB;
}

// public
void CPedestrianDetection::DrawBoundingBox(Mat& imgDisp, const Scalar color)
{
	for (int i = 0; i < (int)m_vecrectDetectedBB.size(); i++)
		rectangle(imgDisp, m_vecrectDetectedBB[i], color, 2);
}

//////////////////////////////////////////////////////////////////////////

// private
bool CPedestrianDetection::Comparator(const pair<float, Rect_<float> >& l, const pair<float, Rect_<float> >& r) // 첫번째 값을 기준으로 내림차순 비교하는 comparator
{
	return l.first > r.first;
}

// private
void CPedestrianDetection::GetChild(float *chns1, unsigned int *cids, unsigned int *fids,
	float *thrs, unsigned int offset, unsigned int &k0, unsigned int &k)
{
	float ftr = chns1[cids[fids[k]]];
	k = (ftr < thrs[k]) ? 1 : 2;
	k0 = k += k0 * 2; k += offset;
}

// private
vector<Rect_<int> > CPedestrianDetection::bbNMS(vector<pair<float, Rect_<float> > >& bbs)
{
	stable_sort(bbs.begin(), bbs.end(), Comparator);

	int n = (int)bbs.size();
	int* kp = new int[n];
	for (int i = 0; i < n; i++)
		kp[i] = 1;

	for (int i = 0; i < n; i++)
	{
		if (kp[i] == 0)
			continue;

		for (int j = i + 1; j < n; j++)
		{
			if (kp[j] == 0)
				continue;

			float iw = min(bbs[i].second.br().x, bbs[j].second.br().x) - max(bbs[i].second.x, bbs[j].second.x);
			if (iw <= 0)
				continue;

			float ih = min(bbs[i].second.br().y, bbs[j].second.br().y) - max(bbs[i].second.y, bbs[j].second.y);
			if (ih <= 0)
				continue;

			float o = iw*ih;
			float u = min(bbs[i].second.area(), bbs[j].second.area());

			if (o / u > m_fOverlap)
				kp[j] = 0;
		}
	}

	vector<Rect_<int> > vecRetVal;
	for (int i = 0; i < n; i++)
	{
		if (kp[i] > 0)
			vecRetVal.push_back((Rect_<int>)bbs[i].second);
	}

	delete[] kp;

	return vecRetVal;
}

// private
void CPedestrianDetection::AcfDetect(vector<Mat>& matData, vector<pair<float, Rect_<float> > >& vecRetVal)
{
	// get input
	int cntLen = 0;
	for (int i = 0; i < (int)matData.size(); i++)
		cntLen += matData[i].total();

	float* chns = new float[cntLen];

	int height = matData[0].rows;
	int width = matData[0].cols;
	int cntAdd = 0;
	for (int i = 0; i < (int)matData.size(); i++)
	{
		Mat temp;
		transpose(matData[i], temp);
		float* idx = (float*)temp.data;
		memcpy(chns + cntAdd, idx, sizeof(float)*height*width);
		cntAdd += height*width;
	}

	int nShrink = (int)m_fShrink;
	const int height1 = (int)ceil(float(height*nShrink - m_sizeModelPad.height + 1) / m_nStride);
	const int width1 = (int)ceil(float(width*nShrink - m_sizeModelPad.width + 1) / m_nStride);

	// construct cids array
	int nFtrs = m_sizeModelPad.height / nShrink*m_sizeModelPad.width / nShrink*m_nChns;
	unsigned int *cids = new unsigned int[nFtrs];
	int m = 0;
	for (int z = 0; z < m_nChns; z++)
		for (int c = 0; c < m_sizeModelPad.width / nShrink; c++)
			for (int r = 0; r < m_sizeModelPad.height / nShrink; r++)
				cids[m++] = z*height*width + c*height + r;

	// apply classifier to each patch
	vector<int> rs, cs; vector<float> hs1;
	for (int c = 0; c < width1; c++) for (int r = 0; r < height1; r++) {
		float h = 0;
		float *chns1 = chns + (r*m_nStride / nShrink) + (c*m_nStride / nShrink)*height;
		if (m_nTreeDepth == 1) {
			// specialized case for treeDepth==1
			for (int t = 0; t < m_nTrees; t++) {
				unsigned int offset = t*m_nTreeNodes, k = offset, k0 = 0;
				GetChild(chns1, cids, m_pFids, m_pThrs, offset, k0, k);
				h += m_pHs[k]; if (h <= m_fCascadeThresh) break;
			}
		}
		else if (m_nTreeDepth == 2) {
			// specialized case for treeDepth==2
			for (int t = 0; t < m_nTrees; t++) {
				unsigned int offset = t*m_nTreeNodes, k = offset, k0 = 0;
				GetChild(chns1, cids, m_pFids, m_pThrs, offset, k0, k);
				GetChild(chns1, cids, m_pFids, m_pThrs, offset, k0, k);
				h += m_pHs[k]; if (h <= m_fCascadeThresh) break;
			}
		}
		else if (m_nTreeDepth > 2) {
			// specialized case for treeDepth>2
			for (int t = 0; t < m_nTrees; t++) {
				unsigned int offset = t*m_nTreeNodes, k = offset, k0 = 0;
				for (int i = 0; i < m_nTreeDepth; i++)
					GetChild(chns1, cids, m_pFids, m_pThrs, offset, k0, k);
				h += m_pHs[k]; if (h <= m_fCascadeThresh) break;
			}
		}
		else {
			// general case (variable tree depth)
			for (int t = 0; t < m_nTrees; t++) {
				unsigned int offset = t*m_nTreeNodes, k = offset, k0 = k;
				while (m_pChild[k]) {
					float ftr = chns1[cids[m_pFids[k]]];
					k = (ftr < m_pThrs[k]) ? 1 : 0;
					k0 = k = m_pChild[k0] - k + offset;
				}
				h += m_pHs[k]; if (h <= m_fCascadeThresh) break;
			}
		}
		if (h > m_fCascadeThresh) { cs.push_back(c); rs.push_back(r); hs1.push_back(h); }
	}
	delete[] cids;
	delete[] chns;
	m = cs.size();

	// convert to bbs
	pair<float, Rect_<float> > pair1;
	for (int i = 0; i < m; i++)
	{
		pair1.first = hs1[i];
		pair1.second = Rect_<float>(Rect_<int>(cs[i] * m_nStride, rs[i] * m_nStride, m_sizeModelPad.width, m_sizeModelPad.height));
		vecRetVal.push_back(pair1);
	}
}

// private
void CPedestrianDetection::BuildPyramid(Mat& imgSrc, vector<vector<Mat> >& matData)
{
	// step 1 : RGB to LUV
	Mat imgLUV32f;
	RgbConvertTo(imgSrc, imgLUV32f);

	// step 2 : get pyramid scales
	if ((Size_<int>)m_sizeInputImage != imgSrc.size())
	{
		m_sizeInputImage = imgSrc.size();
		GetPyramidScales();
	}

	// step 3 : channel computation along the real scales
	matData.resize(m_nScales);
	for (int i = 0; i < m_nScales; i += m_nAppox) // "isR" part
	{
		float s = m_vecfScales[i];
		int nDownHeight = (int)(round(m_sizeInputImage.height * s / m_fShrink)*m_fShrink);
		int nDownWidth = (int)(round(m_sizeInputImage.width* s / m_fShrink)*m_fShrink);

		Mat imgOneScale;
		if (i == 0)
			imgOneScale = imgLUV32f.clone();
		else
		{
			//resize(imgLUV16b, imgPyramid, Size(tWidth, tHeight));
			ResampleImg(imgLUV32f, imgOneScale, nDownHeight, nDownWidth);
		}

		if (s == 0.5)
			imgLUV32f = imgOneScale.clone();

		ComputeChannelFeature(imgOneScale, matData[i]);
	}

	// step 4 : channel computation along the approx. scales
	vector<int> vecRef;
	for (int i = 0; i < m_nScales; i += m_nAppox)
	{
		if (i + m_nAppox < m_nScales)
		{
			int j = (int)((i + i + m_nAppox) / 2);
			vecRef.push_back(j);
		}
	}
	vecRef.push_back(m_nScales - 1);

	for (int i = 0, jcnt = 0; i < m_nScales; i++) // "isA" part
	{
		if (i % m_nAppox == 0)
			continue;

		if (i <= vecRef[jcnt])
		{
			int iR = jcnt*m_nAppox;
			int h1 = (int)round(m_sizeInputImage.height*m_vecfScales[i] / m_fShrink);
			int w1 = (int)round(m_sizeInputImage.width*m_vecfScales[i] / m_fShrink);
			for (int j = 0; j < m_nChns; j++)
			{
				float ratio_ = pow(m_vecfScales[i] / m_vecfScales[iR], -m_fLambda[j]);
				Mat imgReszed;
				ResampleImg(matData[iR][j], imgReszed, h1, w1, (float)ratio_);
				matData[i].push_back(imgReszed);
			}
		}
		else
		{
			jcnt++;
			i--;
		}
	}

	// step 5 : smooth and padding channels
	SmoothAndPadChannels(matData);
}

// private
void CPedestrianDetection::RgbConvertTo(Mat& imgSrc, Mat& imgDst)
{
	//cvtColor(imgSrc, imgDst16b, CV_BGR2Luv);

	int n = imgSrc.rows * imgSrc.cols;
	int d = imgSrc.channels();

	unsigned char* temp = new unsigned char[n*d];
	int cnt = 0;
	for (int k = d - 1; k >= 0; k--)
		for (int j = 0; j < imgSrc.cols; j++)
			for (int i = 0; i < imgSrc.rows; i++)
				temp[cnt++] = imgSrc.data[(i*imgSrc.cols + j) * d + k];

	void *J = (void*)rgbConvert(temp, n, d, m_nColorFlag, 1.0f / 255);

	imgDst = Mat::zeros(imgSrc.rows, imgSrc.cols, CV_32FC3);

	float* temp2 = (float*)imgDst.data;
	cnt = 0;
	for (int k = 0; k < d; k++)
		for (int j = 0; j < imgDst.cols; j++)
			for (int i = 0; i < imgDst.rows; i++)
				temp2[(i*imgDst.cols + j) * d + k] = ((float*)J)[cnt++];

	delete[] temp;
	free(J);
}

// private
void CPedestrianDetection::GetPyramidScales()
{
	// ksizeSmallestPyramid 만큼까지 작아지려면 몇 개의 스케일이 필요한가?
	m_nScales = (int)floor(m_fPerOct*(log2(min(m_sizeInputImage.height / m_sizeSmallestPyramid.height,
		m_sizeInputImage.width / m_sizeSmallestPyramid.width))) + 1);

	m_vecfScales.clear();
	for (int i = 0; i < m_nScales; i++)
	{
		// 2^0, 2^(-1/8), 2^(-2/8), ... 하면서 계속 스케일을 작게 만듬.
		float dScale = pow(2.0f, (float)-i / m_fPerOct);

		if (i == 0)
		{
			m_vecfScales.push_back(dScale);
			continue;
		}

		float s0 = (float)(round(m_sizeInputImage.height*dScale / m_fShrink)*m_fShrink - 0.25*m_fShrink) / m_sizeInputImage.height;
		float s1 = (float)(round(m_sizeInputImage.height*dScale / m_fShrink)*m_fShrink + 0.25*m_fShrink) / m_sizeInputImage.height;
		float ss[101];
		float es0[101];
		float es1[101];
		float maxEs0Es1[101];
		int minIdx = -1;
		float minVal = 9999;
		for (int j = 0; j <= 100; j++)
		{
			ss[j] = (float)j*0.01f*(s1 - s0) + s0;
			es0[j] = m_sizeInputImage.height * ss[j];
			es0[j] = abs(es0[j] - round(es0[j] / m_fShrink)*m_fShrink);
			es1[j] = m_sizeInputImage.width * ss[j];
			es1[j] = abs(es1[j] - round(es1[j] / m_fShrink)*m_fShrink);
			maxEs0Es1[j] = (es0[j] > es1[j]) ? es0[j] : es1[j];
			if (maxEs0Es1[j] < minVal)
			{
				minVal = maxEs0Es1[j];
				minIdx = j;
			}
		}
		m_vecfScales.push_back(ss[minIdx]);
	}

	m_vecfScalesHW.clear();
	for (int i = 0; i < m_nScales - 1; i++)
	{
		if (abs(m_vecfScales[i] - m_vecfScales[i + 1]) < 0.00001f)
		{
			m_vecfScales.erase(m_vecfScales.begin() + i);
			i--;
			continue;
		}

		pair<float, float> scaleHW;
		scaleHW.first = round(m_sizeInputImage.height*m_vecfScales[i] / m_fShrink)*m_fShrink / m_sizeInputImage.height;
		scaleHW.second = round(m_sizeInputImage.width*m_vecfScales[i] / m_fShrink)*m_fShrink / m_sizeInputImage.width;
		m_vecfScalesHW.push_back(scaleHW);
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
				resample((float*)imgIn.data, (float*)imgOut.data, imgSrc.cols, nReWidth, imgSrc.rows, nReHeight, 1, ratioVal);
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
void CPedestrianDetection::ComputeChannelFeature(Mat& imgChn, vector<Mat>& vecimgChns)
{
	int nShrink = (int)m_fShrink;

	// step 0 : crop image
	int h = imgChn.rows;
	int w = imgChn.cols;

	int cr1 = h%nShrink;
	int cr2 = w%nShrink;
	if (cr1 || cr2)
	{
		printf(" ccf 이거 지우면 안될 것 같다!!\n");
		h -= cr1;
		w -= cr2;
		Mat imgCrop = imgChn(Range(0, h), Range(0, w));
		imgChn = imgCrop.clone();
	}

	h /= nShrink;
	w /= nShrink;

	// step 1 : compute color channels
	AddChannel(imgChn, vecimgChns, h, w);

	// step 2 : compute gradient magnitude channel
	Mat imgMag, imgOrient, imgMagNorm;
	CalculateGradMag(imgChn, imgMag, imgOrient);
	NormalizeGradMag(imgMag, imgMagNorm);
	AddChannel(imgMagNorm, vecimgChns, h, w);

	// step 3 : compute gradient histogram channel
	Mat imgHist;
	CalculateGradHist(imgMagNorm, imgOrient, imgHist);
	AddChannel(imgHist, vecimgChns, h, w);
}

// private
void CPedestrianDetection::AddChannel(Mat& imgChn, vector<Mat>& vecimgChns, int h, int w)
{
	int h1 = imgChn.rows;
	int w1 = imgChn.cols;

	vector<Mat> vecimgSrc;
	if (imgChn.channels() > 1)
		split(imgChn, vecimgSrc);
	else
		vecimgSrc.push_back(imgChn);

	for (int i = 0; i < imgChn.channels(); i++)
	{
		Mat imgShrink;
		if (h1 != h || w1 != w)
			ResampleImg(vecimgSrc[i], imgShrink, h, w);
		else
			imgShrink = vecimgSrc[i].clone();

		vecimgChns.push_back(imgShrink.clone());
	}
}

// private
void CPedestrianDetection::CalculateGradMag(Mat& imgSrc, Mat& imgDstMag, Mat& imgDstOrient)
{
	int h = imgSrc.rows;
	int w = imgSrc.cols;
	int d = imgSrc.channels();

	Mat temp = Mat::zeros(w, h, CV_32FC1);
	Mat temp2 = Mat::zeros(w, h, CV_32FC1);

	float* I = new float[h*w*d];
	float* M = (float*)temp.data;
	float* O = (float*)temp2.data;

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

	gradMag(I, M, O, h, w, d, m_bFullOrient);

	transpose(temp, imgDstMag);
	transpose(temp2, imgDstOrient);

	delete[] I;
}

// private
void CPedestrianDetection::ConvTriangle(Mat& imgSrc, Mat& imgDst)
{
	int h = imgSrc.rows;
	int w = imgSrc.cols;
	int d = imgSrc.channels();

	Mat temp1;
	transpose(imgSrc, temp1);
	Mat temp2 = Mat::zeros(w, h, CV_32FC1);

	float* A = (float*)temp1.data;
	float* B = (float*)temp2.data;

	convTri(A, B, h, w, d, m_nRadius, m_nDownSampling);

	transpose(temp2, imgDst);
}

// private
void CPedestrianDetection::NormalizeGradMag(Mat& imgSrc, Mat& imgDst)
{
	Mat imgSmooth;
	ConvTriangle(imgSrc, imgSmooth);

	int h = imgSrc.rows;
	int w = imgSrc.cols;

	Mat temp1, temp2;
	transpose(imgSrc, temp1);
	transpose(imgSmooth, temp2);

	float* M = (float*)temp1.data;
	float* S = (float*)temp2.data;

	gradMagNorm(M, S, h, w, m_fNormConst);

	transpose(temp1, imgDst);
}

// private
void CPedestrianDetection::CalculateGradHist(Mat& imgSrcMag, Mat& imgSrcOrient, Mat& imgDstHist)
{
	int h = imgSrcMag.rows;
	int w = imgSrcMag.cols;

	int h2 = h / m_nBinSize;
	int w2 = w / m_nBinSize;

	float* H = new float[h2 * w2 * m_nOrients];
	memset(H, 0, sizeof(float)*h2 * w2 * m_nOrients);

	Mat temp1, temp2;
	transpose(imgSrcMag, temp1);
	transpose(imgSrcOrient, temp2);

	float* M = (float*)temp1.data;
	float* O = (float*)temp2.data;

	gradHist(M, O, H, h, w, m_nBinSize, m_nOrients, m_nSoftBin, m_bFullOrient);

	Mat imgDstHistPre;
	Mat dummy = Mat::zeros(w2, h2, CV_32FC1);
	vector<Mat> vecimgHist;
	for (int i = 0; i < m_nOrients; i++)
	{
		vecimgHist.push_back(dummy.clone());
		float* idx = (float*)vecimgHist[i].data;
		memcpy(idx, H + h2*w2*i, sizeof(float)*h2*w2);
	}

	merge(vecimgHist, imgDstHistPre);

	transpose(imgDstHistPre, imgDstHist);

	delete[] H;
}

// private
void CPedestrianDetection::SmoothAndPadChannels(vector<vector<Mat> >& matData)
{
	int nShrink = (int)m_fShrink;

	for (int i = 0; i < (int)matData.size(); i++)
	{
		for (int j = 0; j < (int)matData[i].size(); j++)
		{
			int h = matData[i][j].rows;
			int w = matData[i][j].cols;
			int d = matData[i][j].channels();

			Mat temp1;
			transpose(matData[i][j], temp1);
			Mat temp2 = Mat::zeros(w, h, CV_32FC1);

			float* A = (float*)temp1.data;
			float* B = (float*)temp2.data;

			convTri1(A, B, h, w, d, 2.0f, 1);

			Mat temp3 = Mat::zeros(w + m_sizePad.width, h + m_sizePad.height, CV_32FC1);
			float* C = (float*)temp3.data;

			int boundary = j < 3 ? 1 : 0;

			imPad(B, C, h, w, d, m_sizePad.height / nShrink, m_sizePad.height / nShrink,
				m_sizePad.width / nShrink, m_sizePad.width / nShrink, boundary, 0.0f);

			transpose(temp3, matData[i][j]);
		}
	}

}

// private
void CPedestrianDetection::ConvFilterChannels(vector<vector<Mat> >& matData, float* chns)
{
	vector<vector<Mat> > matData40;
	matData40.resize(m_nScales);
	int cnt = 0;
	for (int i = 0; i < m_nScales; i++)
	{
		matData40[i].resize(40);
		for (int j = 0; j < m_nChns * 4; j++)
		{
			int k = j % m_nChns;
			Mat temp;
			filter2D(matData[i][k], temp, -1, m_matFilter[j]);

			Mat temp2;
			ResampleImg(temp, temp2, (temp.rows + 1) / 2, (temp.cols + 1) / 2);

			matData40[i][j] = temp2.clone();
			cnt += temp2.total();
		}
	}

	chns = new float[cnt];
	int cnt2 = 0;
	for (int i = 0; i < m_nScales; i++)
	{
		for (int j = 0; j < m_nChns * 4; j++)
		{
			int h = matData40[i][j].rows;
			int w = matData40[i][j].cols;
			float* idx = (float*)matData40[i][j].data;
			memcpy(chns + cnt2, idx, sizeof(float)*h*w);
			cnt2 += h*w;
		}
	}
}