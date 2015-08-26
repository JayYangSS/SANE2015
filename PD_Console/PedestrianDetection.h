#pragma once

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <xmmintrin.h>

using namespace std;
using namespace cv;

class CPedestrianDetection
{
public:
	CPedestrianDetection();
	~CPedestrianDetection();

	bool LoadClassifier(string strClassifierFile);
	vector<Rect_<int> > Detect(Mat& imgSrc);
	void DrawBoundingBox(Mat& imgDisp);
	void DrawBoundingBox(Mat& imgDisp, vector<Rect_<int> >& rectBB);

private:
	static bool Comparator(const pair<float, Rect_<float> >& l, const pair<float, Rect_<float> >& r);
	void GetChild(float *chns1, unsigned int *cids, unsigned int *fids, float *thrs, unsigned int offset, unsigned int &k0, unsigned int &k);
	void AcfDetect(vector<Mat>& matData, vector<pair<float, Rect_<float> > >& vecRetVal);
	vector<Rect_<int> > bbNMS(vector<pair<float, Rect_<float> > >& bbs);

	void BuildPyramid(Mat& imgSrc, vector<vector<Mat> >& matData);
	void RgbConvertTo(Mat& imgSrc, Mat& imgDst);
	void GetPyramidScales();
	void ResampleImg(Mat& imgSrc, Mat& imgDst, int nReHeight, int nReWidth, float ratioVal = 1.0f);
	void ComputeChannelFeature(Mat& imgChn, vector<Mat>& vecimgChns);
	void AddChannel(Mat& imgChn, vector<Mat>& vecimgChns, int h, int w);
	void CalculateGradMag(Mat& imgSrc, Mat& imgDstMag, Mat& imgDstOrient);
	void ConvTriangle(Mat& imgSrc, Mat& imgDst);
	void NormalizeGradMag(Mat& imgSrc, Mat& imgDst);
	void CalculateGradHist(Mat& imgSrcMag, Mat& imgSrcOrient, Mat& imgDstHist);
	void SmoothAndPadChannels(vector<vector<Mat> >& matData);
	void ConvFilterChannels(vector<vector<Mat> >& matData, float* chns);

	vector<Rect_<int> > m_vecrectDetectedBB;
	Size_<float> m_sizeInputImage;
	int m_nScales;
	vector<float> m_vecfScales;
	vector<pair<float, float> > m_vecfScalesHW;
	float m_fOverlap;

	// classifier data
	Size_<int> m_sizeModelPad;
	int m_nStride;
	float m_fCascadeThresh;
	int m_nTreeDepth;
	int m_nTreeNodes;
	int m_nTrees;
	float *m_pThrs;
	float *m_pHs;
	unsigned int *m_pFids;
	unsigned int *m_pChild;

	// filter data from file
	vector<Mat> m_matFilter;

	// params
	float m_fShrink;
	float m_fPerOct;
	int m_nAppox;
	int m_nChns;
	Size_<float> m_sizeSmallestPyramid; // smallest scale of the pyramid
	int m_nColorFlag; // rgb2luv
	float m_fLambda[10];
	Size2i m_sizePad;

	// gradient params
	bool m_bFullOrient;
	float m_fNormConst;
	int m_nBinSize;
	int m_nOrients;
	int m_nSoftBin;

	// conv_tri params
	int m_nDownSampling;
	int m_nRadius;

private:

	// wrapper functions if compiling from C/C++
	inline void wrError(const char *errormsg) { throw errormsg; }
	inline void* wrCalloc(size_t num, size_t size) { return calloc(num, size); }
	inline void* wrMalloc(size_t size) { return malloc(size); }
	inline void wrFree(void * ptr) { free(ptr); }

	// platform independent aligned memory allocation (see also alFree)
	void* alMalloc(size_t size, int alignment) {
		const size_t pSize = sizeof(void*), a = alignment - 1;
		void *raw = wrMalloc(size + a + pSize);
		void *aligned = (void*)(((size_t)raw + pSize + a) & ~a);
		*(void**)((size_t)aligned - pSize) = raw;
		return aligned;
	}
	// platform independent alignned memory de-allocation (see also alMalloc)
	void alFree(void* aligned) {
		void* raw = *(void**)((char*)aligned - sizeof(void*));
		wrFree(raw);
	}

	// set, load and store values
	inline __m128 SET(const float &x) { return _mm_set1_ps(x); }
	inline __m128 SET(float x, float y, float z, float w) { return _mm_set_ps(x, y, z, w); }
	inline __m128i SET(const int &x) { return _mm_set1_epi32(x); }
	inline __m128 LD(const float &x) { return _mm_load_ps(&x); }
	inline __m128 LDu(const float &x) { return _mm_loadu_ps(&x); }
	inline __m128 STR(float &x, const __m128 y) { _mm_store_ps(&x, y); return y; }
	inline __m128 STR1(float &x, const __m128 y) { _mm_store_ss(&x, y); return y; }
	inline __m128 STRu(float &x, const __m128 y) { _mm_storeu_ps(&x, y); return y; }
	inline __m128 STR(float &x, const float y) { return STR(x, SET(y)); }

	// arithmetic operators
	inline __m128i ADD(const __m128i x, const __m128i y) { return _mm_add_epi32(x, y); }
	inline __m128 ADD(const __m128 x, const __m128 y) { return _mm_add_ps(x, y); }
	inline __m128 ADD(const __m128 x, const __m128 y, const __m128 z) { return ADD(ADD(x, y), z); }
	inline __m128 ADD(const __m128 a, const __m128 b, const __m128 c, const __m128 &d) { return ADD(ADD(ADD(a, b), c), d); }
	inline __m128 SUB(const __m128 x, const __m128 y) { return _mm_sub_ps(x, y); }
	inline __m128 MUL(const __m128 x, const __m128 y) { return _mm_mul_ps(x, y); }
	inline __m128 MUL(const __m128 x, const float y) { return MUL(x, SET(y)); }
	inline __m128 MUL(const float x, const __m128 y) { return MUL(SET(x), y); }
	inline __m128 INC(__m128 &x, const __m128 y) { return x = ADD(x, y); }
	inline __m128 INC(float &x, const __m128 y) { __m128 t = ADD(LD(x), y); return STR(x, t); }
	inline __m128 DEC(__m128 &x, const __m128 y) { return x = SUB(x, y); }
	inline __m128 DEC(float &x, const __m128 y) { __m128 t = SUB(LD(x), y); return STR(x, t); }
	inline __m128 MIN_(const __m128 x, const __m128 y) { return _mm_min_ps(x, y); }
	inline __m128 RCP(const __m128 x) { return _mm_rcp_ps(x); }
	inline __m128 RCPSQRT(const __m128 x) { return _mm_rsqrt_ps(x); }

	// logical operators
	inline __m128 AND(const __m128 x, const __m128 y) { return _mm_and_ps(x, y); }
	inline __m128i AND(const __m128i x, const __m128i y) { return _mm_and_si128(x, y); }
	inline __m128 ANDNOT(const __m128 x, const __m128 y) { return _mm_andnot_ps(x, y); }
	inline __m128 OR(const __m128 x, const __m128 y) { return _mm_or_ps(x, y); }
	inline __m128 XOR(const __m128 x, const __m128 y) { return _mm_xor_ps(x, y); }

	// comparison operators
	inline __m128 CMPGT(const __m128 x, const __m128 y) { return _mm_cmpgt_ps(x, y); }
	inline __m128 CMPLT(const __m128 x, const __m128 y) { return _mm_cmplt_ps(x, y); }
	inline __m128i CMPGT(const __m128i x, const __m128i y) { return _mm_cmpgt_epi32(x, y); }
	inline __m128i CMPLT(const __m128i x, const __m128i y) { return _mm_cmplt_epi32(x, y); }

	// conversion operators
	inline __m128 CVT(const __m128i x) { return _mm_cvtepi32_ps(x); }
	inline __m128i CVT(const __m128 x) { return _mm_cvttps_epi32(x); }

	template<class oT> oT* rgb2luv_setup(oT z, oT *mr, oT *mg, oT *mb, oT &minu, oT &minv, oT &un, oT &vn);
	template<class iT, class oT> void rgb2luv(iT *I, oT *J, int n, oT nrm);
	template<class iT> void rgb2luv_sse(iT *I, float *J, int n, float nrm);
	template<class iT, class oT> void rgb2hsv(iT *I, oT *J, int n, oT nrm);
	template<class iT, class oT> void rgb2gray(iT *I, oT *J, int n, oT nrm);
	template<> void rgb2gray(double *I, float *J, int n, float nrm);
	template<class iT, class oT> void normalize(iT *I, oT *J, int n, oT nrm);
	template<class iT, class oT> oT* rgbConvert(iT *I, int n, int d, int flag, oT nrm);

	template<class T> void resampleCoef(int ha, int hb, int &n, int *&yas, int *&ybs, T *&wts, int bd[2], int pad = 0);
	template<class T> void resample(T *A, T *B, int ha, int hb, int wa, int wb, int d, T r);

	template<class T> void imPad(T *A, T *B, int h, int w, int d, int pt, int pb, int pl, int pr, int flag, T val);

	// implemented into "gradientMex2.cpp"
	void grad1(float *I, float *Gx, float *Gy, int h, int w, int x);
	void grad2(float *I, float *Gx, float *Gy, int h, int w, int d);
	float* acosTable();
	void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full);
	void gradMagNorm(float *M, float *S, int h, int w, float norm);
	void gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1, int nb, int n, float norm, int nOrients, bool full, bool interpolate);
	void gradHist(float *M, float *O, float *H, int h, int w, int bin, int nOrients, int softBin, bool full);
	float* hogNormMatrix(float *H, int nOrients, int hb, int wb, int bin);
	void hogChannels(float *H, const float *R, const float *N, int hb, int wb, int nOrients, float clip, int type);
	void hog(float *M, float *O, float *H, int h, int w, int binSize, int nOrients, int softBin, bool full, float clip);
	void fhog(float *M, float *O, float *H, int h, int w, int binSize, int nOrients, int softBin, float clip);

	// implemented into "convConst2.cpp"
	void convBoxY(float *I, float *O, int h, int r, int s);
	void convBox(float *I, float *O, int h, int w, int d, int r, int s);
	void conv11Y(float *I, float *O, int h, int side, int s);
	void conv11(float *I, float *O, int h, int w, int d, int side, int s);
	void convTriY(float *I, float *O, int h, int r, int s);
	void convTri(float *I, float *O, int h, int w, int d, int r, int s);
	void convTri1Y(float *I, float *O, int h, float p, int s);
	void convTri1(float *I, float *O, int h, int w, int d, float p, int s);
	void convMaxY(float *I, float *O, float *T, int h, int r);
	void convMax(float *I, float *O, int h, int w, int d, int r);
};

// Constants for rgb2luv conversion and lookup table for y-> l conversion
template<class oT> oT* CPedestrianDetection::rgb2luv_setup(oT z, oT *mr, oT *mg, oT *mb,
	oT &minu, oT &minv, oT &un, oT &vn)
{
	// set constants for conversion
	const oT y0 = (oT)((6.0 / 29)*(6.0 / 29)*(6.0 / 29));
	const oT a = (oT)((29.0 / 3)*(29.0 / 3)*(29.0 / 3));
	un = (oT) 0.197833; vn = (oT) 0.468331;
	mr[0] = (oT) 0.430574*z; mr[1] = (oT) 0.222015*z; mr[2] = (oT) 0.020183*z;
	mg[0] = (oT) 0.341550*z; mg[1] = (oT) 0.706655*z; mg[2] = (oT) 0.129553*z;
	mb[0] = (oT) 0.178325*z; mb[1] = (oT) 0.071330*z; mb[2] = (oT) 0.939180*z;
	oT maxi = (oT) 1.0 / 270; minu = -88 * maxi; minv = -134 * maxi;
	// build (padded) lookup table for y->l conversion assuming y in [0,1]
	static oT lTable[1064]; static bool lInit = false;
	if (lInit) return lTable; oT y, l;
	for (int i = 0; i < 1025; i++) {
		y = (oT)(i / 1024.0);
		l = y > y0 ? 116 * (oT)pow((double)y, 1.0 / 3.0) - 16 : y*a;
		lTable[i] = l*maxi;
	}
	for (int i = 1025; i < 1064; i++) lTable[i] = lTable[i - 1];
	lInit = true; return lTable;
}

// Convert from rgb to luv
template<class iT, class oT> void CPedestrianDetection::rgb2luv(iT *I, oT *J, int n, oT nrm) {
	oT minu, minv, un, vn, mr[3], mg[3], mb[3];
	oT *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
	oT *L = J, *U = L + n, *V = U + n; iT *R = I, *G = R + n, *B = G + n;
	for (int i = 0; i < n; i++) {
		oT r, g, b, x, y, z, l;
		r = (oT)*R++; g = (oT)*G++; b = (oT)*B++;
		x = mr[0] * r + mg[0] * g + mb[0] * b;
		y = mr[1] * r + mg[1] * g + mb[1] * b;
		z = mr[2] * r + mg[2] * g + mb[2] * b;
		l = lTable[(int)(y * 1024)];
		*(L++) = l; z = 1 / (x + 15 * y + 3 * z + (oT)1e-35);
		*(U++) = l * (13 * 4 * x*z - 13 * un) - minu;
		*(V++) = l * (13 * 9 * y*z - 13 * vn) - minv;
	}
}

// Convert from rgb to luv using sse
template<class iT> void CPedestrianDetection::rgb2luv_sse(iT *I, float *J, int n, float nrm) {
	const int k = 256; float R[k], G[k], B[k];

	if ((size_t(R) & 15 || size_t(G) & 15 || size_t(B) & 15 || size_t(I) & 15 || size_t(J) & 15)
		|| n % 4>0) {
		rgb2luv(I, J, n, nrm); return;
	}
	int i = 0, i1, n1; float minu, minv, un, vn, mr[3], mg[3], mb[3];
	float *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
	while (i < n) {
		n1 = i + k; if (n1 > n) n1 = n; float *J1 = J + i; float *R1, *G1, *B1;
		// convert to floats (and load input into cache)
		if (typeid(iT) != typeid(float)) {
			R1 = R; G1 = G; B1 = B; iT *Ri = I + i, *Gi = Ri + n, *Bi = Gi + n;
			for (i1 = 0; i1 < (n1 - i); i1++) {
				R1[i1] = (float)*Ri++; G1[i1] = (float)*Gi++; B1[i1] = (float)*Bi++;
			}
		}
		else { R1 = ((float*)I) + i; G1 = R1 + n; B1 = G1 + n; }
		// compute RGB -> XYZ
		for (int j = 0; j < 3; j++) {
			__m128 _mr, _mg, _mb, *_J = (__m128*) (J1 + j*n);
			__m128 *_R = (__m128*) R1, *_G = (__m128*) G1, *_B = (__m128*) B1;
			_mr = SET(mr[j]); _mg = SET(mg[j]); _mb = SET(mb[j]);
			for (i1 = i; i1 < n1; i1 += 4) *(_J++) = ADD(ADD(MUL(*(_R++), _mr),
				MUL(*(_G++), _mg)), MUL(*(_B++), _mb));
		}
	{ // compute XZY -> LUV (without doing L lookup/normalization)
		__m128 _c15, _c3, _cEps, _c52, _c117, _c1024, _cun, _cvn;
		_c15 = SET(15.0f); _c3 = SET(3.0f); _cEps = SET(1e-35f);
		_c52 = SET(52.0f); _c117 = SET(117.0f), _c1024 = SET(1024.0f);
		_cun = SET(13 * un); _cvn = SET(13 * vn);
		__m128 *_X, *_Y, *_Z, _x, _y, _z;
		_X = (__m128*) J1; _Y = (__m128*) (J1 + n); _Z = (__m128*) (J1 + 2 * n);
		for (i1 = i; i1 < n1; i1 += 4) {
			_x = *_X; _y = *_Y; _z = *_Z;
			_z = RCP(ADD(_x, ADD(_cEps, ADD(MUL(_c15, _y), MUL(_c3, _z)))));
			*(_X++) = MUL(_c1024, _y);
			*(_Y++) = SUB(MUL(MUL(_c52, _x), _z), _cun);
			*(_Z++) = SUB(MUL(MUL(_c117, _y), _z), _cvn);
		}
	}
	{ // perform lookup for L and finalize computation of U and V
		for (i1 = i; i1 < n1; i1++) J[i1] = lTable[(int)J[i1]];
		__m128 *_L, *_U, *_V, _l, _cminu, _cminv;
		_L = (__m128*) J1; _U = (__m128*) (J1 + n); _V = (__m128*) (J1 + 2 * n);
		_cminu = SET(minu); _cminv = SET(minv);
		for (i1 = i; i1 < n1; i1 += 4) {
			_l = *(_L++);
			*_U = SUB(MUL(_l, *_U), _cminu); _U++;
			*_V = SUB(MUL(_l, *_V), _cminv); _V++;
		}
	}
	i = n1;
	}
}

// Convert from rgb to hsv
template<class iT, class oT> void CPedestrianDetection::rgb2hsv(iT *I, oT *J, int n, oT nrm) {
	oT *H = J, *S = H + n, *V = S + n;
	iT *R = I, *G = R + n, *B = G + n;
	for (int i = 0; i < n; i++) {
		const oT r = (oT)*(R++), g = (oT)*(G++), b = (oT)*(B++);
		oT h, s, v, minv, maxv;
		if (r == g && g == b) {
			*(H++) = 0; *(S++) = 0; *(V++) = r*nrm; continue;
		}
		else if (r >= g && r >= b) {
			maxv = r; minv = g < b ? g : b;
			h = (g - b) / (maxv - minv) + 6; if (h >= 6) h -= 6;
		}
		else if (g >= r && g >= b) {
			maxv = g; minv = r < b ? r : b;
			h = (b - r) / (maxv - minv) + 2;
		}
		else {
			maxv = b; minv = r < g ? r : g;
			h = (r - g) / (maxv - minv) + 4;
		}
		h *= (oT)(1 / 6.0); s = 1 - minv / maxv; v = maxv*nrm;
		*(H++) = h; *(S++) = s; *(V++) = v;
	}
}

// Convert from rgb to gray
template<class iT, class oT> void CPedestrianDetection::rgb2gray(iT *I, oT *J, int n, oT nrm) {
	oT *GR = J; iT *R = I, *G = R + n, *B = G + n; int i;
	oT mr = (oT).2989360213*nrm, mg = (oT).5870430745*nrm, mb = (oT).1140209043*nrm;
	for (i = 0; i < n; i++) *(GR++) = (oT)*(R++)*mr + (oT)*(G++)*mg + (oT)*(B++)*mb;
}

// Convert from rgb (double) to gray (float)
template<> void CPedestrianDetection::rgb2gray(double *I, float *J, int n, float nrm) {
	float *GR = J; double *R = I, *G = R + n, *B = G + n; int i;
	double mr = .2989360213*nrm, mg = .5870430745*nrm, mb = .1140209043*nrm;
	for (i = 0; i < n; i++) *(GR++) = (float)(*(R++)*mr + *(G++)*mg + *(B++)*mb);
}

// Copy and normalize only
template<class iT, class oT> void CPedestrianDetection::normalize(iT *I, oT *J, int n, oT nrm) {
	for (int i = 0; i < n; i++) *(J++) = (oT)*(I++)*nrm;
}

// Convert rgb to various colorspaces
template<class iT, class oT>
oT* CPedestrianDetection::rgbConvert(iT *I, int n, int d, int flag, oT nrm) {
	oT *J = (oT*)wrMalloc(n*(flag == 0 ? (d == 1 ? 1 : d / 3) : d)*sizeof(oT));
	int i, n1 = d*(n < 1000 ? n / 10 : 100); oT thr = oT(1.001);
	if (flag > 1 && nrm == 1) for (i = 0; i<n1; i++) if (I[i]>thr)
		wrError("For floats all values in I must be smaller than 1.");
	bool useSse = n % 4 == 0 && typeid(oT) == typeid(float);
	if (flag == 2 && useSse)
		for (i = 0; i < d / 3; i++) rgb2luv_sse(I + i*n * 3, (float*)(J + i*n * 3), n, (float)nrm);
	else if ((flag == 0 && d == 1) || flag == 1) normalize(I, J, n*d, nrm);
	else if (flag == 0) for (i = 0; i < d / 3; i++) rgb2gray(I + i*n * 3, J + i*n * 1, n, nrm);
	else if (flag == 2) for (i = 0; i < d / 3; i++) rgb2luv(I + i*n * 3, J + i*n * 3, n, nrm);
	else if (flag == 3) for (i = 0; i < d / 3; i++) rgb2hsv(I + i*n * 3, J + i*n * 3, n, nrm);
	else wrError("Unknown flag.");
	return J;
}

// compute interpolation values for single column for resapling
template<class T> void CPedestrianDetection::resampleCoef(int ha, int hb, int &n, int *&yas,
	int *&ybs, T *&wts, int bd[2], int pad)
{
	const T s = T(hb) / T(ha), sInv = 1 / s; T wt, wt0 = T(1e-3)*s;
	bool ds = ha > hb; int nMax; bd[0] = bd[1] = 0;
	if (ds) { n = 0; nMax = ha + (pad > 2 ? pad : 2)*hb; }
	else { n = nMax = hb; }
	// initialize memory
	wts = (T*)alMalloc(nMax*sizeof(T), 16);
	yas = (int*)alMalloc(nMax*sizeof(int), 16);
	ybs = (int*)alMalloc(nMax*sizeof(int), 16);
	if (ds) for (int yb = 0; yb < hb; yb++) {
		// create coefficients for downsampling
		T ya0f = yb*sInv, ya1f = ya0f + sInv, W = 0;
		int ya0 = int(ceil(ya0f)), ya1 = int(ya1f), n1 = 0;
		for (int ya = ya0 - 1; ya < ya1 + 1; ya++) {
			wt = s; if (ya == ya0 - 1) wt = (ya0 - ya0f)*s; else if (ya == ya1) wt = (ya1f - ya1)*s;
			if (wt > wt0 && ya >= 0) { ybs[n] = yb; yas[n] = ya; wts[n] = wt; n++; n1++; W += wt; }
		}
		if (W > 1) for (int i = 0; i < n1; i++) wts[n - n1 + i] /= W;
		if (n1 > bd[0]) bd[0] = n1;
		while (n1 < pad) { ybs[n] = yb; yas[n] = yas[n - 1]; wts[n] = 0; n++; n1++; }
	}
	else for (int yb = 0; yb < hb; yb++) {
		// create coefficients for upsampling
		T yaf = (T(.5) + yb)*sInv - T(.5); int ya = (int)floor(yaf);
		wt = 1; if (ya >= 0 && ya < ha - 1) wt = 1 - (yaf - ya);
		if (ya < 0) { ya = 0; bd[0]++; } if (ya >= ha - 1) { ya = ha - 1; bd[1]++; }
		ybs[yb] = yb; yas[yb] = ya; wts[yb] = wt;
	}
}

// resample A using bilinear interpolation and and store result in B
template<class T>
void CPedestrianDetection::resample(T *A, T *B, int ha, int hb, int wa, int wb, int d, T r) {
	int hn, wn, x, x1, y, z, xa, xb, ya; T *A0, *A1, *A2, *A3, *B0, wt, wt1;
	T *C = (T*)alMalloc((ha + 4)*sizeof(T), 16); for (y = ha; y < ha + 4; y++) C[y] = 0;
	bool sse = (typeid(T) == typeid(float)) && !(size_t(A) & 15) && !(size_t(B) & 15);
	// get coefficients for resampling along w and h
	int *xas, *xbs, *yas, *ybs; T *xwts, *ywts; int xbd[2], ybd[2];
	resampleCoef<T>(wa, wb, wn, xas, xbs, xwts, xbd, 0);
	resampleCoef<T>(ha, hb, hn, yas, ybs, ywts, ybd, 4);
	if (wa == 2 * wb) r /= 2; if (wa == 3 * wb) r /= 3; if (wa == 4 * wb) r /= 4;
	r /= T(1 + 1e-6); for (y = 0; y < hn; y++) ywts[y] *= r;
	// resample each channel in turn
	for (z = 0; z < d; z++) for (x = 0; x < wb; x++) {
		if (x == 0) x1 = 0; xa = xas[x1]; xb = xbs[x1]; wt = xwts[x1]; wt1 = 1 - wt; y = 0;
		A0 = A + z*ha*wa + xa*ha; A1 = A0 + ha, A2 = A1 + ha, A3 = A2 + ha; B0 = B + z*hb*wb + xb*hb;
		// variables for SSE (simple casts to float)
		float *Af0, *Af1, *Af2, *Af3, *Bf0, *Cf, *ywtsf, wtf, wt1f;
		Af0 = (float*)A0; Af1 = (float*)A1; Af2 = (float*)A2; Af3 = (float*)A3;
		Bf0 = (float*)B0; Cf = (float*)C;
		ywtsf = (float*)ywts; wtf = (float)wt; wt1f = (float)wt1;
		// resample along x direction (A -> C)
#define FORs(X) if(sse) for(; y<ha-4; y+=4) STR(Cf[y],X);
#define FORr(X) for(; y<ha; y++) C[y] = X;
		if (wa == 2 * wb) {
			FORs(ADD(LDu(Af0[y]), LDu(Af1[y])));
			FORr(A0[y] + A1[y]); x1 += 2;
		}
		else if (wa == 3 * wb) {
			FORs(ADD(LDu(Af0[y]), LDu(Af1[y]), LDu(Af2[y])));
			FORr(A0[y] + A1[y] + A2[y]); x1 += 3;
		}
		else if (wa == 4 * wb) {
			FORs(ADD(LDu(Af0[y]), LDu(Af1[y]), LDu(Af2[y]), LDu(Af3[y])));
			FORr(A0[y] + A1[y] + A2[y] + A3[y]); x1 += 4;
		}
		else if (wa > wb) {
			int m = 1; while (x1 + m < wn && xb == xbs[x1 + m]) m++; float wtsf[4];
			for (int x0 = 0; x0 < (m < 4 ? m : 4); x0++) wtsf[x0] = float(xwts[x1 + x0]);
#define U(x) MUL( LDu(*(Af ## x + y)), SET(wtsf[x]) )
#define V(x) *(A ## x + y) * xwts[x1+x]
			if (m == 1) { FORs(U(0));                     FORr(V(0)); }
			if (m == 2) { FORs(ADD(U(0), U(1)));           FORr(V(0) + V(1)); }
			if (m == 3) { FORs(ADD(U(0), U(1), U(2)));      FORr(V(0) + V(1) + V(2)); }
			if (m >= 4) { FORs(ADD(U(0), U(1), U(2), U(3))); FORr(V(0) + V(1) + V(2) + V(3)); }
#undef U
#undef V
			for (int x0 = 4; x0 < m; x0++) {
				A1 = A0 + x0*ha; wt1 = xwts[x1 + x0]; Af1 = (float*)A1; wt1f = float(wt1); y = 0;
				FORs(ADD(LD(Cf[y]), MUL(LDu(Af1[y]), SET(wt1f)))); FORr(C[y] + A1[y] * wt1);
			}
			x1 += m;
		}
		else {
			bool xBd = x < xbd[0] || x >= wb - xbd[1]; x1++;
			if (xBd) memcpy(C, A0, ha*sizeof(T));
			if (!xBd) FORs(ADD(MUL(LDu(Af0[y]), SET(wtf)), MUL(LDu(Af1[y]), SET(wt1f))));
			if (!xBd) FORr(A0[y] * wt + A1[y] * wt1);
		}
#undef FORs
#undef FORr
		// resample along y direction (B -> C)
		if (ha == hb * 2) {
			T r2 = r / 2; int k = ((~((size_t)B0) + 1) & 15) / 4; y = 0;
			for (; y < k; y++)  B0[y] = (C[2 * y] + C[2 * y + 1])*r2;
			if (sse) for (; y < hb - 4; y += 4) STR(Bf0[y], MUL((float)r2, _mm_shuffle_ps(ADD(
				LDu(Cf[2 * y]), LDu(Cf[2 * y + 1])), ADD(LDu(Cf[2 * y + 4]), LDu(Cf[2 * y + 5])), 136)));
			for (; y < hb; y++) B0[y] = (C[2 * y] + C[2 * y + 1])*r2;
		}
		else if (ha == hb * 3) {
			for (y = 0; y < hb; y++) B0[y] = (C[3 * y] + C[3 * y + 1] + C[3 * y + 2])*(r / 3);
		}
		else if (ha == hb * 4) {
			for (y = 0; y < hb; y++) B0[y] = (C[4 * y] + C[4 * y + 1] + C[4 * y + 2] + C[4 * y + 3])*(r / 4);
		}
		else if (ha > hb) {
			y = 0;
			//if( sse && ybd[0]<=4 ) for(; y<hb; y++) // Requires SSE4
			//  STR1(Bf0[y],_mm_dp_ps(LDu(Cf[yas[y*4]]),LDu(ywtsf[y*4]),0xF1));
#define U(o) C[ya+o]*ywts[y*4+o]
			if (ybd[0] == 2) for (; y < hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1); }
			if (ybd[0] == 3) for (; y < hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1) + U(2); }
			if (ybd[0] == 4) for (; y < hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1) + U(2) + U(3); }
			if (ybd[0] > 4)  for (; y < hn; y++) { B0[ybs[y]] += C[yas[y]] * ywts[y]; }
#undef U
		}
		else {
			for (y = 0; y < ybd[0]; y++) B0[y] = C[yas[y]] * ywts[y];
			for (; y < hb - ybd[1]; y++) B0[y] = C[yas[y]] * ywts[y] + C[yas[y] + 1] * (r - ywts[y]);
			for (; y < hb; y++)        B0[y] = C[yas[y]] * ywts[y];
		}
	}
	alFree(xas); alFree(xbs); alFree(xwts); alFree(C);
	alFree(yas); alFree(ybs); alFree(ywts);
}

// pad A by [pt,pb,pl,pr] and store result in B
template<class T> void CPedestrianDetection::imPad(T *A, T *B, int h, int w, int d, int pt, int pb,
	int pl, int pr, int flag, T val)
{
	int h1 = h + pt, hb = h1 + pb, w1 = w + pl, wb = w1 + pr, x, y, z, mPad;
	int ct = 0, cb = 0, cl = 0, cr = 0;
	if (pt<0) { ct = -pt; pt = 0; } if (pb < 0) { h1 += pb; cb = -pb; pb = 0; }
	if (pl < 0) { cl = -pl; pl = 0; } if (pr < 0) { w1 += pr; cr = -pr; pr = 0; }
	int *xs, *ys; x = pr > pl ? pr : pl; y = pt > pb ? pt : pb; mPad = x > y ? x : y;
	bool useLookup = ((flag == 2 || flag == 3) && (mPad>h || mPad>w))
		|| (flag == 3 && (ct || cb || cl || cr));
	// helper macro for padding
#define PAD(XL,XM,XR,YT,YM,YB) \
  for(x=0;  x<pl; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XL+cl)*h+YT+ct]; \
  for(x=0;  x<pl; x++) for(y=pt; y<h1; y++) B[x*hb+y]=A[(XL+cl)*h+YM+ct]; \
  for(x=0;  x<pl; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XL+cl)*h+YB-cb]; \
  for(x=pl; x<w1; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XM+cl)*h+YT+ct]; \
  for(x=pl; x<w1; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XM+cl)*h+YB-cb]; \
  for(x=w1; x<wb; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XR-cr)*h+YT+ct]; \
  for(x=w1; x<wb; x++) for(y=pt; y<h1; y++) B[x*hb+y]=A[(XR-cr)*h+YM+ct]; \
  for(x=w1; x<wb; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XR-cr)*h+YB-cb];
	// build lookup table for xs and ys if necessary
	if (useLookup) {
		xs = (int*)wrMalloc(wb*sizeof(int)); int h2 = (pt + 1) * 2 * h;
		ys = (int*)wrMalloc(hb*sizeof(int)); int w2 = (pl + 1) * 2 * w;
		if (flag == 2) {
			for (x = 0; x < wb; x++) { z = (x - pl + w2) % (w * 2); xs[x] = z < w ? z : w * 2 - z - 1; }
			for (y = 0; y < hb; y++) { z = (y - pt + h2) % (h * 2); ys[y] = z < h ? z : h * 2 - z - 1; }
		}
		else if (flag == 3) {
			for (x = 0; x < wb; x++) xs[x] = (x - pl + w2) % w;
			for (y = 0; y < hb; y++) ys[y] = (y - pt + h2) % h;
		}
	}
	// pad by appropriate value
	for (z = 0; z < d; z++) {
		// copy over A to relevant region in B
		for (x = 0; x < w - cr - cl; x++)
			memcpy(B + (x + pl)*hb + pt, A + (x + cl)*h + ct, sizeof(T)*(h - ct - cb));
		// set boundaries of B to appropriate values
		if (flag == 0 && val != 0) { // "constant"
			for (x = 0; x < pl; x++) for (y = 0; y < hb; y++) B[x*hb + y] = val;
			for (x = pl; x < w1; x++) for (y = 0; y < pt; y++) B[x*hb + y] = val;
			for (x = pl; x < w1; x++) for (y = h1; y < hb; y++) B[x*hb + y] = val;
			for (x = w1; x < wb; x++) for (y = 0; y < hb; y++) B[x*hb + y] = val;
		}
		else if (useLookup) { // "lookup"
			PAD(xs[x], xs[x], xs[x], ys[y], ys[y], ys[y]);
		}
		else if (flag == 1) {  // "replicate"
			PAD(0, x - pl, w - 1, 0, y - pt, h - 1);
		}
		else if (flag == 2) { // "symmetric"
			PAD(pl - x - 1, x - pl, w + w1 - 1 - x, pt - y - 1, y - pt, h + h1 - 1 - y);
		}
		else if (flag == 3) { // "circular"
			PAD(x - pl + w, x - pl, x - pl - w, y - pt + h, y - pt, y - pt - h);
		}
		A += h*w;  B += hb*wb;
	}
	if (useLookup) { wrFree(xs); wrFree(ys); }
#undef PAD
}