#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct SDetector
{
	static struct opts
	{
		static struct pPyramid
		{
			static struct pChns
			{
				double shrink;
				static struct pColor
				{
					double enabled;
					double smooth;
					string colorSpace;
				};
				static struct pGradMag
				{
					double enabled;
					double colorChn;
					double normRad;
					double normConst;
					double full;
				};
				static struct pGradHist
				{
					double enabled;
					double binSize;
					double nOrients;
					double softBin;
					double useHog;
					double clipHog;
				};
				double complete;
			};
			double nPerOct;
			double nOctUp;
			double nApprox;
			double lambdas[3];
			double pad[2];
			double minDs[2];
			double smooth;
			double concat;
			double complete;
		};
		double filters;
		double modelDs[2];
		double modelDsPad[2];
		static struct pNms
		{
			string type;
			double overlap;
			string ovrDnm;
		};
		double stride;
		double cascThr;
		double cascCal;
		double nWeak[4];
		static struct pBoost
		{
			static struct pTree
			{
				double nBins;
				double maxDepth;
				double minWeight;
				double fracFtrs;
				double nThreads;
			};
			double nWeak;
			double discrete;
			double verbose;
		};
		double seed;
		// imreadf = imread function handle
		static struct pLoad
		{
			double squarify[2];
			double lbls;
			double ilbls;
			double hRng[2];
			double vRng[2];
		};
		double nPos;
		double nNeg;
		double nPerNeg;
		double nAccNeg;
		static struct pJitter
		{
			double flip;
		};
		double windsSave;
	};
	static struct clf
	{
		unsigned int fids[63][4096];
		double thrs[63][4096];
		unsigned int child[63][4096];
		double hs[63][4096];
		double weights[63][4096];
		unsigned int depth[63][4096];
		double errs[4096];
		double losses[4096];
		int treeDepth;
	};
	static struct info_color
	{
		string name;
		static struct pChn
		{
			double enabled;
			double smooth;
			string colorSpace;
		};
		double nChns;
		string padWith;
	};
	static struct info_gradMag
	{
		string name;
		static struct pChn
		{
			double enabled;
			double colorChn;
			double normRad;
			double normConst;
			double full;
		};
		double nChns;
	};
	static struct info_gradhist
	{
		string name;
		static struct pChn
		{
			double enabled;
			double binSize;
			double nOrients;
			double softBin;
			double useHog;
			double clipHog;
		};
		double nChns;
	};
};

void LoadDetector(string strFileName, SDetector& detector);
Rect_<int> acfDetect(Mat& img, SDetector detector);
void chnsPyramid(Mat& img, SDetector::opts::pPyramid pyramid);