#include "commons.h"

void acfDetect(float* chns, int height, int width)
{
	const int shrink = 4;
	const int modelHt = 64;
	const int modelWd = 32;
	const int stride = 4;
	const float cascThr = -1.0f;
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

	const int height1 = (int)ceil(float(height*shrink - modelHt + 1) / stride);
	const int width1 = (int)ceil(float(width*shrink - modelWd + 1) / stride);

	float *thrs;// detector.clf.thrs임 (고정값)
	float *hs;// detector.clf.hs임 (고정값)
	unsigned long int *fids;// detector.clf.fids임 (고정값)
	unsigned long int *child;// detector.clf.child임 (고정값)

	// apply classifier to each patch
	vector<int> rs;
	vector<int> cs;
	vector<float> hs1;
	for (int c = 0; c < width1; c++)
	{
		for (int r = 0; r < height1; r++)
		{
			float h = 0;
			float *chns1 = chns + (r*stride / shrink) + (c*stride / shrink)*height; // chns 가 p.data{i} 임. 피라미드가 구성되어있을듯
			// general case (variable tree depth)
			for (int t = 0; t < nTrees; t++)
			{
				unsigned long int offset = t*nTreeNodes;
				unsigned long int k = offset;
				unsigned long int k0 = k;
				while (child[k])
				{
					float ftr = chns1[cids[fids[k]]];
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
}

int main()
{
	VideoCapture vcap("PD_sunny.mp4");
	if (!vcap.isOpened())
		return -1;

	Mat imgInput;
	int delayms = 1;

	int nScales = 0;
	vector<double> scales;
	vector<Size_<double> > scalesHW;
	double nPerOct = 8;
	double nOctUp = 0;
	Size_<double> minDs(20.5, 50);
	Size_<double> sz;
	double shrink = 2;

	bool isFirst = false;

	while (1)
	{
		vcap >> imgInput;
		if (imgInput.empty()) break;
		//////////////////////////////////////////////////////////////////////////
		// 시작

		Mat imgColorLUV;
		cvtColor(imgInput, imgColorLUV, CV_RGB2Luv);

		// getScales
		if (isFirst == false)
		{
			isFirst = true;
			sz = imgInput.size();
			nScales = floor(nPerOct*(nOctUp + log2(min(sz.height / minDs.height, sz.width / minDs.width))) + 1);
			for (int i = 0; i < nScales; i++)
			{
				double temp = pow(2, ((double)-i / nPerOct + nOctUp));
				if (i == 0)
				{
					scales.push_back(temp);
				}
				else
				{
					double s0 = (round(sz.height*temp / shrink)*shrink - 0.25*shrink) / sz.height;
					double s1 = (round(sz.height*temp / shrink)*shrink + 0.25*shrink) / sz.height;
					double ss[101];
					double es0[101];
					double es1[101];
					double maxEs0Es1[101];
					int minIdx = -1;
					double minVal = 9999;
					for (int j = 0; j <= 100; j++)
					{
						ss[j] = (double)j*0.01*(s1 - s0) + s0;
						es0[j] = sz.height * ss[j];
						es0[j] = abs(es0[j] - round(es0[j] / shrink)*shrink);
						es1[j] = sz.width * ss[j];
						es1[j] = abs(es1[j] - round(es1[j] / shrink)*shrink);
						maxEs0Es1[j] = (es0[j] > es1[j]) ? es0[j] : es1[j];
						if (maxEs0Es1[j] < minVal)
						{
							minVal = maxEs0Es1[j];
							minIdx = j;
						}
					}
					scales.push_back(ss[minIdx]);
				}
			}
			for (int i = 0; i < scales.size() - 1; i++)
			{
				if (scales[i] == scales[i + 1])
				{
					scales.erase(scales.begin() + i);
					i--;
					continue;
				}

				Size_<double> temp;
				temp.height = round(sz.height*scales[i] / shrink)*shrink / sz.height;
				temp.width = round(sz.width*scales[i] / shrink)*shrink / sz.width;
				scalesHW.push_back(temp);
			}
		}



		// 끝
		//////////////////////////////////////////////////////////////////////////

		//for (int i = 0; i < P.nScales; i++) // 피라미드 개수 만큼 루프
		{
			acfDetect(NULL, 43, 163); // 파일에서 정보를 불러옴 height width
		}

		imshow("asdf", imgInput);
		int key = waitKey(delayms);
		if (key == 27) break;
		else if (key == 32) delayms = 1 - delayms;
		else if (key == 'f') { delayms = 0; }
	}

	return 0;
}
