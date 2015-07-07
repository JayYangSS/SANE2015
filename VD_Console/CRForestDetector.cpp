/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>


using namespace std;

void CRForestDetector::detectColor(IplImage *img, vector<IplImage* >& imgDetect, std::vector<float>& ratios) {

	// extract features
	vector<IplImage*> vImg;
	CRPatch::extractFeatureChannels(img, vImg);

	// reset output image
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cvSetZero( imgDetect[c] );

	// get pointers to feature channels
	int stepImg;
	uchar** ptFCh     = new uchar*[vImg.size()];
	uchar** ptFCh_row = new uchar*[vImg.size()];
	for(unsigned int c=0; c<vImg.size(); ++c) {
		cvGetRawData( vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);

	// get pointer to output image
	int stepDet;
	float** ptDet = new float*[imgDetect.size()];
	for(unsigned int c=0; c<imgDetect.size(); ++c)
		cvGetRawData( imgDetect[c], (uchar**)&(ptDet[c]), &stepDet);
	stepDet /= sizeof(ptDet[0][0]);

	int xoffset = width/2;
	int yoffset = height/2;
	
	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset; 

	for(y=0; y<img->height-height; ++y, ++cy) {
		// Get start of row
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset; 
		
		for(x=0; x<img->width-width; ++x, ++cx) {					

			// regression for a single patch
			vector<const LeafNode*> result;
			crForest->regression(result, ptFCh_row, stepImg);
			
			// vote for all trees (leafs) 
			for(vector<const LeafNode*>::const_iterator itL = result.begin(); itL!=result.end(); ++itL) {

				// To speed up the voting, one can vote only for patches 
			        // with a probability for foreground > 0.5
			        // 
				 if((*itL)->pfg>0.5) {

					// voting weight for leaf 
					float w = (*itL)->pfg / float( (*itL)->vCenter.size() * result.size() );

					// vote for all points stored in the leaf
					for(vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it!=(*itL)->vCenter.end(); ++it) {

						for(int c=0; c<(int)imgDetect.size(); ++c) {
						  int x = int(cx - (*it)[0].x * ratios[c] + 0.5);
						  int y = cy-(*it)[0].y;
						  if(y>=0 && y<imgDetect[c]->height && x>=0 && x<imgDetect[c]->width) {
						    *(ptDet[c]+x+y*stepDet) += w;
						  }
						}
					}

				  } // end if

			}

			// increase pointer - x
			for(unsigned int c=0; c<vImg.size(); ++c)
				++ptFCh_row[c];

		} // end for x

		// increase pointer - y
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh[c] += stepImg;

	} // end for y 	

	// smooth result image
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cvSmooth( imgDetect[c], imgDetect[c], CV_GAUSSIAN, 3);

	// release feature channels
	for(unsigned int c=0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);
	
	delete[] ptFCh;
	delete[] ptFCh_row;
	delete[] ptDet;

}


void CRForestDetector::detectColor_Revised(std::vector<IplImage *>& vImg, IplImage* & imgDetect) {
	// reset output image
	cvSetZero(imgDetect);

	// get pointers to feature channels
	int stepImg=0;
	uchar** ptFCh = new uchar*[vImg.size()];
	uchar** ptFCh_row = new uchar*[vImg.size()];
	for (unsigned int c = 0; c < vImg.size(); ++c) {
		cvGetRawData(vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);

	// get pointer to output image
	int stepDet;
	float* ptDet;
	cvGetRawData(imgDetect, (uchar**)&(ptDet), &stepDet);
	stepDet /= sizeof(ptDet[0]);

	int xoffset = width / 2;
	int yoffset = height / 2;

	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset;

	for (y = 0; y < vImg[0]->height - height; ++y, ++cy) {
		// Get start of row
		for (unsigned int c = 0; c < vImg.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset;

		for (x = 0; x < vImg[0]->width - width; ++x, ++cx) {

			// regression for a single patch
			vector<const LeafNode*> result;
			crForest->regression(result, ptFCh_row, stepImg);

			// vote for all trees (leafs) 

			
			for (vector<const LeafNode*>::const_iterator itL = result.begin(); itL != result.end(); ++itL) {

				// To speed up the voting, one can vote only for patches 
				// with a probability for foreground > 0.5
				// 
				if ((*itL)->pfg > 0.5) {

					// voting weight for leaf 
					float w = (*itL)->pfg / float((*itL)->vCenter.size() * result.size());

					// vote for all points stored in the leaf
					for (vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it != (*itL)->vCenter.end(); ++it) {
						int x = cx - (*it)[0].x;
						int y = cy - (*it)[0].y;
						if (y >= 0 && y < imgDetect->height && x >= 0 && x < imgDetect->width)
							*(ptDet + x + y*stepDet) += w;
					}
				} // end if
			}

			// increase pointer - x
			for (unsigned int c = 0; c < vImg.size(); ++c)
				++ptFCh_row[c];

		} // end for x

		// increase pointer - y
		for (unsigned int c = 0; c < vImg.size(); ++c)
			ptFCh[c] += stepImg;

	} // end for y 

	//smoothing to procedure the Hough image	
	cvSmooth(imgDetect, imgDetect, CV_GAUSSIAN, 3);

	delete[] ptFCh;
	delete[] ptFCh_row;
	//delete[] ptDet;

}

/*C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\libnvvp;;C:\ProgramData\Oracle\Java\javapath;C:\Program Files\Common Files\Microsoft Shared\Windows Live;C:\Program Files (x86)\Common Files\Microsoft Shared\Windows Live;C:\Program Files (x86)\Wizvera\Delfino;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files (x86)\Microsoft SQL Server\100\Tools\Binn\;C:\Program Files\Microsoft SQL Server\100\Tools\Binn\;C:\Program Files\Microsoft SQL Server\100\DTS\Binn\;C:\opencv\build\x86\vc10\bin;C:\Program Files (x86)\OpenNI\Bin;C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\;C:\Program Files\Microsoft SQL Server\110\Tools\Binn\;C:\Program Files (x86)\Microsoft SDKs\TypeScript\1.0\;D:\opencv\build\x86\vc12\bin;D:\VSPS\OpenCV\bin;C:\Program Files (x86)\Windows Live\Shared;C:\Program Files (x86)\Intel\IPP\6.1.2.041\ia32\bin;C:\Program Files\\jdk1.8.0_40\bin;C:\Python27\;C:\Python27\Scripts\;*/

void CRForestDetector::detectColor_Revised_Mat(const vector<Mat>& vImg, Mat& imgDetect){
	//reset the output image
	imgDetect.setTo(Scalar::all(0));

	int cnt = 0;
	int neg_cnt = 0;
	float weight = 0;
	nStrideX =6;
	nStrideY =6;
	//get pointers to feature channels
	int stepImg;
	uchar** ptFch = new uchar*[vImg.size()];
	uchar** ptFch_row = new uchar*[vImg.size()];

	for (unsigned int c = 0; c < vImg.size(); c++)
		ptFch[c] = vImg[c].data;
	stepImg = vImg[0].step / sizeof(ptFch[0][0]);

	//get pointer to output image
	int stepDet;
	float* ptDet;
	ptDet = (float*)imgDetect.data;
	stepDet = imgDetect.step / sizeof(ptDet[0]);

	int xoffset = width / 2;	//width = 12 --> xoffset = 6
	int yoffset = height / 2;

	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset;
	if (vImg[0].rows > height && vImg[0].cols > width){
		for (y = 0; y < vImg[0].rows - height; y += nStrideY, cy += nStrideY) {
			// Get start of row
			for (unsigned int c = 0; c < vImg.size(); ++c)
				ptFch_row[c] = &ptFch[c][0];
			cx = xoffset;

			for (x = 0; x < vImg[0].cols - width; x += nStrideX, cx += nStrideX) {

				// regression for a single patch
				vector<const LeafNode*> result;
				crForest->regression(result, ptFch_row, stepImg);	
				cnt = 1; neg_cnt = 1;

				//vote for all leafs of trees
				for (vector<const LeafNode*>::const_iterator itL = result.begin(); itL != result.end(); itL++){
					// iterator for leaf nodes
					if ((*itL)->pfg > 0.5){
						cnt++;
						float w = (*itL)->pfg / (float)((*itL)->vCenter.size() * result.size());

						//vote for all points stored in the leaf node
						for (vector<vector<CvPoint>>::const_iterator it = (*itL)->vCenter.begin(); it != (*itL)->vCenter.end(); it++){
							int x = cx - (*it)[0].x;
							int y = cy - (*it)[0].y;
							if (y >= 0 && y < imgDetect.rows && x >= 0 && x < imgDetect.cols)
								*(ptDet + x + y*stepDet) += w*nStrideX*nStrideY;	//check
								//cout << "weight  =	" << w << endl;
								//*(ptDet + x + y*stepDet) += w;	//check
						}
					}//end if
					else neg_cnt++;
				}
				for (unsigned int c = 0; c < vImg.size(); c++){
					//++ptFch_row[c];
					ptFch_row[c]+=nStrideX;
				}
			}//end for x loop

			//increase pointer y
			for (unsigned int c = 0; c < vImg.size(); c++)
				ptFch[c] += stepImg*nStrideY;	
		}//end for y loop
	}
	//smoothing to procedure the Hough image	
	GaussianBlur(imgDetect, imgDetect, Size(7, 7), 2.5);
 	//cout << "check" << endl;
}


void CRForestDetector::buildHoughMap(const vector<Mat>& vImg, Mat& imgDetect, Mat& imgCount){
	//reset the output image
	imgDetect.setTo(Scalar::all(0));
	imgCount.setTo(Scalar::all(0));

	Mat imgMultiply(Size(imgDetect.cols,imgDetect.rows), CV_32FC1);
	Mat imgDivide(Size(imgDetect.cols, imgDetect.rows), CV_32FC1);

	int cnt = 0;
	int neg_cnt = 0;
	float weight = 0;
	nStrideX = 4;
	nStrideY = 4;
	double dMax=0;
	double dAverage = 0;

	//get pointers to feature channels
	int stepImg;
	uchar** ptFch = new uchar*[vImg.size()];
	uchar** ptFch_row = new uchar*[vImg.size()];
	for (unsigned int c = 0; c < vImg.size(); c++)
		ptFch[c] = vImg[c].data;
	stepImg = vImg[0].step / sizeof(ptFch[0][0]);

	//get pointer to output image
	int stepDet;
	float* ptDet;
	ptDet = (float*)imgDetect.data;
	stepDet = imgDetect.step / sizeof(ptDet[0]);

	//get pointer to vote count image
	int stepCnt;
	float* ptCnt;
	ptCnt = (float*)imgCount.data;
	stepCnt = imgCount.step / sizeof(ptCnt[0]);

	int xoffset = width / 2;	//width = 12 --> xoffset = 6
	int yoffset = height / 2;

	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset;
	if (vImg[0].rows > height && vImg[0].cols > width){
		for (y = 0; y < vImg[0].rows - height; y += nStrideY, cy += nStrideY) {
			// Get start of row
			for (unsigned int c = 0; c < vImg.size(); ++c)
				ptFch_row[c] = &ptFch[c][0];
			cx = xoffset;

			for (x = 0; x < vImg[0].cols - width; x += nStrideX, cx += nStrideX) {

				// regression for a single patch
				vector<const LeafNode*> result;
				crForest->regression(result, ptFch_row, stepImg);	//ptFch_row는 데이터??

				//vote for all leafs of trees
				for (vector<const LeafNode*>::const_iterator itL = result.begin(); itL != result.end(); itL++){
					// iterator for leaf nodes.. 모든 leaf node에 대해서 for문을 돈다
					if ((*itL)->pfg > 0.5){
						float w = (*itL)->pfg / (float)(/*(*itL)->vCenter.size() **/ result.size());

						//vote for all points stored in the leaf node
						for (vector<vector<CvPoint>>::const_iterator it = (*itL)->vCenter.begin(); it != (*itL)->vCenter.end(); it++){
							int x = cx - (*it)[0].x;
							int y = cy - (*it)[0].y;
							if (y >= 0 && y < imgDetect.rows && x >= 0 && x < imgDetect.cols){
								*(ptDet + x + y*stepDet) += w*nStrideX*nStrideY;	//check
								*(ptCnt + x + y*stepCnt) += 1;	//check
							}
						}
					}//end if
				}
			
				for (unsigned int c = 0; c < vImg.size(); c++){
					//++ptFch_row[c];
					ptFch_row[c] += nStrideX;
				}
			}//end for x loop

			//increase pointer y
			for (unsigned int c = 0; c < vImg.size(); c++)
				ptFch[c] += stepImg*nStrideY;
		}//end for y loop
	}
	//smoothing to procedure the Hough image	
	GaussianBlur(imgDetect, imgDetect, Size(7, 7), 2.5);

	minMaxLoc(imgCount, NULL, &dMax, NULL, NULL);
	//nMaxCount.push_back(dMax);

	dAverage = (sum(imgCount)[0]) / (double)(imgCount.cols*imgCount.rows);
	cout << "average value: " << dAverage<<endl;
	cout << "max value: " << dMax << endl;
	//make histogram??

	//threshold(imgCount, imgCount, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//threshold(imgCount, imgCount, dAverage, 255, CV_THRESH_BINARY);
	//minMaxLoc(imgCount, NULL, &dMax, NULL, NULL);
	//nMaxCount.push_back(dMax);

	//divide(imgDetect, imgCount, imgDivide);
	//multiply(imgDetect, imgCount, imgMultiply);

	cout << "check" << endl;
}

void CRForestDetector::detectPyramid(IplImage *img, vector<vector<IplImage*> >& vImgDetect, std::vector<float>& ratios) {	

	if(img->nChannels==1) {

		std::cerr << "Gray color images are not supported." << std::endl;

	} else { // color

		//cout << "Timer" << endl;
		int tstart = clock();

		for(int i=0; i<int(vImgDetect.size()); ++i) {
			IplImage* cLevel = cvCreateImage( cvSize(vImgDetect[i][0]->width,vImgDetect[i][0]->height) , IPL_DEPTH_8U , 3);				
			cvResize( img, cLevel, CV_INTER_LINEAR );	

			// detection
			detectColor(cLevel,vImgDetect[i],ratios);

			cvReleaseImage(&cLevel);
		}

		cout << "Time " << (double)(clock() - tstart)/CLOCKS_PER_SEC*1000 << " msec" << endl;

	}

}

void CRForestDetector::detectPyramid_Revised(IplImage *img, vector<IplImage*> & vImgDetect, std::vector<float> scales) {

	if (img->nChannels == 1) {
		std::cerr << "Gray color images are not supported." << std::endl;
	}
	else { // color

		//cout << "Timer" << endl;
		int tstart = clock();
		
		//	feature vectors
		vector<IplImage*> vImg, vImgTemp;
		IplImage * imgTemp;
		
		//	calculate feature channels for origin image (scale = 1.0) - 각각 벡터(vImg)에 저장
		CRPatch::extractFeatureChannels(img, vImg);
		vImgTemp.resize(3);
		for (unsigned int s = 0; s < scales.size(); s++){
			CvSize size = cvSize(int(img->width*scales[s] + 0.5), int(img->height*scales[s]));
			
			//이 사이즈에 스케일별 ROI 적용 

			for (unsigned int c = 0; c < vImgTemp.size(); c++){
				vImgTemp[c] = cvCreateImage(size, vImg[c]->depth, vImg[c]->nChannels);
				cvResize(vImg[c], vImgTemp[c]);
			}
			detectColor_Revised(vImgTemp, vImgDetect[s]);
		}
		cout << "Time " << (double)(clock() - tstart) / CLOCKS_PER_SEC * 1000 << " msec" << endl;

	}

}

void CRForestDetector::detectPyramid_Revised2(CRForestDetector& crDetect, std::vector<IplImage *>img, vector<IplImage*> & vImgDetect, std::vector<float> scales, std::vector<CvRect> vecRoi) {

	if (img[0]->nChannels == 1) {
		std::cerr << "Gray color images are not supported." << std::endl;
	}
	else { // color

		//cout << "Timer" << endl;
		int tstart = clock();

		//	feature vectors
		vector<IplImage*> vImg, vImgTemp, vImgTempScale, vImgStart/*, vImgSameSize*/;
		vector<vector<IplImage*>> vImgSameSize/*, vImgStart*/;
		vector<sImgTemp> vecTest/*, vImgStart*/;

		//	calculate feature channels for origin image (scale = 1.0) - 각각 벡터(vImg)에 저장
		CRPatch::extractFeatureChannels(crDetect.imgTotal, vImg);	//feature를 vImg에 저장 (사이즈 3 - intensity, sobel x, sobel y)
		vImgTemp.resize(3); 
		vImgTempScale.resize(3);
		vImgSameSize.resize(3);		//vImgStart.resize(3);

		for (unsigned int c = 0; c < vImg.size(); c++){	//	feature 갯수마다
			vImgTemp = crDetect.CropNScaleRoi(vImg[c], scales);	//	cropped roi 저장   (3개)

			for (unsigned int s = 0; s < vImgTemp.size(); s++){	//각각의 ROI에 scale 적용 - vImgTempScale에 push
			
				CvSize size = cvSize(int(vImgTemp[s]->width*scales[s] + 0.5), int(vImgTemp[s]->height*scales[s] + 0.5));

				vImgTempScale[s] = cvCreateImage(size, vImg[c]->depth, vImg[c]->nChannels);
				cvResize(vImgTemp[s], vImgTempScale[s]);
				printf("%d %d\n", c, s);
				//vImgSameSize.at(c).push_back(vImgTempScale[s]);

				/*vecTest.at(c).nIdx = s;
	vecTest.at(c).imgScaled.push_back(vImgTempScale[s]);*/

			}		
		}

		//for (unsigned int s = 0; vImgTemp.size(); s++){
		//	for (unsigned int c = 0; vImg.size(); c++){

		//		vImgStart.at(s).push_back(vImgSameSize.at(s).at(c));
		//	}
		//}

		for (unsigned int s = 0; vImgSameSize.size(); s++){
			for (unsigned int c = 0; vImg.size(); c++){
				vImgStart.push_back(vImgSameSize.at(c).at(s));
			}
			detectColor_Revised(vImgStart, vImgDetect[s]);
			//detectColor_Revised_Mat();
			/*cvShowImage("vImgDetect", vImgDetect[s]);
			cvWaitKey(0);*/
		}

		cout << "Time " << (double)(clock() - tstart) / CLOCKS_PER_SEC * 1000 << " msec" << endl;

	}

}

void CRForestDetector::detectPyramid_Revised_Mat(CRForestDetector& crDetect, std::vector<Mat>& vec_ROIs, std::vector<float>& scale) {

	if (crDetect.mRoi1.channels() == 1) {
		std::cerr << "Gray color images are not supported." << std::endl;
	}
	else {
		Mat imgTemp, mTmp;
		Mat Temp2;

		//feature vectors
		vector<Mat> vImg, vImgTemp;
		vector<Mat> matTmp;
		vector<vector<Mat>> vec_storeROI;
		vector<Mat> vecTemp;

		vec_storeROI.resize(scale.size());
	
		//calculate feature channels for origin image (scale = 1.0) - 각각 벡터(vImg)에 저장
		CRPatch::extractFeatureChannels_Mat(crDetect.mRoi1, vImg);
	
		vImgTemp.resize(3);
		unsigned int num_scale, num_feature;

		for (num_scale = 0; num_scale < scale.size(); num_scale++){
			vec_storeROI[num_scale].resize(vImg.size());
		}

		for (num_feature = 0; num_feature < vImg.size(); num_feature++){
			vImgTemp = CropNScaleRoi_Reivised_Mat2(vImg[num_feature], scale);

			if (num_feature == 0){
				for (int i = 0; i < scale.size(); i++){
					mTmp = Mat(vImgTemp[i].rows, vImgTemp[i].cols, CV_32FC1);
					vImgDetect.push_back(mTmp);
				}
			}
			
			for (num_scale = 0; num_scale < scale.size(); num_scale++){
				vec_storeROI[num_scale][num_feature] = vImgTemp[num_scale];
					/*imshow("feature", vImgTemp[num_scale]);
					cvWaitKey(0);*/
			}
		}
		
		for (num_scale = 0; num_scale < scale.size(); num_scale++){
			for (num_feature = 0; num_feature < vImg.size(); num_feature++){

				Temp2 = vec_storeROI.at(num_scale).at(num_feature);
				vecTemp.push_back(Temp2);
			}
			Mat imgCount(Size(crDetect.vImgDetect[num_scale].cols, crDetect.vImgDetect[num_scale].rows), CV_32FC1);
			//buildHoughMap(vecTemp, crDetect.vImgDetect[num_scale], imgCount);
			detectColor_Revised_Mat(vecTemp, crDetect.vImgDetect[num_scale]);

			vecTemp.clear();
		}
	}
}


void CRForestDetector::detectInPyramid( std::vector<float>& scale) {

	if (mTmp.channels() == 1) {
		std::cerr << "Gray color images are not supported." << std::endl;
	}
	else {
		Mat imgTemp;
		Mat Temp2;

		//feature vectors
		vector<Mat>  vImgTemp;
		vector<Mat> matTmp;
		vector<vector<Mat>> vec_storeROI;
		vector<Mat> vecTemp;

		vec_storeROI.resize(scale.size());

		//calculate feature channels for origin image (scale = 1.0) - 각각 벡터(vImg)에 저장
		CRPatch::extractFeatureChannels_Mat(mTmp, vec_mFeature);

		vImgTemp.resize(scale.size());
		unsigned int num_scale, num_feature;

		for (num_scale = 0; num_scale < scale.size(); num_scale++){
			vec_storeROI[num_scale].resize(vec_mFeature.size());
		}

		for (num_feature = 0; num_feature < vec_mFeature.size(); num_feature++){
			vImgTemp = CropNROI_scale(vec_mFeature[num_feature], scale);

			if (num_feature == 0){
				for (int i = 0; i < scale.size(); i++){
					imgTemp = Mat(vImgTemp[i].rows, vImgTemp[i].cols, CV_32FC1);
					vImgDetect.push_back(imgTemp);
				}
			}

			for (num_scale = 0; num_scale < scale.size(); num_scale++){
				vec_storeROI[num_scale][num_feature] = vImgTemp[num_scale];
				/*imshow("feature", vImgTemp[num_scale]);
				cvWaitKey(0);*/
			}
		}

		for (num_scale = 0; num_scale < scale.size(); ++num_scale){
			for (num_feature = 0; num_feature < vec_mFeature.size(); ++num_feature){

				Temp2 = vec_storeROI.at(num_scale).at(num_feature);
				vecTemp.push_back(Temp2);
			}
			Mat imgCount(vImgDetect[num_scale].size(), CV_32FC1);
			//buildHoughMap(vecTemp, vImgDetect[num_scale], imgCount);
			detectColor_Revised_Mat(vecTemp, vImgDetect[num_scale]);

			vecTemp.clear();
		}
	}
}


void CRForestDetector::detectMultiROI(std::vector<Rect>& vecValid) {
	vector<vector<Mat>> vec_rectFeatureROI;
	float fRatio=0;
	Size size_Resize;
	vector<Mat> vec_mCopyFeature;

	sLocalROI.resize(vecValid.size());

	//copy features
	for (int i = 0; i < vec_mFeature.size(); i++){
		vec_mFeature[i].copyTo(vec_mCopyFeature[i]);
	}

	//crop ROIs and store
	for (int i = 0; i < vecValid.size(); i++){
		for (int j = 0; j < vec_mCopyFeature.size(); j++){
			vec_rectFeatureROI[i].push_back(vec_mCopyFeature[j](vecValid[i]));
		}
	}
	//resize feature ROIs
	for (int i = 0; i < vec_rectFeatureROI.size(); i++){
		fRatio = 1 / vecValid[i].width / sizeTrainImg.width;
		sLocalROI[i].scale = fRatio;
		size_Resize = Size(vecValid[i].width * (float)(fRatio), vecValid[i].height * (float)(fRatio));
		for (int j = 0; j < vec_rectFeatureROI[i].size(); j++){
			resize(vec_rectFeatureROI[i][j], vec_rectFeatureROI[i][j], size_Resize);
		}
	}

	//generate blank images for hough map
	//generate HoughMap
	for (int j = 0; j < vec_rectFeatureROI.size(); j++){
		sLocalROI[j].mROI = Mat(vec_rectFeatureROI[j][0].cols, vec_rectFeatureROI[j][0].rows, CV_32FC1);
		detectColor_Revised_Mat(vec_rectFeatureROI[j], sLocalROI[j].mROI);
	}
}
