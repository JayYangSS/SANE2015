/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRPatch.h"
#include <highgui.h>
#include <stdio.h>

#include <deque>

using namespace std;

void CRPatch::extractPatches(IplImage *img, unsigned int n, int label, CvRect* box, std::vector<CvPoint>* vCenter) {
	// extract features
	vector<IplImage*> vImg;
	extractFeatureChannels(img, vImg);	// extract feature channels and find the optimi 

	CvMat tmp;
	int offx = width/2; 
	int offy = height/2;

	// generate x,y locations
	CvMat* locations = cvCreateMat( n, 1, CV_32SC2 );
	if(box==0)
		cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(0,0,0,0), cvScalar(img->width-width,img->height-height,0,0) );
	else
		cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(box->x,box->y,0,0), cvScalar(box->x+box->width-width,box->y+box->height-height,0,0) );

	// reserve memory
	unsigned int offset = vLPatches[label].size();
	vLPatches[label].reserve(offset+n);
	for(unsigned int i=0; i<n; ++i) {
		CvPoint pt = *(CvPoint*)cvPtr1D( locations, i, 0 );
		
		PatchFeature pf;
		vLPatches[label].push_back(pf);	//positive feature vector에 push

		vLPatches[label].back().roi.x = pt.x;  vLPatches[label].back().roi.y = pt.y;  
		vLPatches[label].back().roi.width = width;  vLPatches[label].back().roi.height = height; 

		if(vCenter!=0) {
			vLPatches[label].back().center.resize(vCenter->size());
			for(unsigned int c = 0; c<vCenter->size(); ++c) {
				vLPatches[label].back().center[c].x = pt.x + offx - (*vCenter)[c].x;
				vLPatches[label].back().center[c].y = pt.y + offy - (*vCenter)[c].y;
			}
		}

		vLPatches[label].back().vPatch.resize(vImg.size());
		for(unsigned int c=0; c<vImg.size(); ++c) {
			cvGetSubRect( vImg[c], &tmp,  vLPatches[label].back().roi );
			vLPatches[label].back().vPatch[c] = cvCloneMat(&tmp);
		}

	}

	cvReleaseMat(&locations);
	for(unsigned int c=0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);
}

void CRPatch::extractFeatureChannels(IplImage *img, std::vector<IplImage*>& vImg) {	//3 channel로 수정
	// 32 feature channels
	// 7+9 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
	// 16+16 channels: minfilter + maxfilter on 5x5 neighborhood 

	//vImg.resize(32);

	////convert to gray image
	//IplImage *imgGray;
	//if (img->nChannels == 1) imgGray = img;
	//else cvConvertImage(img, imgGray, CV_BGR2GRAY);


	vImg.resize(3);
	for(unsigned int c=0; c<vImg.size(); ++c)
		vImg[c] = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U , 1); 

	// Get intensity
	cvConvertImage(img, vImg[0], CV_BGR2GRAY);
	//cvCvtColor( img, vImg[0], CV_RGB2GRAY );

	// Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
	IplImage* I_x = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1); 
	IplImage* I_y = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1); 
	
	// |I_x|, |I_y|
	cvSobel(vImg[0],I_x,1,0,3);			

	cvSobel(vImg[0],I_y,0,1,3);			

	//cvConvertScaleAbs( I_x, vImg[3], 0.25);

	//cvConvertScaleAbs( I_y, vImg[4], 0.25);
	
	cvConvertScaleAbs( I_x, vImg[1], 0.25);

	cvConvertScaleAbs( I_y, vImg[2], 0.25);

	//cvNamedWindow("intensity");
	//cvNamedWindow("sobel_x");
	//cvNamedWindow("sobel_y");
	//cvShowImage("intensity", vImg[0]);
	//cvShowImage("sobel_x", vImg[1]);
	//cvShowImage("sobel_y", vImg[2]);
	//cvWaitKey(0);

	////imshow("x", vImg[1]);

	//{
	//  short* dataX;
	//  short* dataY;
	//  uchar* dataZ;
	//  int stepX, stepY, stepZ;
	//  CvSize size;
	//  int x, y;

	//  cvGetRawData( I_x, (uchar**)&dataX, &stepX, &size);
	//  cvGetRawData( I_y, (uchar**)&dataY, &stepY);
	//  cvGetRawData( vImg[1], (uchar**)&dataZ, &stepZ);
	//  stepX /= sizeof(dataX[0]);
	//  stepY /= sizeof(dataY[0]);
	//  stepZ /= sizeof(dataZ[0]);
	//  
	//  // Orientation of gradients
	//  for( y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ  )
	//    for( x = 0; x < size.width; x++ ) {
	//      // Avoid division by zero
	//      float tx = (float)dataX[x] + (float)_copysign(0.000001f, (float)dataX[x]);
	//      // Scaling [-pi/2 pi/2] -> [0 80*pi]
	//      dataZ[x]=uchar( ( atan((float)dataY[x]/tx)+3.14159265f/2.0f ) * 80 ); 
	//    }
	//}
	//
	//{
	//  short* dataX;
	//  short* dataY;
	//  uchar* dataZ;
	//  int stepX, stepY, stepZ;
	//  CvSize size;
	//  int x, y;
	//  
	//  cvGetRawData( I_x, (uchar**)&dataX, &stepX, &size);
	//  cvGetRawData( I_y, (uchar**)&dataY, &stepY);
	//  cvGetRawData( vImg[2], (uchar**)&dataZ, &stepZ);
	//  stepX /= sizeof(dataX[0]);
	//  stepY /= sizeof(dataY[0]);
	//  stepZ /= sizeof(dataZ[0]);
	//  
	//  // Magnitude of gradients
	//  for( y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ  )
	//    for( x = 0; x < size.width; x++ ) {
	//      dataZ[x] = (uchar)( sqrt((float)dataX[x]*(float)dataX[x] + (float)dataY[x]*(float)dataY[x]) );
	//    }
	//}

	//// 9-bin HOG feature stored at vImg[7] - vImg[15] 
	//hog.extractOBin(vImg[1], vImg[2], vImg, 7);
	//
	//// |I_xx|, |I_yy|

	//cvSobel(vImg[0],I_x,2,0,3);
	//cvConvertScaleAbs( I_x, vImg[5], 0.25);	
	//
	//cvSobel(vImg[0],I_y,0,2,3);
	//cvConvertScaleAbs( I_y, vImg[6], 0.25);
	//
	//// L, a, b
	//cvCvtColor( img, img, CV_RGB2Lab  );

	//cvReleaseImage(&I_x);
	//cvReleaseImage(&I_y);	
	//
	//cvSplit( img, vImg[0], vImg[1], vImg[2], 0);

	////여기서 에러
	////아마 두점에서의 차의 value의 최소, 최대 구해서 feature로 사용할 수 있도록하는 부분
	//// min filter  ??
	//for(int c=0; c<2; ++c)
	//	minfilt(vImg[c], vImg[c+1], 5);

	////max filter
	//for(int c=0; c<3; ++c)
	//	maxfilt(vImg[c], 5);


	//
#if 0
	// for debugging only
	char buffer[40];
	for(unsigned int i = 0; i<vImg.size();++i) {
		sprintf(buffer,"out-%d.png",i);
		cvNamedWindow(buffer,1);
		cvShowImage(buffer, vImg[i]);
		//cvSaveImage( buffer, vImg[i] );
	}

	cvWaitKey();

	for(unsigned int i = 0; i<vImg.size();++i) {
		sprintf(buffer,"%d",i);
		cvDestroyWindow(buffer);
	}
#endif


}


void CRPatch::extractFeatureChannels_Mat(Mat& img, std::vector<Mat>& vImg) {	//3 channel로 수정
	// 32 feature channels
	// 7+9 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
	// 16+16 channels: minfilter + maxfilter on 5x5 neighborhood 

	//vImg.resize(32);
	Mat mGray = Mat(img.rows, img.cols, CV_8UC1);

	vImg.resize(3);
	for (unsigned int c = 0; c < vImg.size(); c++)
		vImg[c] = Mat(img.rows, img.cols, CV_8UC1);

	// Get intensity
	//cvtColor(img, vImg[0], CV_RGB2GRAY);
	cvtColor(img, mGray, CV_RGB2GRAY);
	//cvCvtColor( img, vImg[0], CV_RGB2GRAY );

	// Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
	Mat I_x, I_y;
	// |I_x|, |I_y|

	equalizeHist(mGray, vImg[0]);

	//first order derivative by x and y directions
	Sobel(vImg[0], I_x, CV_16S, 1, 0, 3);
	Sobel(vImg[0], I_y, CV_16S, 0, 1, 3);

	//////cvlab_1,8
	//convertScaleAbs(I_x*0.8, vImg[1]);
	//convertScaleAbs(I_y*0.8, vImg[2]);

	////cvlab_1,8
	convertScaleAbs(I_x*0.5, vImg[1]);
	convertScaleAbs(I_y*0.5, vImg[2]);

	//////cvlab_2,5,6
	//convertScaleAbs(I_x*0.25, vImg[1]);
	//convertScaleAbs(I_y*0.25, vImg[2]);

	////HoG
	//int x, y;
	//for (y = 0; y < I_x.rows; y++){
	//	short* data_x = I_x.ptr<short>(y);
	//	short* data_y = I_y.ptr<short>(y);
	//	uchar* data_zo = vImg[1].ptr<uchar>(y);
	//	uchar* data_zm = vImg[2].ptr<uchar>(y);
	//	for (x = 0; x < I_x.cols; x++){
	//		float tx = (float)data_x[x] + (float)_copysign(0.0000001f, (float)data_x[x]);
	//		data_zo[x] = (uchar)((atan((float)data_y[x] / tx) + CV_PI / 2.0f) * 80);
	//		data_zm[x] = (uchar)sqrt(data_x[x] * data_x[x] + data_y[x] * data_y[x]);
	//	}
	//}

	////second order derivative by x and y directions
	//Sobel(vImg[0], I_x, CV_16S, 2, 0, 3);
	//Sobel(vImg[0], I_y, CV_16S, 0, 2, 3);
	//convertScaleAbs(I_x*0.25, vImg[3]);
	//convertScaleAbs(I_y*0.25, vImg[4]);

	////check derivatives
	//Mat subX, subY;
	//subtract(vImg[1],vImg[3],subX);
	//subtract(vImg[2], vImg[4], subY);

	//imshow("x1", vImg[1]);
	//imshow("y1", vImg[2]);
	//imshow("x2", vImg[3]);
	//imshow("y2", vImg[4]);
	//imshow("sub_x", subX);
	//imshow("sub_y", subY);

	//cvWaitKey(0);

#if 0
	// for debugging only
	char buffer[40];
	for (unsigned int i = 0; i < vImg.size(); ++i) {
		sprintf(buffer, "out-%d.png", i);
		imshow(buffer, vImg[i]);
	}
	cvWaitKey();
#endif

}


void CRPatch::maxfilt(IplImage *src, unsigned int width) {

	uchar* s_data;
	int step;
	CvSize size;

	cvGetRawData( src, (uchar**)&s_data, &step, &size );
	step /= sizeof(s_data[0]);

	for(int  y = 0; y < size.height; y++) {
		maxfilt(s_data+y*step, 1, size.width, width);
	}

	cvGetRawData( src, (uchar**)&s_data);

	for(int  x = 0; x < size.width; x++)
		maxfilt(s_data+x, step, size.height, width);

}

void CRPatch::maxfilt(IplImage *src, IplImage *dst, unsigned int width) {

	uchar* s_data;
	uchar* d_data;
	int step;
	CvSize size;

	cvGetRawData( src, (uchar**)&s_data, &step, &size );
	cvGetRawData( dst, (uchar**)&d_data, &step, &size );
	step /= sizeof(s_data[0]);

	for(int  y = 0; y < size.height; y++)
		maxfilt(s_data+y*step, d_data+y*step, 1, size.width, width);

	cvGetRawData( src, (uchar**)&d_data);

	for(int  x = 0; x < size.width; x++)
		maxfilt(d_data+x, step, size.height, width);

}

void CRPatch::minfilt(IplImage *src, unsigned int width) {

	uchar* s_data;
	int step;
	CvSize size;

	cvGetRawData( src, (uchar**)&s_data, &step, &size );
	step /= sizeof(s_data[0]);

	for(int  y = 0; y < size.height; y++)
		minfilt(s_data+y*step, 1, size.width, width);

	cvGetRawData( src, (uchar**)&s_data);

	for(int  x = 0; x < size.width; x++)
		minfilt(s_data+x, step, size.height, width);

}

void CRPatch::minfilt(IplImage *src, IplImage *dst, unsigned int width) {

	uchar* s_data;
	uchar* d_data;
	int step;
	CvSize size;

	cvGetRawData( src, (uchar**)&s_data, &step, &size );
	cvGetRawData( dst, (uchar**)&d_data, &step, &size );
	step /= sizeof(s_data[0]);

	for(int  y = 0; y < size.height; y++)
		minfilt(s_data+y*step, d_data+y*step, 1, size.width, width);

	cvGetRawData( src, (uchar**)&d_data);

	for(int  x = 0; x < size.width; x++)
		minfilt(d_data+x, step, size.height, width);

}


void CRPatch::maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	maxvalues[0] = data[0];
	for(unsigned int i=0; i < d-step; i+=step) {
		for(unsigned int k=i; k<d+i; k+=step) {
			if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
		}
		maxvalues[i+step] = maxvalues[i];
	}

	maxvalues[size-step] = data[size-step];
	for(unsigned int i=size-step; i > size-d; i-=step) {
		for(unsigned int k=i; k>i-d; k-=step) {
			if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
		}
		maxvalues[i-step] = maxvalues[i];
	}

    deque<int> maxfifo;
    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			maxvalues[i-d] = data[maxfifo.size()>0 ? maxfifo.front(): i-step];
		}
    
		if(data[i] < data[i-step]) { 

			maxfifo.push_back(i-step);
			if(i==  width+maxfifo.front()) 
				maxfifo.pop_front();

		} else {

			while(maxfifo.size() > 0) {
				if(data[i] <= data[maxfifo.back()]) {
					if(i==  width+maxfifo.front()) 
						maxfifo.pop_front();
				break;
				}
				maxfifo.pop_back();
			}

		}

    }  

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];
 
}

void CRPatch::maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	deque<uchar> tmp;

	tmp.push_back(data[0]);
	for(unsigned int k=step; k<d; k+=step) {
		if(data[k]>tmp.back()) tmp.back() = data[k];
	}

	for(unsigned int i=step; i < d-step; i+=step) {
		tmp.push_back(tmp.back());
		if(data[i+d-step]>tmp.back()) tmp.back() = data[i+d-step];
	}


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
			data[i-width] = tmp.front();
			tmp.pop_front();
		}
    
		if(data[i] < data[i-step]) { 

			minfifo.push_back(i-step);
			if(i==  width+minfifo.front()) 
				minfifo.pop_front();

		} else {

			while(minfifo.size() > 0) {
				if(data[i] <= data[minfifo.back()]) {
					if(i==  width+minfifo.front()) 
						minfifo.pop_front();
				break;
				}
				minfifo.pop_back();
			}

		}

    }  

	tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);
	
	for(unsigned int k=size-step-step; k>=size-d; k-=step) {
		if(data[k]>data[size-step]) data[size-step] = data[k];
	}

	for(unsigned int i=size-step-step; i >= size-d; i-=step) {
		data[i] = data[i+step];
		if(data[i-d+step]>data[i]) data[i] = data[i-d+step];
	}

	for(unsigned int i=size-width; i<=size-d; i+=step) {
		data[i] = tmp.front();
		tmp.pop_front();
	}
 
}

void CRPatch::minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	minvalues[0] = data[0];
	for(unsigned int i=0; i < d-step; i+=step) {
		for(unsigned int k=i; k<d+i; k+=step) {
			if(data[k]<minvalues[i]) minvalues[i] = data[k];
		}
		minvalues[i+step] = minvalues[i];
	}

	minvalues[size-step] = data[size-step];
	for(unsigned int i=size-step; i > size-d; i-=step) {
		for(unsigned int k=i; k>i-d; k-=step) {
			if(data[k]<minvalues[i]) minvalues[i] = data[k];
		}
		minvalues[i-step] = minvalues[i];
	}

    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
		}
    
		if(data[i] > data[i-step]) { 

			minfifo.push_back(i-step);
			if(i==  width+minfifo.front()) 
				minfifo.pop_front();

		} else {

			while(minfifo.size() > 0) {
				if(data[i] >= data[minfifo.back()]) {
					if(i==  width+minfifo.front()) 
						minfifo.pop_front();
				break;
				}
				minfifo.pop_back();
			}

		}

    }  

    minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];
 
}

void CRPatch::minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	deque<uchar> tmp;

	tmp.push_back(data[0]);
	for(unsigned int k=step; k<d; k+=step) {
		if(data[k]<tmp.back()) tmp.back() = data[k];
	}

	for(unsigned int i=step; i < d-step; i+=step) {
		tmp.push_back(tmp.back());
		if(data[i+d-step]<tmp.back()) tmp.back() = data[i+d-step];
	}


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
			data[i-width] = tmp.front();
			tmp.pop_front();
		}
    
		if(data[i] > data[i-step]) { 

			minfifo.push_back(i-step);
			if(i==  width+minfifo.front()) 
				minfifo.pop_front();

		} else {

			while(minfifo.size() > 0) {
				if(data[i] >= data[minfifo.back()]) {
					if(i==  width+minfifo.front()) 
						minfifo.pop_front();
				break;
				}
				minfifo.pop_back();
			}

		}

    }  

	tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);
	
	for(unsigned int k=size-step-step; k>=size-d; k-=step) {
		if(data[k]<data[size-step]) data[size-step] = data[k];
	}

	for(unsigned int i=size-step-step; i >= size-d; i-=step) {
		data[i] = data[i+step];
		if(data[i-d+step]<data[i]) data[i] = data[i-d+step];
	}
 
	for(unsigned int i=size-width; i<=size-d; i+=step) {
		data[i] = tmp.front();
		tmp.pop_front();
	}
}

void CRPatch::maxminfilt(uchar* data, uchar* maxvalues, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	maxvalues[0] = data[0];
	minvalues[0] = data[0];
	for(unsigned int i=0; i < d-step; i+=step) {
		for(unsigned int k=i; k<d+i; k+=step) {
			if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
			if(data[k]<minvalues[i]) minvalues[i] = data[k];
		}
		maxvalues[i+step] = maxvalues[i];
		minvalues[i+step] = minvalues[i];
	}

	maxvalues[size-step] = data[size-step];
	minvalues[size-step] = data[size-step];
	for(unsigned int i=size-step; i > size-d; i-=step) {
		for(unsigned int k=i; k>i-d; k-=step) {
			if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
			if(data[k]<minvalues[i]) minvalues[i] = data[k];
		}
		maxvalues[i-step] = maxvalues[i];
		minvalues[i-step] = minvalues[i];
	}

    deque<int> maxfifo, minfifo;

    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			maxvalues[i-d] = data[maxfifo.size()>0 ? maxfifo.front(): i-step];
			minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
		}
    
		if(data[i] > data[i-step]) { 

			minfifo.push_back(i-step);
			if(i==  width+minfifo.front()) 
				minfifo.pop_front();
			while(maxfifo.size() > 0) {
				if(data[i] <= data[maxfifo.back()]) {
					if (i==  width+maxfifo.front()) 
						maxfifo.pop_front();
					break;
				}
				maxfifo.pop_back();
			}

		} else {

			maxfifo.push_back(i-step);
			if (i==  width+maxfifo.front()) 
				maxfifo.pop_front();
			while(minfifo.size() > 0) {
				if(data[i] >= data[minfifo.back()]) {
					if(i==  width+minfifo.front()) 
						minfifo.pop_front();
				break;
				}
				minfifo.pop_back();
			}

		}

    }  

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];
	minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];
 
}
