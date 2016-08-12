#include "opencv/cv.h"
#include "opencv/highgui.h"

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <assert.h> 
#include <math.h> 
#include <float.h> 
#include <limits.h> 
#include <time.h> 
#include <ctype.h>

#ifdef _EiC 
#define WIN32 
#endif

static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

void detect_and_draw(IplImage* image);

const char* cascade_name =
"haarcascade_frontalface_alt.xml";
/*    "haarcascade_profileface.xml";*/

int main(int argc, char** argv)
{
	cascade_name = "haarcascade_frontalface_alt2.xml";
	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);

	if (!cascade)
	{
		fprintf(stderr, "ERROR: Could not load classifier cascade\n");
		return -1;
	}
	storage = cvCreateMemStorage(0);//用来创建一个内存存储器,括号内参数对应内存器中每个内存块的大小，为0时内存块默认大小为64k。
	cvNamedWindow("result", 1);//创建XXX的窗口

	const char* filename = "1.jpg";
	IplImage* image = cvLoadImage(filename, 1);//-1 默认读取图像的原通道数, 0 强制转化读取图像为灰度图, 1 读取彩色图

	if (image)
	{
		detect_and_draw(image);
		cvWaitKey(0);//按任意键退出
		cvReleaseImage(&image);
	}

	cvDestroyWindow("result");

	return 0;
}


void detect_and_draw(IplImage* img)
{
	double scale = 1.2;//缩放比率
	static CvScalar colors[] = {
		{ { 0, 0, 255 } }, { { 0, 128, 255 } }, { { 0, 255, 255 } }, { { 0, 255, 0 } },
		{ { 255, 128, 0 } }, { { 255, 255, 0 } }, { { 255, 0, 0 } }, { { 255, 0, 255 } }
	};//Just some pretty colors to draw with

	//Image Preparation 
	// 
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);//IplImage* cvCreateImage( CvSize size, int depth, int channels );
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width / scale), cvRound(img->height / scale)), 8, 1);
	cvCvtColor(img, gray, CV_BGR2GRAY);//得到灰度图
	cvResize(gray, small_img, CV_INTER_LINEAR);

	cvEqualizeHist(small_img, small_img); //直方图均衡
	/**
	*直方图均衡化，可以将比较淡的图像变换为比较深的图像（即增强图像的亮度及对比度）。
	直方图均衡化后面潜在的数学原理是一个分布（输入的亮度直方图）被映射到另一个分布
	（一个更宽，理想统一的亮度值分布），映射函数是一个累积分布函数。对于连续分布，
	结果将是准确的均衡化。在cvEqualizeHist中，原始图像及目标图像必须是单通道，大小
	相同的8位图像，对于彩色图像，必须先将每个通道分开，再分别进行直方图均衡化处理，
	然后将通道合并形成新的图像。
	*/

	//Detect objects if any 
	// 
	cvClearMemStorage(storage);
	double t = (double)cvGetTickCount();
	CvSeq* objects = cvHaarDetectObjects(small_img,
		cascade,
		storage,
		1.1,
		2,
		0/*CV_HAAR_DO_CANNY_PRUNING*/,
		cvSize(30, 30));
	/*
	image 被检图像
	cascade haar 分类器级联的内部标识形式
	storage 用来存储检测到的一序列候选目标矩形框的内存区域。
	scale_factor 在前后两次相继的扫描中，搜索窗口的比例系数。例如1.1指将搜索窗口依次扩大10%
	min_neighbors 构成检测目标的相邻矩形的最小个数(缺省－1)。如果组成检测目标的小矩形的个数和小于 min_neighbors-1 都会被排除。如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，这种设定值一般用在用户自定义对检测结果的组合程序上。
	flags 操作方式。当前唯一可以定义的操作方式是 CV_HAAR_DO_CANNY_PRUNING。如果被设定，函数利用Canny边缘检测器来排除一些边缘很少或者很多的图像区域，因为这样的区域一般不含被检目标。人脸检测中通过设定阈值使用了这种方法，并因此提高了检测速度。
	min_size 检测窗口的最小尺寸。缺省的情况下被设为分类器训练时采用的样本尺寸(人脸检测中缺省大小是~20×20)。
	
	*/
	t = (double)cvGetTickCount() - t;
	printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));

	//Loop through found objects and draw boxes around them 
	for (int i = 0; i<(objects ? objects->total : 0)/*总共检测了几张脸就要画几个矩形*/; ++i)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(objects, i);
		cvRectangle(img, cvPoint(r->x*scale, r->y*scale), cvPoint((r->x + r->width)*scale, (r->y + r->height)*scale), colors[i % 8]);
	}
	for (int i = 0; i < (objects ? objects->total : 0); i++)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(objects, i);
		CvPoint center;
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);
		cvCircle(img, center, radius, colors[i % 8], 3, 8, 0);
	}

	cvShowImage("result", img);
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
}
