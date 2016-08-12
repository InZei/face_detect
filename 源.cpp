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
	storage = cvCreateMemStorage(0);//��������һ���ڴ�洢��,�����ڲ�����Ӧ�ڴ�����ÿ���ڴ��Ĵ�С��Ϊ0ʱ�ڴ��Ĭ�ϴ�СΪ64k��
	cvNamedWindow("result", 1);//����XXX�Ĵ���

	const char* filename = "1.jpg";
	IplImage* image = cvLoadImage(filename, 1);//-1 Ĭ�϶�ȡͼ���ԭͨ����, 0 ǿ��ת����ȡͼ��Ϊ�Ҷ�ͼ, 1 ��ȡ��ɫͼ

	if (image)
	{
		detect_and_draw(image);
		cvWaitKey(0);//��������˳�
		cvReleaseImage(&image);
	}

	cvDestroyWindow("result");

	return 0;
}


void detect_and_draw(IplImage* img)
{
	double scale = 1.2;//���ű���
	static CvScalar colors[] = {
		{ { 0, 0, 255 } }, { { 0, 128, 255 } }, { { 0, 255, 255 } }, { { 0, 255, 0 } },
		{ { 255, 128, 0 } }, { { 255, 255, 0 } }, { { 255, 0, 0 } }, { { 255, 0, 255 } }
	};//Just some pretty colors to draw with

	//Image Preparation 
	// 
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);//IplImage* cvCreateImage( CvSize size, int depth, int channels );
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width / scale), cvRound(img->height / scale)), 8, 1);
	cvCvtColor(img, gray, CV_BGR2GRAY);//�õ��Ҷ�ͼ
	cvResize(gray, small_img, CV_INTER_LINEAR);

	cvEqualizeHist(small_img, small_img); //ֱ��ͼ����
	/**
	*ֱ��ͼ���⻯�����Խ��Ƚϵ���ͼ��任Ϊ�Ƚ����ͼ�񣨼���ǿͼ������ȼ��Աȶȣ���
	ֱ��ͼ���⻯����Ǳ�ڵ���ѧԭ����һ���ֲ������������ֱ��ͼ����ӳ�䵽��һ���ֲ�
	��һ����������ͳһ������ֵ�ֲ�����ӳ�亯����һ���ۻ��ֲ����������������ֲ���
	�������׼ȷ�ľ��⻯����cvEqualizeHist�У�ԭʼͼ��Ŀ��ͼ������ǵ�ͨ������С
	��ͬ��8λͼ�񣬶��ڲ�ɫͼ�񣬱����Ƚ�ÿ��ͨ���ֿ����ٷֱ����ֱ��ͼ���⻯����
	Ȼ��ͨ���ϲ��γ��µ�ͼ��
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
	image ����ͼ��
	cascade haar �������������ڲ���ʶ��ʽ
	storage �����洢��⵽��һ���к�ѡĿ����ο���ڴ�����
	scale_factor ��ǰ��������̵�ɨ���У��������ڵı���ϵ��������1.1ָ������������������10%
	min_neighbors ���ɼ��Ŀ������ھ��ε���С����(ȱʡ��1)�������ɼ��Ŀ���С���εĸ�����С�� min_neighbors-1 ���ᱻ�ų������min_neighbors Ϊ 0, ���������κβ����ͷ������еı����ѡ���ο������趨ֵһ�������û��Զ���Լ��������ϳ����ϡ�
	flags ������ʽ����ǰΨһ���Զ���Ĳ�����ʽ�� CV_HAAR_DO_CANNY_PRUNING��������趨����������Canny��Ե��������ų�һЩ��Ե���ٻ��ߺܶ��ͼ��������Ϊ����������һ�㲻������Ŀ�ꡣ���������ͨ���趨��ֵʹ�������ַ��������������˼���ٶȡ�
	min_size ��ⴰ�ڵ���С�ߴ硣ȱʡ������±���Ϊ������ѵ��ʱ���õ������ߴ�(���������ȱʡ��С��~20��20)��
	
	*/
	t = (double)cvGetTickCount() - t;
	printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));

	//Loop through found objects and draw boxes around them 
	for (int i = 0; i<(objects ? objects->total : 0)/*�ܹ�����˼�������Ҫ����������*/; ++i)
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
