#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;

Mat getHistogram(const Mat &srcImage)
{
	//����ֱ��ͼ
	MatND hist;
	int channel = 0;
	int dims = 1;
	int size = 256;
	float hranges[] = { 0, 255 };
	const float *ranges[] = { hranges };
	calcHist(&srcImage, 1, &channel, Mat(), hist, dims, &size, ranges);
	//��ȡ���ֵ����Сֵ
	double minVal = 0, maxVal = 0;
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	//��ʾֱ��ͼ
	Mat dstImage(size, size, CV_8U, Scalar(255));
	int hpt = static_cast<int>(size);
	for (int i = 0; i < size; i++) {
		//����ֱ��ͼ�߶�
		float binVal = hist.at<float>(i);
		int intVal = static_cast<int>(binVal*hpt / maxVal);
		//����֮�����һ���߶�
		line(dstImage, Point(i, size), Point(i, size - intVal), Scalar(0));
	}
	return dstImage;
}

int main()
{
	//��ȡԭͼ
	Mat srcImage = imread("orange.jpg", 0);
	imshow("source Image", srcImage);
	if (!srcImage.data)
	{
		cout << "fail to load image!" << endl;
	}
	//����ֱ��ͼ
	Mat hist1;
	hist1 = getHistogram(srcImage);
	imshow("source Histogram", hist1);
	imwrite("srcHist.jpg", hist1);
	//������⻯ͼ
	Mat dstImage;
	equalizeHist(srcImage, dstImage);
	imshow("equalized Image", dstImage);
	imwrite("dstImage.jpg", dstImage);
	//������⻯ֱ��ͼ
	Mat hist2;
	hist2 = getHistogram(dstImage);
	imshow("equalized Histogram", hist2);
	imwrite("dstHist.jpg", hist2);

	waitKey(0);
}