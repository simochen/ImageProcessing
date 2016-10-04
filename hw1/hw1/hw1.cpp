#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;

Mat getHistogram(const Mat &srcImage)
{
	//计算直方图
	MatND hist;
	int channel = 0;
	int dims = 1;
	int size = 256;
	float hranges[] = { 0, 255 };
	const float *ranges[] = { hranges };
	calcHist(&srcImage, 1, &channel, Mat(), hist, dims, &size, ranges);
	//获取最大值和最小值
	double minVal = 0, maxVal = 0;
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	//显示直方图
	Mat dstImage(size, size, CV_8U, Scalar(255));
	int hpt = static_cast<int>(size);
	for (int i = 0; i < size; i++) {
		//计算直方图高度
		float binVal = hist.at<float>(i);
		int intVal = static_cast<int>(binVal*hpt / maxVal);
		//两点之间绘制一条线段
		line(dstImage, Point(i, size), Point(i, size - intVal), Scalar(0));
	}
	return dstImage;
}

int main()
{
	//读取原图
	Mat srcImage = imread("orange.jpg", 0);
	imshow("source Image", srcImage);
	if (!srcImage.data)
	{
		cout << "fail to load image!" << endl;
	}
	//计算直方图
	Mat hist1;
	hist1 = getHistogram(srcImage);
	imshow("source Histogram", hist1);
	imwrite("srcHist.jpg", hist1);
	//计算均衡化图
	Mat dstImage;
	equalizeHist(srcImage, dstImage);
	imshow("equalized Image", dstImage);
	imwrite("dstImage.jpg", dstImage);
	//计算均衡化直方图
	Mat hist2;
	hist2 = getHistogram(dstImage);
	imshow("equalized Histogram", hist2);
	imwrite("dstHist.jpg", hist2);

	waitKey(0);
}