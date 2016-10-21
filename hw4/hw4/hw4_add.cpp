#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;

void centralize(Mat &image)
{
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			image.at<float>(i, j) *= pow(-1, i + j);
		}
	}
}

Mat performDFT(const Mat &srcImage)
{
	//获得最佳的DFT尺寸
	int m, n;
	m = getOptimalDFTSize(srcImage.rows);
	n = getOptimalDFTSize(srcImage.cols);
	//边界扩展
	Mat dstImage;
	srcImage.convertTo(dstImage, CV_32F);
	copyMakeBorder(dstImage, dstImage, 0, m - srcImage.rows, 0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
	//中心化
	centralize(dstImage);
	//新建两页的浮点数array，其中第一页用中心化后的图像初始化，第二页初始化为0 
	Mat imgArray[] = { Mat_<float>(dstImage), Mat::zeros(dstImage.size(), CV_32F) };
	//把两页合成一个2通道的图像
	merge(imgArray, 2, dstImage);
	//dft，傅里叶变换结果为复数。通道1存的是实部，通道2存的是虚部。
	dft(dstImage, dstImage);
	return dstImage;
}

Mat performIDFT(const Mat &srcImage)
{
	Mat dstImage;
	//idft，取实部
	idft(srcImage, dstImage, DFT_REAL_OUTPUT);
	//中心化
	centralize(dstImage);
	//归一化
	normalize(dstImage, dstImage, 0, 1, CV_MINMAX);
	return dstImage;
}

Mat complexDiv(const Mat &cMat1, const Mat &cMat2, double scale = 1) {
	Mat plane1[2], plane2[2], divPlane[2];
	split(cMat1, plane1);
	split(cMat2, plane2);
	Mat mul[6];
	multiply(plane2[0], plane2[0], mul[0]);
	multiply(plane2[1], plane2[1], mul[1]);
	multiply(plane1[0], plane2[0], mul[2]);
	multiply(plane1[1], plane2[1], mul[3]);
	multiply(plane1[1], plane2[0], mul[4]);
	multiply(plane1[0], plane2[1], mul[5]);
	divide(mul[2] + mul[3], mul[0] + mul[1], divPlane[0], scale);
	divide(mul[4] - mul[5], mul[0] + mul[1], divPlane[1], scale);
	Mat divMat;
	merge(divPlane, 2, divMat);
	return divMat;
}

int main() {
	//读取原图
	Mat srcImage, graySrc;
	srcImage = imread("lena.jpg");
	srcImage = srcImage(Rect(30, 12, 320, 320));
	cvtColor(srcImage, graySrc, CV_RGB2GRAY);
	imshow("Source Image", srcImage);
	imshow("Gray Source Image", graySrc);
	Mat blurImage, grayBlur;
	blurImage = imread("blur.jpg");
	//imshow("Blur Image", blurImage);
	cout << srcImage.rows << " , " << srcImage.cols << endl;
	cout << blurImage.rows << " , " << blurImage.cols << endl;
	resize(blurImage, blurImage, srcImage.size());
	cvtColor(blurImage, grayBlur, CV_RGB2GRAY);
	imshow("Resized Blur Image", blurImage);
	imshow("Gray Blur Image", grayBlur);
	Mat srcF, blurF, degF;
	srcF = performDFT(graySrc);
	blurF = performDFT(grayBlur);
	degF = complexDiv(blurF, srcF);
	Mat blurPlane[3], est_srcN[3];
	split(blurImage, blurPlane);
	est_srcN[0] = performIDFT(complexDiv(performDFT(blurPlane[0]), degF));
	est_srcN[1] = performIDFT(complexDiv(performDFT(blurPlane[1]), degF));
	est_srcN[2] = performIDFT(complexDiv(performDFT(blurPlane[2]), degF));
	Mat est_srcImage;
	merge(est_srcN, 3, est_srcImage);
	imshow("Estimate Source Image", est_srcImage);
	waitKey(0);

	return 0;
}