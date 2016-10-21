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

Mat calcNoise(const Mat &srcDFT, const Mat &blurDFT) {
	Mat srcPlane[2], blurPlane[2], noisePlane[2];
	split(srcDFT, srcPlane);
	split(blurDFT, blurPlane);
	Mat mul[6];
	multiply(blurPlane[0], blurPlane[0], mul[0]);
	multiply(blurPlane[1], blurPlane[1], mul[1]);
	multiply(srcPlane[0], blurPlane[0], mul[2]);
	multiply(srcPlane[1], blurPlane[1], mul[3]);
	multiply(srcPlane[1], blurPlane[0], mul[4]);
	multiply(srcPlane[0], blurPlane[1], mul[5]);
	divide(mul[2] + mul[3], mul[0] + mul[1], noisePlane[0]);
	divide(mul[4] - mul[5], mul[0] + mul[1], noisePlane[1]);

}

int main() {
	//读取原图
	Mat srcImage, graySrc;
	srcImage = imread("lena.jpg");
	srcImage = srcImage(Rect(32, 12, 320, 320));
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
	Mat srcDFT, blurDFT;
	srcDFT = performDFT(graySrc);
	blurDFT = performDFT(grayBlur);
	waitKey(0);

	return 0;
}