#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <cmath>

using namespace cv;
using namespace std;

#define PI atan(1)*4

//离散傅里叶变换
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
	//新建两页的浮点数array，其中第一页用中心化后的图像初始化，第二页初始化为0 
	Mat imgArray[] = { Mat_<float>(dstImage), Mat::zeros(dstImage.size(), CV_32F) };
	//把两页合成一个2通道的图像
	merge(imgArray, 2, dstImage);
	//dft，傅里叶变换结果为复数。通道1存的是实部，通道2存的是虚部。
	dft(dstImage, dstImage);
	return dstImage;
}

//离散傅里叶逆变换
Mat performIDFT(const Mat &srcImage)
{
	Mat dstImage;
	//idft，取实部
	idft(srcImage, dstImage, DFT_REAL_OUTPUT);
	//归一化
	normalize(dstImage, dstImage, 0, 1, CV_MINMAX);
	return dstImage;
}

// 运动模糊
Mat motionBlur(Size msize, double a, double b, double T) {
	//建立一个2通道32位浮点数的矩阵，通道1存放实部，通道2存放虚部
	Mat degMat = Mat(msize, CV_32FC2);
	double k;
	for (int i = 0; i < msize.width; i++) {
		for (int j = 0; j < msize.height; j++) {
			k = 2 * PI * (i * a + j * b);
			//处理分母为0的情况
			if (k == 0) k = 0.0001;
			degMat.at<Vec2f>(i, j)[0] = T * sin(k) / k;
			degMat.at<Vec2f>(i, j)[1] = T * (cos(k) - 1) / k;
		}
	}
	return degMat;
}

//维纳滤波
Mat wienerFilter(const Mat &blurMat, const Mat &degMat, double K = 0.01) {
	Mat blur_plane[2], deg_plane[2], res_plane[2];
	//分割2通道的矩阵，提取实部和虚部
	split(blurMat, blur_plane);
	split(degMat, deg_plane);
	Mat mul[6];
	multiply(deg_plane[0], deg_plane[0], mul[0]);
	multiply(deg_plane[1], deg_plane[1], mul[1]);
	multiply(blur_plane[0], deg_plane[0], mul[2]);
	multiply(blur_plane[1], deg_plane[1], mul[3]);
	multiply(blur_plane[1], deg_plane[0], mul[4]);
	multiply(blur_plane[0], deg_plane[1], mul[5]);
	divide(mul[2] + mul[3], mul[0] + mul[1] + K, res_plane[0]);
	divide(mul[4] - mul[5], mul[0] + mul[1] + K, res_plane[1]);
	Mat resMat;
	merge(res_plane, 2, resMat);
	return resMat;
}

int main() {
	//读取模糊图像
	Mat blurImage;
	blurImage = imread("blur.jpg");
	imshow("Blur Image", blurImage);
	//分割RGB三通道图像
	Mat blur_plane[3], blur_spec[3];
	split(blurImage, blur_plane);
	//离散傅里叶变换
	blur_spec[0] = performDFT(blur_plane[0]);
	blur_spec[1] = performDFT(blur_plane[1]);
	blur_spec[2] = performDFT(blur_plane[2]);
	//生成运动模糊退化函数
	Mat H;
	H = motionBlur(Size(blur_spec[0].cols, blur_spec[0].rows), 0.0005, -0.0001, 1);
	//维纳滤波，傅里叶逆变换
	Mat res_plane[3];
	res_plane[0] = performIDFT(wienerFilter(blur_spec[0], H, 0.02));
	res_plane[1] = performIDFT(wienerFilter(blur_spec[1], H, 0.02));
	res_plane[2] = performIDFT(wienerFilter(blur_spec[2], H, 0.02));
	//合成三通道图像
	Mat resImage;
	merge(res_plane, 3, resImage);
	imshow("Restored Image", resImage);
	
	waitKey(0);

	return 0;
}