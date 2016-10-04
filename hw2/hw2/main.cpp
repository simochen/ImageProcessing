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

Mat calcSpectrum(const Mat &srcImage)
{
	//将二通道图像分割成两个单通道图像
	Mat imgArray[2];
	split(srcImage, imgArray);
	//计算幅值
	Mat dstImage;
	magnitude(imgArray[0], imgArray[1], dstImage);
	//取log(1+|F|)
	dstImage += Scalar(1);
	log(dstImage, dstImage);
	//归一化
	normalize(dstImage, dstImage, 0, 1, NORM_MINMAX);
	return dstImage;
}

Mat filter(const Mat &srcImage)
{
	//巴特沃斯滤波器
	Mat BF(srcImage.size(), CV_32F);
	Mat dstImage(srcImage.size(), CV_32F);
	int Dl = 1500, Dh = 40000;
	int n = 4;
	float lp, hp;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			//低通部分
			lp = 1 / (1 + pow(((pow((i + 1 - srcImage.rows / 2), 2) + pow((j + 1 - srcImage.cols / 2), 2)) / (Dl ^ 2)), n));
			//高通部分
			hp = 1 / (1 + pow((Dh ^ 2) / ((pow((i + 1 - srcImage.rows / 2), 2) + pow((j + 1 - srcImage.cols / 2), 2))), n));
			BF.at<float>(i, j) = lp + hp;
		}
	}
	//调整带阻滤波器，使阻带部分衰减至10%（而非完全阻断）
	normalize(BF, BF, 0.1, 1, CV_MINMAX);
	//频域滤波
	multiply(srcImage, BF, dstImage);
	imshow("Butterworth Filter", BF);
	return dstImage;
}

Mat performIDFT(const Mat &srcImage)
{
	//将二通道图像分割成两个单通道图像
	Mat imgArray[2];
	split(srcImage, imgArray);
	//频域滤波
	imgArray[0] = filter(imgArray[0]);
	imgArray[1] = filter(imgArray[1]);
	//把滤波后的实部和虚部合成一个2通道的图像
	Mat dstImage;
	merge(imgArray, 2, dstImage);
	//idft，取实部
	idft(dstImage, dstImage, DFT_REAL_OUTPUT);
	//中心化
	centralize(dstImage);
	//归一化
	normalize(dstImage, dstImage, 0, 1, CV_MINMAX);
	return dstImage;
}

int main()
{
	//读取原图
	Mat srcImage = imread("origin.jpg", 0);
	imshow("Source Image", srcImage);
	if (!srcImage.data)
	{
		cout << "fail to load image!" << endl;
	}
	//计算原图频谱
	Mat srcDFT, srcSpec;
	srcDFT = performDFT(srcImage);
	srcSpec = calcSpectrum(srcDFT);
	imshow("Source Spectrum Image", srcSpec);
	//读取加噪图像
	Mat noiImage = imread("addnoise.jpg", 0);
	imshow("Noisy Image", noiImage);
	if (!noiImage.data)
	{
		cout << "fail to load image!" << endl;
	}
	//计算加噪图频谱
	Mat noiDFT, noiSpec, filSpec;
	noiDFT = performDFT(noiImage);
	noiSpec = calcSpectrum(noiDFT);
	imshow("Noisy Spectrum Image", noiSpec);
	//频域滤波
	filSpec = filter(noiSpec);
	imshow("Filtered Spectrum", filSpec);
	Mat dstImage = performIDFT(noiDFT);
	//裁剪扩展的边界
	dstImage = dstImage(Rect(0, 0, noiImage.cols, noiImage.rows));
	imshow("Filtered Image", dstImage);
	waitKey(0);
	return 0;
}