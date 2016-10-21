#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <time.h>
#include <cmath>

using namespace cv;
using namespace std;

Mat addNoise(const Mat &srcImage, double p) {
	Mat dstImage;
	srcImage.copyTo(dstImage);
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			double randnum;
			randnum = rand() / double(RAND_MAX);
			if (randnum < 0.5 * p) dstImage.at<Vec3b>(i, j) = { 0, 0, 0 };
			else if (randnum < p) dstImage.at<Vec3b>(i, j) = { 255, 255, 255 };
		}
	}
	return dstImage;
}

Mat aritMeanFilter(const Mat &srcImage, Size ksize) {
	Mat expImage, dstImage, imageROI;
	srcImage.copyTo(expImage);
	int a, b;
	a = (ksize.width - 1) / 2;
	b = (ksize.height - 1) / 2;
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_REFLECT);
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		Scalar s = sum(expImage(Rect(0, i, ksize.width, ksize.height)));
		for (int j = 0; j < dstImage.cols; j++) {
			if (j > 0) {
				s -= sum(expImage(Rect(j - 1, i, 1, ksize.height)));
				s += sum(expImage(Rect(j + ksize.width - 1, i, 1, ksize.height)));
			}
			Vec4b m = (Vec4b)(s / (ksize.width * ksize.height));
			dstImage.at<Vec3b>(i, j) = { m[0], m[1], m[2] };
		}
	}
	return dstImage;
}

Mat geoMeanFilter(const Mat &srcImage, Size ksize) {
	Mat expImage, dstImage, imageROI;
	srcImage.copyTo(expImage);
	int a, b;
	a = (ksize.width - 1) / 2;
	b = (ksize.height - 1) / 2;
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_REFLECT);
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		Vec3d product = { 1, 1, 1 };
		for (int m = 0; m < ksize.height; m++) {
			for (int n = 0; n < ksize.width; n++) {
				if (expImage.at<Vec3b>(i + m, n)[0] == 0) expImage.at<Vec3b>(i + m, n) = { 255, 255, 255 };
				product[0] *= expImage.at<Vec3b>(i + m, n)[0];
				product[1] *= expImage.at<Vec3b>(i + m, n)[1];
				product[2] *= expImage.at<Vec3b>(i + m, n)[2];
			}
		}
		for (int j = 0; j < dstImage.cols; j++) {
			if (j > 0) {
				for (int m = 0; m < ksize.height; m++) {
					if (expImage.at<Vec3b>(i + m, n)[0] == 0) expImage.at<Vec3b>(i + m, j + ksize.width - 1) = { 255, 255, 255 };
					product[0] /= expImage.at<Vec3b>(i + m, j - 1)[0];
					product[0] *= expImage.at<Vec3b>(i + m, j + ksize.width - 1)[0];
					product[1] /= expImage.at<Vec3b>(i + m, j - 1)[1];
					product[1] *= expImage.at<Vec3b>(i + m, j + ksize.width - 1)[1];
					product[2] /= expImage.at<Vec3b>(i + m, j - 1)[2];
					product[2] *= expImage.at<Vec3b>(i + m, j + ksize.width - 1)[2];
				}
			}
			Vec3b c;
			double b = 1.0 / (ksize.width * ksize.height);
			c[0] = (uchar)pow(product[0], b);
			c[1] = (uchar)pow(product[1], b);
			c[2] = (uchar)pow(product[2], b);
			dstImage.at<Vec3b>(i, j) = c;
		}
	}
	return dstImage;
}

Mat harMeanFilter(const Mat &srcImage, Size ksize) {
	Mat expImage, dstImage, imageROI;
	srcImage.copyTo(expImage);
	int a, b;
	a = (ksize.width - 1) / 2;
	b = (ksize.height - 1) / 2;
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_REFLECT);
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			Vec3d numer = { 0, 0, 0 };
			for (int m = 0; m < ksize.height; m++) {
				for (int n = 0; n < ksize.width; n++) {
					if (expImage.at<Vec3b>(i + m, j + n)[0] == 0) expImage.at<Vec3b>(i + m, j + n) = { 255, 255, 255 };
					numer[0] += 1.0 / expImage.at<Vec3b>(i + m, j + n)[0];
					numer[1] += 1.0 / expImage.at<Vec3b>(i + m, j + n)[1];
					numer[2] += 1.0 / expImage.at<Vec3b>(i + m, j + n)[2];
				}
			}
			Vec3b c;
			c[0] = (uchar)(ksize.width * ksize.height / numer[0]);
			c[1] = (uchar)(ksize.width * ksize.height / numer[1]);
			c[2] = (uchar)(ksize.width * ksize.height / numer[2]);
			dstImage.at<Vec3b>(i, j) = c;
		}
	}
	return dstImage;
}

Mat medianFilter(const Mat &srcImage, Size ksize) {
	Mat expImage, dstImage, imageROI;
	srcImage.copyTo(expImage);
	int a, b;
	a = (ksize.width - 1) / 2;
	b = (ksize.height - 1) / 2;
	//边界反射扩展，目标图像上的像素坐标对应于ROI的左上角
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_REFLECT);
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		imageROI = expImage(Rect(0, i, ksize.width, ksize.height));
		//将三通道的ROI分割为三个单通道图像
		Mat ROIplane[3];
		split(imageROI, ROIplane);
		//计算每个单通道图像的直方图
		Mat hist[3];
		int channel = 0;
		int histSize = 256;
		int dims = 1;
		float hranges[] = { 0, 255 };
		const float *ranges[] = { hranges };
		calcHist(&ROIplane[0], 1, &channel, Mat(), hist[0], dims, &histSize, ranges);
		calcHist(&ROIplane[1], 1, &channel, Mat(), hist[1], dims, &histSize, ranges);
		calcHist(&ROIplane[2], 1, &channel, Mat(), hist[2], dims, &histSize, ranges);
		for (int j = 0; j < dstImage.cols; j++) {
			if (j > 0) {
				//更新直方图，删去原ROI最左边的列，添加右边一列
				for (int k = 0; k < ksize.height; k++) {
					hist[0].at<float>(expImage.at<Vec3b>(i + k, j - 1)[0]) -= 1;
					hist[0].at<float>(expImage.at<Vec3b>(i + k, j + ksize.width - 1)[0]) += 1;
					hist[1].at<float>(expImage.at<Vec3b>(i + k, j - 1)[1]) -= 1;
					hist[1].at<float>(expImage.at<Vec3b>(i + k, j + ksize.width - 1)[1]) += 1;
					hist[2].at<float>(expImage.at<Vec3b>(i + k, j - 1)[2]) -= 1;
					hist[2].at<float>(expImage.at<Vec3b>(i + k, j + ksize.width - 1)[2]) += 1;
				}
			}
			//计算中值
			Vec3b m;
			int total = ksize.width * ksize.height;
			m[0] = calcMedium(hist[0], total);
			m[1] = calcMedium(hist[1], total);
			m[2] = calcMedium(hist[2], total);
			dstImage.at<Vec3b>(i, j) = m;
		}
	}
	return dstImage;
}

int main() {
	//设置随机数种子
	srand((unsigned)time(NULL));
	//读取原图
	Mat srcImage = imread("lena.jpg");
	imshow("Source Image", srcImage);
	//加入椒盐噪声
	Mat noiImage = addNoise(srcImage, 0.1);
	imshow("Noisy Image", noiImage);
	//算数均值滤波
	Mat aritImage = aritMeanFilter(noiImage, Size(3, 3));
	imshow("Arithmetic Mean Filtered Image", aritImage);
	//几何均值滤波
	Mat geoImage = geoMeanFilter(noiImage, Size(3, 3));
	imshow("Geometric Mean Filtered Image", geoImage);
	//谐波均值滤波
	Mat harImage = harMeanFilter(noiImage, Size(3, 3));
	imshow("Harmonic Mean Filtered Image", harImage);
	//中值滤波
	Mat midImage = medianFilter(noiImage, Size(3, 3));
	imshow("Median Filtered Image", midImage);
	waitKey(0);

	return 0;
}