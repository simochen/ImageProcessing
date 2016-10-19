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
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_CONSTANT, Scalar::all(0));
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			imageROI = expImage(Rect(j, i, ksize.width, ksize.height));
			Vec4b m = (Vec4b)mean(imageROI);
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
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_CONSTANT, Scalar::all(1));
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			Vec3d product = {1, 1, 1};
			for (int m = 0; m < ksize.height; m++) {
				for (int n = 0; n < ksize.width; n++) {
					Vec3b a = { 1, 1, 1 };
					if (expImage.at<Vec3b>(i + m, j + n)[0] != 0) a = expImage.at<Vec3b>(i + m, j + n);
					product[0] *= a[0];
					product[1] *= a[1];
					product[2] *= a[2];
				}
			}
			Vec3b c;
			double b = 1.0 / (ksize.width * ksize.height);
			c[0] = (uchar)pow(product[0], b);
			c[1] = (uchar)pow(product[1], b);
			c[2] = (uchar)pow(product[2], b);
			dstImage.at<Vec3b>(i, j) = { c[0], c[1], c[2] };
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
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_CONSTANT, Scalar::all(0));
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			imageROI = expImage(Rect(j, i, ksize.width, ksize.height));
			Mat mv[3];
			split(imageROI, mv);
			Vec3b m;
			m[0] = (uchar)((ksize.width * ksize.height) / sum(1.0 / (mv[0] + 1))[0]);
			m[1] = (uchar)((ksize.width * ksize.height) / sum(1.0 / (mv[1] + 1))[0]);
			m[2] = (uchar)((ksize.width * ksize.height) / sum(1.0 / (mv[2] + 1))[0]);
			dstImage.at<Vec3b>(i, j) = m;
		}
	}
	return dstImage;
}

int main() {
	srand((unsigned)time(NULL));
	Mat srcImage = imread("lena.jpg");
	imshow("Source Image", srcImage);
	Mat noiImage = addNoise(srcImage, 0.1);
	imshow("Noisy Image", noiImage);
	Mat aritImage = aritMeanFilter(noiImage, Size(3, 3));
	imshow("Arithmetry Mean Filtered Image", aritImage);
	Mat geoImage = harMeanFilter(noiImage, Size(3, 3));
	imshow("Geometry Mean Filtered Image", geoImage);
	waitKey(0);
	return 0;
}