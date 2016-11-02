#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;

#define RED 0
#define GREEN 1
#define BLUE 2

//灰度变换
//x：像素灰度
//color：RED、GREEN 或 BLUE
//maxVal：灰度最大值，由图像深度得到
//若图像深度为CV_8U，则maxVal=255；若图像深度为CV_32F，则maxVal=1
float trans(float x, int color, int maxVal = 1)
{
	//将输入灰度 x 缩放到 [0,1] 范围内
	x /= maxVal;
	if (x < 0) x = 0;
	if (x > 1) x = 1;
	float y;
	if (color == RED) {
		if (x < 0.5) y = 0;
		else if (x < 0.75) y = 4 * x - 2;
		else y = 1;
	}
	else if (color == GREEN) {
		if (x < 0.25) y = 4 * x;
		else if (x < 0.75) y = 1;
		else y = -4 * x + 4;
	}
	else if (color == BLUE){
		if (x < 0.25) y = 1;
		else if (x < 0.5) y = -4 * x + 2;
		else y = 0;
	}
	//若输入颜色不属于R、G、B，返回输入灰度
	else y = x;
	y *= maxVal;
	return y;
}

Mat colorProcess(const Mat &grayImage) {
	Mat plane[3];
	for (int k = 0; k < 3; k++) {
		plane[k] = Mat(grayImage.size(), grayImage.type());
		for (int i = 0; i < grayImage.rows; i++) {
			for (int j = 0; j < grayImage.cols; j++) {
				plane[k].at<uchar>(i, j) = trans((grayImage.at<uchar>(i, j)), k, 255);
			}
		}
	}
	Mat rgbImage;
	merge(plane, 3, rgbImage);
	return rgbImage;
}

int main()
{
	Mat grayImage, rgbImage;
	grayImage = imread("lena.bmp", 0);
	imshow("Gray Image", grayImage);
	rgbImage = colorProcess(grayImage);
	imshow("RGB Image", rgbImage);
	waitKey(0);
	return 0;
}