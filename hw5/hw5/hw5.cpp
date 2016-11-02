#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;

#define RED 0
#define GREEN 1
#define BLUE 2

//�Ҷȱ任
//x�����ػҶ�
//color��RED��GREEN �� BLUE
//maxVal���Ҷ����ֵ����ͼ����ȵõ�
//��ͼ�����ΪCV_8U����maxVal=255����ͼ�����ΪCV_32F����maxVal=1
float trans(float x, int color, int maxVal = 1)
{
	//������Ҷ� x ���ŵ� [0,1] ��Χ��
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
	//��������ɫ������R��G��B����������Ҷ�
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