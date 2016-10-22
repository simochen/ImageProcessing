#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <cmath>

using namespace cv;
using namespace std;

#define PI atan(1)*4

//��ɢ����Ҷ�任
Mat performDFT(const Mat &srcImage)
{
	//�����ѵ�DFT�ߴ�
	int m, n;
	m = getOptimalDFTSize(srcImage.rows);
	n = getOptimalDFTSize(srcImage.cols);
	//�߽���չ
	Mat dstImage;
	srcImage.convertTo(dstImage, CV_32F);
	copyMakeBorder(dstImage, dstImage, 0, m - srcImage.rows, 0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
	//�½���ҳ�ĸ�����array�����е�һҳ�����Ļ����ͼ���ʼ�����ڶ�ҳ��ʼ��Ϊ0 
	Mat imgArray[] = { Mat_<float>(dstImage), Mat::zeros(dstImage.size(), CV_32F) };
	//����ҳ�ϳ�һ��2ͨ����ͼ��
	merge(imgArray, 2, dstImage);
	//dft������Ҷ�任���Ϊ������ͨ��1�����ʵ����ͨ��2������鲿��
	dft(dstImage, dstImage);
	return dstImage;
}

//��ɢ����Ҷ��任
Mat performIDFT(const Mat &srcImage)
{
	Mat dstImage;
	//idft��ȡʵ��
	idft(srcImage, dstImage, DFT_REAL_OUTPUT);
	//��һ��
	normalize(dstImage, dstImage, 0, 1, CV_MINMAX);
	return dstImage;
}

// �˶�ģ��
Mat motionBlur(Size msize, double a, double b, double T) {
	//����һ��2ͨ��32λ�������ľ���ͨ��1���ʵ����ͨ��2����鲿
	Mat degMat = Mat(msize, CV_32FC2);
	double k;
	for (int i = 0; i < msize.width; i++) {
		for (int j = 0; j < msize.height; j++) {
			k = 2 * PI * (i * a + j * b);
			//�����ĸΪ0�����
			if (k == 0) k = 0.0001;
			degMat.at<Vec2f>(i, j)[0] = T * sin(k) / k;
			degMat.at<Vec2f>(i, j)[1] = T * (cos(k) - 1) / k;
		}
	}
	return degMat;
}

//ά���˲�
Mat wienerFilter(const Mat &blurMat, const Mat &degMat, double K = 0.01) {
	Mat blur_plane[2], deg_plane[2], res_plane[2];
	//�ָ�2ͨ���ľ�����ȡʵ�����鲿
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
	//��ȡģ��ͼ��
	Mat blurImage;
	blurImage = imread("blur.jpg");
	imshow("Blur Image", blurImage);
	//�ָ�RGB��ͨ��ͼ��
	Mat blur_plane[3], blur_spec[3];
	split(blurImage, blur_plane);
	//��ɢ����Ҷ�任
	blur_spec[0] = performDFT(blur_plane[0]);
	blur_spec[1] = performDFT(blur_plane[1]);
	blur_spec[2] = performDFT(blur_plane[2]);
	//�����˶�ģ���˻�����
	Mat H;
	H = motionBlur(Size(blur_spec[0].cols, blur_spec[0].rows), 0.0005, -0.0001, 1);
	//ά���˲�������Ҷ��任
	Mat res_plane[3];
	res_plane[0] = performIDFT(wienerFilter(blur_spec[0], H, 0.02));
	res_plane[1] = performIDFT(wienerFilter(blur_spec[1], H, 0.02));
	res_plane[2] = performIDFT(wienerFilter(blur_spec[2], H, 0.02));
	//�ϳ���ͨ��ͼ��
	Mat resImage;
	merge(res_plane, 3, resImage);
	imshow("Restored Image", resImage);
	
	waitKey(0);

	return 0;
}