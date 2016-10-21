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
	//�����ѵ�DFT�ߴ�
	int m, n;
	m = getOptimalDFTSize(srcImage.rows);
	n = getOptimalDFTSize(srcImage.cols);
	//�߽���չ
	Mat dstImage;
	srcImage.convertTo(dstImage, CV_32F);
	copyMakeBorder(dstImage, dstImage, 0, m - srcImage.rows, 0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
	//���Ļ�
	centralize(dstImage);
	//�½���ҳ�ĸ�����array�����е�һҳ�����Ļ����ͼ���ʼ�����ڶ�ҳ��ʼ��Ϊ0 
	Mat imgArray[] = { Mat_<float>(dstImage), Mat::zeros(dstImage.size(), CV_32F) };
	//����ҳ�ϳ�һ��2ͨ����ͼ��
	merge(imgArray, 2, dstImage);
	//dft������Ҷ�任���Ϊ������ͨ��1�����ʵ����ͨ��2������鲿��
	dft(dstImage, dstImage);
	return dstImage;
}

Mat performIDFT(const Mat &srcImage)
{
	Mat dstImage;
	//idft��ȡʵ��
	idft(srcImage, dstImage, DFT_REAL_OUTPUT);
	//���Ļ�
	centralize(dstImage);
	//��һ��
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
	//��ȡԭͼ
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