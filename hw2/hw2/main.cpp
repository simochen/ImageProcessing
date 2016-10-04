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

Mat calcSpectrum(const Mat &srcImage)
{
	//����ͨ��ͼ��ָ��������ͨ��ͼ��
	Mat imgArray[2];
	split(srcImage, imgArray);
	//�����ֵ
	Mat dstImage;
	magnitude(imgArray[0], imgArray[1], dstImage);
	//ȡlog(1+|F|)
	dstImage += Scalar(1);
	log(dstImage, dstImage);
	//��һ��
	normalize(dstImage, dstImage, 0, 1, NORM_MINMAX);
	return dstImage;
}

Mat filter(const Mat &srcImage)
{
	//������˹�˲���
	Mat BF(srcImage.size(), CV_32F);
	Mat dstImage(srcImage.size(), CV_32F);
	int Dl = 1500, Dh = 40000;
	int n = 4;
	float lp, hp;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			//��ͨ����
			lp = 1 / (1 + pow(((pow((i + 1 - srcImage.rows / 2), 2) + pow((j + 1 - srcImage.cols / 2), 2)) / (Dl ^ 2)), n));
			//��ͨ����
			hp = 1 / (1 + pow((Dh ^ 2) / ((pow((i + 1 - srcImage.rows / 2), 2) + pow((j + 1 - srcImage.cols / 2), 2))), n));
			BF.at<float>(i, j) = lp + hp;
		}
	}
	//���������˲�����ʹ�������˥����10%��������ȫ��ϣ�
	normalize(BF, BF, 0.1, 1, CV_MINMAX);
	//Ƶ���˲�
	multiply(srcImage, BF, dstImage);
	imshow("Butterworth Filter", BF);
	return dstImage;
}

Mat performIDFT(const Mat &srcImage)
{
	//����ͨ��ͼ��ָ��������ͨ��ͼ��
	Mat imgArray[2];
	split(srcImage, imgArray);
	//Ƶ���˲�
	imgArray[0] = filter(imgArray[0]);
	imgArray[1] = filter(imgArray[1]);
	//���˲����ʵ�����鲿�ϳ�һ��2ͨ����ͼ��
	Mat dstImage;
	merge(imgArray, 2, dstImage);
	//idft��ȡʵ��
	idft(dstImage, dstImage, DFT_REAL_OUTPUT);
	//���Ļ�
	centralize(dstImage);
	//��һ��
	normalize(dstImage, dstImage, 0, 1, CV_MINMAX);
	return dstImage;
}

int main()
{
	//��ȡԭͼ
	Mat srcImage = imread("origin.jpg", 0);
	imshow("Source Image", srcImage);
	if (!srcImage.data)
	{
		cout << "fail to load image!" << endl;
	}
	//����ԭͼƵ��
	Mat srcDFT, srcSpec;
	srcDFT = performDFT(srcImage);
	srcSpec = calcSpectrum(srcDFT);
	imshow("Source Spectrum Image", srcSpec);
	//��ȡ����ͼ��
	Mat noiImage = imread("addnoise.jpg", 0);
	imshow("Noisy Image", noiImage);
	if (!noiImage.data)
	{
		cout << "fail to load image!" << endl;
	}
	//�������ͼƵ��
	Mat noiDFT, noiSpec, filSpec;
	noiDFT = performDFT(noiImage);
	noiSpec = calcSpectrum(noiDFT);
	imshow("Noisy Spectrum Image", noiSpec);
	//Ƶ���˲�
	filSpec = filter(noiSpec);
	imshow("Filtered Spectrum", filSpec);
	Mat dstImage = performIDFT(noiDFT);
	//�ü���չ�ı߽�
	dstImage = dstImage(Rect(0, 0, noiImage.cols, noiImage.rows));
	imshow("Filtered Image", dstImage);
	waitKey(0);
	return 0;
}