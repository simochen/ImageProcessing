#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <time.h>
#include <cmath>

using namespace cv;
using namespace std;

//���뽷������
Mat addNoise(const Mat &srcImage, double p) {
	Mat dstImage;
	srcImage.copyTo(dstImage);
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			//���� [0,1) �������
			double randnum;
			randnum = rand() / double(RAND_MAX);
			//�����������
			if (randnum < 0.5 * p) dstImage.at<Vec3b>(i, j) = { 0, 0, 0 };
			//����������
			else if (randnum < p) dstImage.at<Vec3b>(i, j) = { 255, 255, 255 };
		}
	}
	return dstImage;
}

//������ֵ�˲�
Mat aritMeanFilter(const Mat &srcImage, Size ksize) {
	Mat expImage, dstImage, imageROI;
	srcImage.copyTo(expImage);
	int a, b;
	a = (ksize.width - 1) / 2;
	b = (ksize.height - 1) / 2;
	//�߽練����չ��Ŀ��ͼ���ϵ����������Ӧ�ڴ�����������Ͻ�λ��
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_REFLECT);
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			imageROI = expImage(Rect(j, i, ksize.width, ksize.height));
			//����������ֵ
			Vec4b m = (Vec4b)mean(imageROI);
			dstImage.at<Vec3b>(i, j) = { m[0], m[1], m[2] };
		}
	}
	return dstImage;
}

//���ξ�ֵ�˲�
Mat geoMeanFilter(const Mat &srcImage, Size ksize) {
	Mat expImage, dstImage;
	srcImage.copyTo(expImage);
	int a, b;
	a = (ksize.width - 1) / 2;
	b = (ksize.height - 1) / 2;
	//�߽練����չ��Ŀ��ͼ���ϵ����������Ӧ��ROI�����Ͻ�
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_REFLECT);
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			//��ʼ���˻�
			Vec3d product = { 1, 1, 1 };
			for (int m = 0; m < ksize.height; m++) {
				for (int n = 0; n < ksize.width; n++) {
					//����������ȡ������Ϊ������
					if (expImage.at<Vec3b>(i + m, j + n)[0] == 0) expImage.at<Vec3b>(i + m, j + n) = {255, 255, 255};
					//�۳����ػҶ�ֵ
					product[0] *= expImage.at<Vec3b>(i + m, j + n)[0];
					product[1] *= expImage.at<Vec3b>(i + m, j + n)[1];
					product[2] *= expImage.at<Vec3b>(i + m, j + n)[2];
				}
			}
			//���㼸�ξ�ֵ
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

//г����ֵ�˲�
Mat harMeanFilter(const Mat &srcImage, Size ksize) {
	Mat expImage, dstImage;
	srcImage.copyTo(expImage);
	int a, b;
	a = (ksize.width - 1) / 2;
	b = (ksize.height - 1) / 2;
	//�߽練����չ��Ŀ��ͼ���ϵ����������Ӧ��ROI�����Ͻ�
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_REFLECT);
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			//��ʼ����ĸ���Ҷ�ֵ����֮�ͣ�
			Vec3d den = { 0, 0, 0 };
			for (int m = 0; m < ksize.height; m++) {
				for (int n = 0; n < ksize.width; n++) {
					//����������ȡ������Ϊ������
					if (expImage.at<Vec3b>(i + m, j + n)[0] == 0) expImage.at<Vec3b>(i + m, j + n) = {255, 255, 255};
					//�ۼӻҶ�ֵ����
					den[0] += 1.0 / expImage.at<Vec3b>(i + m, j + n)[0];
					den[1] += 1.0 / expImage.at<Vec3b>(i + m, j + n)[1];
					den[2] += 1.0 / expImage.at<Vec3b>(i + m, j + n)[2];
				}
			}
			//����г����ֵ
			Vec3b c;
			c[0] = (uchar)(ksize.width * ksize.height / den[0]);
			c[1] = (uchar)(ksize.width * ksize.height / den[1]);
			c[2] = (uchar)(ksize.width * ksize.height / den[2]);
			dstImage.at<Vec3b>(i, j) = c;
		}
	}
	return dstImage;
}

//��ֱ��ͼ�м�����ֵ
int calcMedium(const Mat &hist, int total) {
	//������ֵʱ��ֹͣ����ֵΪ���������� 1/2
	int stopCount = total / 2;
	float s = 0;
	for (int i = 0; i < hist.size[0]; i++) {
		s += hist.at<float>(i);
		if (s > stopCount) return i;
	}
}

//��ֵ�˲�
Mat medianFilter(const Mat &srcImage, Size ksize) {
	Mat expImage, dstImage, imageROI;
	srcImage.copyTo(expImage);
	int a, b;
	a = (ksize.width - 1) / 2;
	b = (ksize.height - 1) / 2;
	//�߽練����չ��Ŀ��ͼ���ϵ����������Ӧ��ROI�����Ͻ�
	copyMakeBorder(expImage, expImage, b, b, a, a, BORDER_REFLECT);
	dstImage = Mat(srcImage.size(), srcImage.type());
	for (int i = 0; i < dstImage.rows; i++) {
		imageROI = expImage(Rect(0, i, ksize.width, ksize.height));
		//����ͨ����ROI�ָ�Ϊ������ͨ��ͼ��
		Mat ROIplane[3];
		split(imageROI, ROIplane);
		//����ÿ����ͨ��ͼ���ֱ��ͼ
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
				//����ֱ��ͼ��ɾȥԭROI����ߵ��У�����ұ�һ��
				for (int k = 0; k < ksize.height; k++) {
					hist[0].at<float>(expImage.at<Vec3b>(i + k, j - 1)[0]) -= 1;
					hist[0].at<float>(expImage.at<Vec3b>(i + k, j + ksize.width - 1)[0]) += 1;
					hist[1].at<float>(expImage.at<Vec3b>(i + k, j - 1)[1]) -= 1;
					hist[1].at<float>(expImage.at<Vec3b>(i + k, j + ksize.width - 1)[1]) += 1;
					hist[2].at<float>(expImage.at<Vec3b>(i + k, j - 1)[2]) -= 1;
					hist[2].at<float>(expImage.at<Vec3b>(i + k, j + ksize.width - 1)[2]) += 1;
				}
			}
			//������ֵ
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
	//�������������
	srand((unsigned)time(NULL));
	//��ȡԭͼ
	Mat srcImage = imread("lena.jpg");
	imshow("Source Image", srcImage);
	//���뽷������
	Mat noiImage = addNoise(srcImage, 0.1);
	imshow("Noisy Image", noiImage);
	//������ֵ�˲�
	Mat aritImage = aritMeanFilter(noiImage, Size(3, 3));
	imshow("Arithmetic Mean Filtered Image", aritImage);
	//���ξ�ֵ�˲�
	Mat geoImage = geoMeanFilter(noiImage, Size(3, 3));
	imshow("Geometric Mean Filtered Image", geoImage);
	//г����ֵ�˲�
	Mat harImage = harMeanFilter(noiImage, Size(3, 3));
	imshow("Harmonic Mean Filtered Image", harImage);
	//��ֵ�˲�
	Mat midImage = medianFilter(noiImage, Size(3, 3));
	imshow("Median Filtered Image", midImage);
	waitKey(0);

	return 0;
}