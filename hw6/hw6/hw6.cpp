#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

//��ӡ����
void printMat(Mat src)
{
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			cout << src.at<double>(i, j) << " ";
		}
		cout << endl;
	}
	cout << endl;
}

//�Ծ���ֵ������������
void roundMat(Mat src)
{
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			src.at<double>(i, j) = round(src.at<double>(i, j));
		}
	}
}

//�Ծ���ֵ������λ
//c����λֵ
void shiftMat(Mat src, int c)
{
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			src.at<double>(i, j) += c;
		}
	}
}

int main()
{
	//JPEG ��׼����
	Mat Z = (Mat_<double>(8, 8) <<
		16, 11, 10, 16, 24, 40, 51, 61,
		12, 12, 14, 19, 26, 58, 60, 55,
		14, 13, 16, 24, 40, 57, 69, 56,
		14, 17, 22, 29, 51, 87, 80, 62,
		18, 22, 37, 56, 68, 109, 103, 77,
		24, 35, 55, 64, 81, 104, 113, 92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103, 99);
	
	//ԭʼ����
	Mat src = (Mat_<double>(8, 8) << 
		52, 55, 61, 66, 70, 61, 64, 73,
		63, 59, 66, 90, 109, 85, 69, 72,
		62, 59, 68, 113, 144, 104, 66, 73,
		63, 58, 71, 122, 154, 106, 70, 69,
		67, 61, 68, 104, 126, 88, 68, 70,
		79, 65, 60, 70, 77, 68, 58, 75,
		85, 71, 64, 59, 55, 61, 65, 83,
		87, 79, 69, 68, 65, 76, 78, 94);
	cout << "ԭʼ����" << endl;
	printMat(src);
	
	Mat T, diff;
	cout << "������λ -128 ���Ҷȼ���" << endl;
	src.copyTo(T);
	shiftMat(T, -128);
	printMat(T);
	
	cout << "DCT �任���������룺" << endl;
	dct(T, T);
	roundMat(T);
	printMat(T);
	
	cout << "����JPEG��׼����������" << endl;
	divide(T, Z, T);
	roundMat(T);
	printMat(T);
	
	cout << "ȥ��񻯣�" << endl;
	multiply(T, Z, T);
	printMat(T);
	
	cout << "IDCT �任���������룺" << endl;
	idct(T, T);
	roundMat(T);
	printMat(T);
	
	cout << "������λ +128 ���Ҷȼ���" << endl;
	shiftMat(T, 128);
	printMat(T);
	
	cout << "�ؽ�������ԭ����֮�" << endl;
	diff = T - src;
	printMat(diff);

	double m = mean(abs(diff))[0];
	cout << "������ֵƽ����" << m << endl;
	double rmse = sqrt(mean(diff.mul(diff))[0]);
	cout << "��������" << rmse << endl;
	getchar();
	return 0;
}