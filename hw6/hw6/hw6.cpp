#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

//打印矩阵
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

//对矩阵值进行四舍五入
void roundMat(Mat src)
{
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			src.at<double>(i, j) = round(src.at<double>(i, j));
		}
	}
}

//对矩阵值进行移位
//c：移位值
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
	//JPEG 标准阵列
	Mat Z = (Mat_<double>(8, 8) <<
		16, 11, 10, 16, 24, 40, 51, 61,
		12, 12, 14, 19, 26, 58, 60, 55,
		14, 13, 16, 24, 40, 57, 69, 56,
		14, 17, 22, 29, 51, 87, 80, 62,
		18, 22, 37, 56, 68, 109, 103, 77,
		24, 35, 55, 64, 81, 104, 113, 92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103, 99);
	
	//原始矩阵
	Mat src = (Mat_<double>(8, 8) << 
		52, 55, 61, 66, 70, 61, 64, 73,
		63, 59, 66, 90, 109, 85, 69, 72,
		62, 59, 68, 113, 144, 104, 66, 73,
		63, 58, 71, 122, 154, 106, 70, 69,
		67, 61, 68, 104, 126, 88, 68, 70,
		79, 65, 60, 70, 77, 68, 58, 75,
		85, 71, 64, 59, 55, 61, 65, 83,
		87, 79, 69, 68, 65, 76, 78, 94);
	cout << "原始矩阵：" << endl;
	printMat(src);
	
	Mat T, diff;
	cout << "像素移位 -128 个灰度级：" << endl;
	src.copyTo(T);
	shiftMat(T, -128);
	printMat(T);
	
	cout << "DCT 变换并四舍五入：" << endl;
	dct(T, T);
	roundMat(T);
	printMat(T);
	
	cout << "采用JPEG标准阵列量化：" << endl;
	divide(T, Z, T);
	roundMat(T);
	printMat(T);
	
	cout << "去规格化：" << endl;
	multiply(T, Z, T);
	printMat(T);
	
	cout << "IDCT 变换并四舍五入：" << endl;
	idct(T, T);
	roundMat(T);
	printMat(T);
	
	cout << "像素移位 +128 个灰度级：" << endl;
	shiftMat(T, 128);
	printMat(T);
	
	cout << "重建矩阵与原矩阵之差：" << endl;
	diff = T - src;
	printMat(diff);

	double m = mean(abs(diff))[0];
	cout << "误差绝对值平均：" << m << endl;
	double rmse = sqrt(mean(diff.mul(diff))[0]);
	cout << "均方根误差：" << rmse << endl;
	getchar();
	return 0;
}