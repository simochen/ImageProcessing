#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	Mat srcImage = imread("orange.jpg", 0);
	imshow("orange", srcImage);
	if (!srcImage.data)
	{
		cout << "fail to load image!" << endl;
	}
	waitKey(0);
}
