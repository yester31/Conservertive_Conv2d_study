#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


void ValueCheck(vector<float> &ValueCheck_input,size_t Input_H, size_t Input_W) {

	size_t w_idx = 0;
	size_t ⁠h_idx = 0;
	size_t ⁠g_idx = 0;

	size_t temp = 0;


			for (⁠h_idx = 0; ⁠h_idx < Input_H; ⁠h_idx++)
			{
				temp = ⁠h_idx * Input_W;
				for (w_idx = 0; w_idx < Input_W; w_idx++)
				{
					⁠g_idx = w_idx + temp;

					cout << setw(5) << ValueCheck_input[⁠g_idx] << " ";
				}cout << endl;
			}cout << endl;
		}


void ZeroPadding2d(vector<float> &ZeroPadding_output, vector<float> &ZeroPadding_input, size_t Input_H, size_t Input_W, int TopPadingSize, int BottomPadingSize, int LeftPadingSize, int RightPadingSize)
{

	size_t w_idx = 0;
	size_t ⁠h_idx = 0;
	size_t ⁠g_idx = 0;
	size_t ⁠g_idx1 = 0;

	size_t temp = 0;
	size_t temp1 = 0;

	ZeroPadding_output.resize((Input_H + TopPadingSize + BottomPadingSize) * (Input_W + LeftPadingSize + RightPadingSize));

	for (⁠h_idx = 0; ⁠h_idx < Input_H; ⁠h_idx++)
	{
		temp = ⁠h_idx * Input_W;
		temp1 = (⁠h_idx + TopPadingSize) * (Input_W + RightPadingSize + LeftPadingSize) + LeftPadingSize;
		for (w_idx = 0; w_idx < Input_W; w_idx++)
		{
			⁠g_idx = w_idx + temp;

			⁠g_idx1 = w_idx + temp1;

			ZeroPadding_output[⁠g_idx1] = ZeroPadding_input[⁠g_idx] ;

		}
	}
}


int main() {

	cout << endl;

	size_t Input_H = 3;
	size_t Input_W = 3;

	vector<float> input(Input_H * Input_W);

	for (int i = 0; i < 9; i++) {
		input[i] = i;
	}

	ValueCheck(input, Input_H, Input_W);

	int TopPadingSize = 3;
	int BottomPadingSize = 2;
	int LeftPadingSize = 1;
	int RightPadingSize = 2;

	vector<float> Output_zero((Input_H + TopPadingSize + BottomPadingSize)*(Input_W + LeftPadingSize + RightPadingSize));

	ZeroPadding2d(Output_zero, input, Input_H, Input_W,  TopPadingSize, BottomPadingSize, LeftPadingSize, RightPadingSize);

	ValueCheck(Output_zero, Input_H + TopPadingSize + BottomPadingSize, Input_W + LeftPadingSize + RightPadingSize);



	return 0;
}