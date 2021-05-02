#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


void ValueCheck3D(vector<float> &ValueCheck_input, size_t Input_C, size_t Input_H, size_t Input_W) {
	size_t c_idx = 0;
	size_t w_idx = 0;
	size_t ⁠h_idx = 0;
	size_t ⁠g_idx = 0;

	size_t temp = 0;
	size_t temp2 = 0;

	for (c_idx = 0; c_idx < Input_C; c_idx++)
	{
		temp2 = c_idx * Input_W * Input_H;
		for (⁠h_idx = 0; ⁠h_idx < Input_H; ⁠h_idx++)
		{
			temp = ⁠h_idx * Input_W + temp2;
			for (w_idx = 0; w_idx < Input_W; w_idx++)
			{
				⁠g_idx = w_idx + temp;

				cout << setw(5) << ValueCheck_input[⁠g_idx] << " ";
		}cout << endl;
	}cout << endl; cout << endl;
	}
}


void ZeroPadding3d(vector<float> &ZeroPadding_output, vector<float> &ZeroPadding_input, size_t Input_C, size_t Input_H, size_t Input_W, int TopPadingSize, int BottomPadingSize, int LeftPadingSize, int RightPadingSize)
{
	size_t c_idx = 0;
	size_t w_idx = 0;
	size_t ⁠h_idx = 0;
	size_t ⁠g_idx = 0;
	size_t ⁠g_idx1 = 0;

	size_t temp = 0;
	size_t temp1 = 0;
	size_t temp2 = 0;
	size_t temp3 = 0;
	ZeroPadding_output.resize(Input_C*(Input_H + TopPadingSize + BottomPadingSize) * (Input_W + LeftPadingSize + RightPadingSize));

	for (c_idx = 0; c_idx < Input_C; c_idx++)
	{
		temp2 = c_idx * Input_W * Input_H;
		temp3 = c_idx * (Input_W + LeftPadingSize + RightPadingSize) * (Input_H + TopPadingSize + BottomPadingSize);
	for (⁠h_idx = 0; ⁠h_idx < Input_H; ⁠h_idx++)
	{
		temp = ⁠h_idx * Input_W + temp2;
		temp1 = (⁠h_idx + TopPadingSize) * (Input_W + RightPadingSize + LeftPadingSize) + LeftPadingSize+ temp3;
		for (w_idx = 0; w_idx < Input_W; w_idx++)
		{
			⁠g_idx = w_idx + temp;

			⁠g_idx1 = w_idx + temp1;

			ZeroPadding_output[⁠g_idx1] = ZeroPadding_input[⁠g_idx];
		}
		}
	}
}


int main() {

	cout << endl;

	size_t Input_C = 3;
	size_t Input_H = 3;
	size_t Input_W = 3;

	vector<float> input(Input_H * Input_W*Input_C);

	for (int i = 0; i < Input_H * Input_W * Input_C; i++) {
		input[i] = i;
	}

	ValueCheck3D(input, Input_C, Input_H, Input_W);

	int TopPadingSize = 3;
	int BottomPadingSize = 2;
	int LeftPadingSize = 1;
	int RightPadingSize = 2;

	vector<float> Output_zero((Input_H + TopPadingSize + BottomPadingSize)*(Input_W + LeftPadingSize + RightPadingSize));

	ZeroPadding3d(Output_zero, input, Input_C, Input_H, Input_W, TopPadingSize, BottomPadingSize, LeftPadingSize, RightPadingSize);

	ValueCheck3D(Output_zero, Input_C, Input_H + TopPadingSize + BottomPadingSize, Input_W + LeftPadingSize + RightPadingSize);



	return 0;
}