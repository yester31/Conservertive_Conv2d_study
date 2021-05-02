#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//이미지 데이터 및 이미지 이름 가져오기
vector<pair<Mat, string>> TraverseFilesUsingDFS(const string& folder_path)
{
	_finddata_t file_info;
	string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);
	vector<pair<Mat, string>> ImgBox;

	//If folder_path exsist, using any_file_pattern will find at least two files "." and "..",
	//of which "." means current dir and ".." means parent dir
	if (handle == -1)
	{
		cerr << "folder path not exist: " << folder_path << endl;
		exit(-1);
	}

	//iteratively check each file or sub_directory in current folder
	do
	{
		string file_name = file_info.name; //from char array to string

		//check whtether it is a sub direcotry or a file
		if (file_info.attrib & _A_SUBDIR)
		{
			if (file_name != "." && file_name != "..")
			{
				string sub_folder_path = folder_path + "\\" + file_name;
				TraverseFilesUsingDFS(sub_folder_path);
				cout << "a sub_folder path: " << sub_folder_path << endl;
			}
		}
		else  //cout << "file name: " << file_name << endl;
		{
			int npo1 = file_name.find('_') + 1;
			int npo2 = file_name.find('.');
			int npo3 = npo2 - npo1;
			string newname = file_name.substr(npo1, npo3);
			string sub_folder_path2 = folder_path + "\\" + file_name;
			Mat img = imread(sub_folder_path2);
			ImgBox.push_back({ { img }, { newname } });
		}
	} while (_findnext(handle, &file_info) == 0);

	//
	_findclose(handle);
	return ImgBox;
}

float ActivationTanh(float x) {
	return tanh(x);
}

float ActivationSigmoid(float x) {
	return (1.f / (exp(-x) + 1.f));
}

float ActivationReLU(float x) {
	return (x > 0.f ? x : 0.f);
}


void ValueCheck(vector<float> &ValueCheck_input, int Input_N, int Input_C, int Input_H, int Input_W, int offset = 0) {
	if (offset == 1) {
		Input_N = 1;
	}

	int w_idx = 0;
	int ⁠h_idx = 0;
	int ⁠c_idx = 0;
	int ⁠n_idx = 0;
	int ⁠g_idx = 0;
	int temp1 = 0;
	int temp2 = 0;
	int temp3 = 0;
	int temp4 = 0;

	temp1 = Input_W * Input_H * Input_C;
	for (⁠n_idx = 0; ⁠n_idx < Input_N; ⁠n_idx++)
	{
		temp2 = ⁠n_idx * temp1;
		for (⁠c_idx = 0; ⁠c_idx < Input_C; ⁠c_idx++)
		{
			temp3 = ⁠c_idx * Input_W * Input_H + temp2;
			for (⁠h_idx = 0; ⁠h_idx < Input_H; ⁠h_idx++)
			{
				temp4 = ⁠h_idx * Input_W + temp3;
				for (w_idx = 0; w_idx < Input_W; w_idx++)
				{
					⁠g_idx = w_idx + temp4;
					cout << setw(5) << ValueCheck_input[⁠g_idx] << " ";
				}cout << endl;
			}cout << endl; cout << endl;
		}
	}
}

void ZeroPadding(vector<float> &ZeroPaddingOutput, vector<float> &ZeroPaddingInput, int Input_N, int Input_C, int Input_H, int Input_W, int TopPadingSize, int BottomPadingSize, int LeftPadingSize, int RightPadingSize) {

	int ⁠g_idx = 0;
	int temp1 = 0;
	int temp2 = 0;
	int temp3 = 0;
	int temp4 = 0;
	int g_idx_Output = 0;
	int temp4o = 0;
	int temp3o = 0;
	int temp1o = 0;
	int temp2o = 0;
	cout << "===== Zero Padding ===== \n";

	ZeroPaddingOutput.resize(Input_N * Input_C*(Input_H + TopPadingSize + BottomPadingSize)*(Input_W + LeftPadingSize + RightPadingSize));

	temp1 = Input_W * Input_H * Input_C;
	temp1o = (Input_H + TopPadingSize + BottomPadingSize)*(Input_W + LeftPadingSize + RightPadingSize)* Input_C;
	for (int ⁠n_idx = 0; ⁠n_idx < Input_N; ⁠n_idx++)
	{
		temp2 = ⁠n_idx * temp1;
		temp2o = ⁠n_idx * temp1o;
		for (int ⁠c_idx = 0; ⁠c_idx < Input_C; ⁠c_idx++)
		{
			temp3 = ⁠c_idx * Input_W * Input_H + temp2;
			temp3o = ⁠c_idx * (Input_W + LeftPadingSize + RightPadingSize) * (Input_H + TopPadingSize + BottomPadingSize) + temp2o;
			for (int ⁠h_idx = 0; ⁠h_idx < Input_H; ⁠h_idx++)
			{
				temp4 = ⁠h_idx * Input_W + temp3;
				temp4o = (⁠h_idx + TopPadingSize)*(Input_W + LeftPadingSize + RightPadingSize) + LeftPadingSize + temp3o;

				for (int w_idx = 0; w_idx < Input_W; w_idx++)
				{
					⁠g_idx = w_idx + temp4;
					g_idx_Output = w_idx + temp4o;
					ZeroPaddingOutput[g_idx_Output] = ZeroPaddingInput[⁠g_idx];
				}
			}
		}
	}
}

void Activation(vector<float> &ActivationInput) {
	cout << "===== Activation ===== \n";
	for (int i = 0; i < ActivationInput.size(); i++)
	{
		ActivationInput[i] = ActivationSigmoid(ActivationInput[i]);
		//ActivationInput[i] = ActivationTanh(ActivationInput[i]);
		//ActivationInput[i] = ActivationReLU(ActivationInput[i]);
	}
}

void Convolution(vector<float> &Conv_output, vector<float> &Conv_input, vector<float> &kernel,
	int kernelSize,int stride, int Input_N, int Input_C, int Input_H, int Input_W, int Ouput_C) {
	int OutputHeightSize = ((Input_H - kernelSize) / stride) + 1;
	int OutputWidthSize = ((Input_W - kernelSize) / stride) + 1;
	//Conv_output.resize(Input_N * Ouput_C * OutputHeightSize * OutputWidthSize);

	int ⁠g_idx_i = 0;
	int g_idx_o = 0;
	int g_idx_k = 0;
	int temp1k = 0;
	int temp2k = 0;
	int temp3k = 0;
	int temp4k = 0;
	int temp4o = 0;
	int temp3o = 0;
	int temp1o = 0;
	int temp2o = 0;
	int temp4i = 0;
	int temp3i = 0;
	int temp1i = 0;
	int temp2i = 0;

	//cout << "===== Convolution ===== \n";
	temp1i = Input_H * Input_W *Input_C;
	temp1o = OutputHeightSize * OutputWidthSize * Ouput_C;
	temp1k = kernelSize * kernelSize * Input_C;
	for (int ⁠n_idx = 0; ⁠n_idx < Input_N; ⁠n_idx++)
	{
		temp2i = ⁠n_idx * temp1i;
		temp2o = ⁠n_idx * temp1o;
		for (int k_idx = 0; k_idx < Ouput_C; k_idx++)
		{
			temp2k = k_idx * temp1k;
			temp3o = k_idx * OutputHeightSize * OutputWidthSize + temp2o;
			for (int ⁠c_idx = 0; ⁠c_idx < Input_C; ⁠c_idx++)
			{
				temp3i = ⁠c_idx * Input_W * Input_H + temp2i;
				temp3k = ⁠c_idx * kernelSize * kernelSize + temp2k;
				for (int rowStride = 0; rowStride < OutputHeightSize; rowStride++) {
					temp4o = rowStride * OutputWidthSize + temp3o;
					for (int colStride = 0; colStride < OutputWidthSize; colStride++) {
						float sum = 0;
						g_idx_o = colStride + temp4o;
						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							temp4i = x * Input_W + temp3i;
							temp4k = (x - rowStride * stride) * kernelSize + temp3k;
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								⁠g_idx_i = y + temp4i;
								g_idx_k = (y - colStride * stride) + temp4k;
								sum += Conv_input[⁠g_idx_i] * kernel[g_idx_k];
							}
						}
						Conv_output[g_idx_o] += sum;
					}
				}
			}
 		}
	}
}





int main()
{
	cout << "4D([N][C][H][W]) Convolutions ! \n\n";
	cout << "===== 1. Image loading ===== \n";
	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS("C:\\cifar\\test10");// 이미지가 저장되어 있는 폴더 경로


	//입력변수
	int Input_N = ImgBox.size(); // 10
	int Input_C = ImgBox[0].first.channels(); // 3
	int Input_H = ImgBox[0].first.rows;//H 32 
	int Input_W = ImgBox[0].first.cols;//W 32

	int InputDataSize = Input_N * Input_C * Input_H * Input_W;

	vector<float> Input_Vec(InputDataSize);

	int w_idx = 0;
	int ⁠h_idx = 0;
	int ⁠c_idx = 0;
	int ⁠n_idx = 0;
	int ⁠g_idx = 0;
	int temp1 = 0;
	int temp2 = 0;
	int temp3 = 0;
	int temp4 = 0;
	int temp5 = 0;
	cout << "===== 2. zeropadding and mat -> vector  ===== \n";

	// mat 형식 - > 4차 행렬
	temp1 = Input_W * Input_H * Input_C;
	for (⁠n_idx = 0; ⁠n_idx < Input_N; ⁠n_idx++)
	{
		unsigned char* temp = ImgBox[⁠n_idx].first.data;
		temp2 = ⁠n_idx * temp1;
		for (⁠c_idx = 0; ⁠c_idx < Input_C; ⁠c_idx++)
		{
			temp3 = ⁠c_idx * Input_W * Input_H + temp2;
			for (⁠h_idx = 0; ⁠h_idx < Input_H; ⁠h_idx++)
			{
				temp4 = ⁠h_idx * Input_W + temp3;
				temp5 = Input_C * Input_W * ⁠h_idx;
				for (w_idx = 0; w_idx < Input_W; w_idx++)
				{
					⁠g_idx = w_idx + temp4;
					Input_Vec[⁠g_idx] = temp[temp5 + Input_C * w_idx + ⁠c_idx];
				}
			}
		}
	}

	cout << "===== Input Value check  ===== \n";
	ValueCheck(Input_Vec, Input_N, Input_C, Input_H, Input_W, 1);



	// 커널, window, weight
	//cout << "===== 3. weight(filter) generation ===== \n";
	//vector<float> kernel = InitWeightsXavier(3, 2, 3, 3);


	//cout << "===== weight check ===== \n";
	//ValueCheck(kernel, 0);
	vector<float> kernel(4*3*3*3);
	for (int i = 0; i < kernel.size(); i++) {
		kernel[i] = 1;
	}

	//Activation(kernel);
	//ValueCheck(kernel, 0);

	cout << "===== 4. Convolution ===== \n";
	//ValueCheck(input_vec);

	int TopPadingSize = 1;
	int BottomPadingSize = 1;
	int LeftPadingSize = 1;
	int RightPadingSize = 1;

	vector<float> Input_Vec_ZeroPading(Input_N*Input_C*(Input_H + TopPadingSize + BottomPadingSize)*(Input_W + LeftPadingSize + RightPadingSize));

	ZeroPadding(Input_Vec_ZeroPading, Input_Vec, Input_N, Input_C, Input_H, Input_W, TopPadingSize, BottomPadingSize, LeftPadingSize, RightPadingSize);

	ValueCheck(Input_Vec_ZeroPading, Input_N, Input_C, Input_H + TopPadingSize + BottomPadingSize, Input_W + LeftPadingSize + RightPadingSize, 1);

	int oc = 4 ;
	int kernelSize = 3;
	int stride = 1; 
	int OutputHeightSize = ((Input_H + TopPadingSize + BottomPadingSize - kernelSize) / stride) + 1;
	int OutputWidthSize = ((Input_W + LeftPadingSize + RightPadingSize - kernelSize) / stride) + 1;

	vector<float> output_1(Input_N*oc*OutputHeightSize*OutputWidthSize);

	Convolution(output_1, Input_Vec_ZeroPading, kernel, kernelSize, stride, Input_N, Input_C, Input_H + TopPadingSize + BottomPadingSize, Input_W + LeftPadingSize + RightPadingSize, oc);

	ValueCheck(output_1, Input_N, 4, 32, 32, 1);


	return 0;
}