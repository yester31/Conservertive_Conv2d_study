#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


/***************************************************************************
	일단  winograd convolution 계산과 비교하기 위한 기존 구조 convolution
*****************************************************************************/

//이미지 데이터 및 이미지 이름 가져오기
vector<pair<Mat, string>> TraverseFilesUsingDFS(const string& folder_path)
{
	_finddata_t file_info;
	string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);
	vector<pair<Mat, string>> imgBox;

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
			imgBox.push_back({ { img }, { newname } });
		}
	} while (_findnext(handle, &file_info) == 0);

	//
	_findclose(handle);
	return imgBox;
}

float activationTanh(float x) {
	return tanh(x);
}

float activationSigmoid(float x) {
	return (1.f / (exp(-x) + 1.f));
}

float activationReLU(float x) {
	return (x > 0.f ? x : 0.f);
}

void valueCheck(vector<float>& valueCheckInput, int input_n, int input_c, int input_h, int input_w, int offset = 0) {
	if (offset == 1) { input_n = 1; }

	int temp1 = input_w * input_h * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;
					cout << setw(5) << valueCheckInput[g_idx] << " ";
				}cout << endl;
			}cout << endl; cout << endl;
		}
	}
}

void valueCheck(vector<float>& valueCheckInput, int input_n, int input_c, int offset = 0) {
	if (offset == 1) { input_n = 1; }

	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int	temp2 = ⁠n_idx * input_c;

		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int ⁠g_idx = ⁠c_idx + temp2;
			cout << setw(5) << valueCheckInput[⁠g_idx] << " ";
		}cout << endl;
	}cout << endl; cout << endl;
}

void zeroPadding(vector<float>& zeroPaddingOutput, vector<float>& zeroPaddingInput, int input_n, int input_c, int input_h, int input_w, int topPadingSize, int bottomPadingSize, int leftPadingSize, int rightPadingSize) {
	//cout << "===== Zero Padding ===== \n";
	//zeroPaddingOutput.resize(input_n * input_c*(input_h + topPadingSize + bottomPadingSize)*(input_w + leftPadingSize + rightPadingSize));
	int temp1 = input_w * input_h * input_c;
	int temp1o = (input_h + topPadingSize + bottomPadingSize) * (input_w + leftPadingSize + rightPadingSize) * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		int temp2o = ⁠n_idx * temp1o;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			int temp3o = ⁠c_idx * (input_w + leftPadingSize + rightPadingSize) * (input_h + topPadingSize + bottomPadingSize) + temp2o;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				int temp4o = (⁠h_idx + topPadingSize) * (input_w + leftPadingSize + rightPadingSize) + leftPadingSize + temp3o;

				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int ⁠g_idx = w_idx + temp4;
					int g_idx_Output = w_idx + temp4o;
					zeroPaddingOutput[g_idx_Output] = zeroPaddingInput[⁠g_idx];
				}
			}
		}
	}
}



void convolution(vector<float>& convOutput, vector<float>& convInput, vector<float>& kernel, int kernelSize, int stride, int input_n, int input_c, int input_h, int input_w, int ouput_c) {
	int outputHeightSize = ((input_h - kernelSize) / stride) + 1;
	int outputWidthSize = ((input_w - kernelSize) / stride) + 1;
	//Conv_output.resize(input_n * Ouput_C * outputHeightSize * outputHeightSize);
	//cout << "===== Convolution ===== \n";

	int temp1i = input_h * input_w * input_c;
	int temp1o = outputHeightSize * outputWidthSize * ouput_c;
	int temp1k = kernelSize * kernelSize * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2i = ⁠n_idx * temp1i;
		int temp2o = ⁠n_idx * temp1o;
		for (int k_idx = 0; k_idx < ouput_c; k_idx++)
		{
			int temp2k = k_idx * temp1k;
			int temp3o = k_idx * outputHeightSize * outputWidthSize + temp2o;
			for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
			{
				int temp3i = ⁠c_idx * input_w * input_h + temp2i;
				int temp3k = ⁠c_idx * kernelSize * kernelSize + temp2k;
				for (int rowStride = 0; rowStride < outputHeightSize; rowStride++) {
					int temp4o = rowStride * outputWidthSize + temp3o;
					for (int colStride = 0; colStride < outputWidthSize; colStride++) {
						float sum = 0;
						int g_idx_o = colStride + temp4o;
						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							int temp4i = x * input_w + temp3i;
							int temp4k = (x - rowStride * stride) * kernelSize + temp3k;
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								int ⁠g_idx_i = y + temp4i;
								int g_idx_k = (y - colStride * stride) + temp4k;
								sum += convInput[⁠g_idx_i] * kernel[g_idx_k];
							}
						}
						convOutput[g_idx_o] += sum;
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
	vector<pair<Mat, string>> imgBox; // 이미지 데이터, 이미지 이름
	imgBox = TraverseFilesUsingDFS("C:\\cifar\\test10");// 이미지가 저장되어 있는 폴더 경로

	//입력변수
	int input_n = imgBox.size(); // 10
	int input_c = imgBox[0].first.channels(); // 3
	int input_h = imgBox[0].first.rows;//H 32 
	int input_w = imgBox[0].first.cols;//W 32
	int inputDataSize = input_n * input_c * input_h * input_w;
	vector<float> inputVec(inputDataSize);

	cout << "===== 2. zeroPadding and mat -> vector  ===== \n";
	// mat 형식 - > 4차 행렬
	int temp1 = input_w * input_h * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		unsigned char* temp = imgBox[⁠n_idx].first.data;
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				int temp5 = input_c * input_w * ⁠h_idx;
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int ⁠g_idx = w_idx + temp4;
					inputVec[⁠g_idx] = temp[temp5 + input_c * w_idx + ⁠c_idx];
				}
			}
		}
	}


	cout << "===== Input Value check  ===== \n";
	valueCheck(inputVec, input_n, input_c, input_h, input_w, 1);

	// 커널, window, weight
	//cout << "===== 3. weight(filter) generation ===== \n";
	//InitWeightsXavier(4, 3, 3, 3);
	//  임시 커널
	vector<float> kernel(4 * 3 * 3 * 3); // [oc, ic, h, w]
	for (int i = 0; i < kernel.size(); i++) {
		kernel[i] = 1;
	}

	clock_t start, finish;
	double  duration;

	cout << "===== 5. Convolution ===== \n";
	int outputCh = 4;
	int kernelSize = 3;
	int stride = 1;
	int outputHeight = ((input_h  - kernelSize) / stride) + 1;
	int outputWidth = ((input_w  - kernelSize) / stride) + 1;
	int convOutputSize = input_n * outputCh * outputHeight * outputWidth;

	vector<float> conv1Output(convOutputSize);
	start = clock();
	convolution(conv1Output, inputVec, kernel, kernelSize, stride, input_n, input_c, input_h, input_w, outputCh);
	finish = clock();

	duration = (double)(finish - start) / CLOCKS_PER_SEC;

	valueCheck(conv1Output, input_n, outputCh, outputHeight, outputWidth, 1);

	printf("%2.5f seconds\n", duration);

	return 0;
}