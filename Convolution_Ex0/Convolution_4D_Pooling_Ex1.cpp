#include <io.h>
#include <iostream>
#include <string>
#include <vector>
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
			size_t npo1 = file_name.find('_') + 1;
			size_t npo2 = file_name.find('.');
			size_t npo3 = npo2 - npo1;
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
	return (float)tanh(x);
}

float ActivationSigmoid(float x) {
	return (float)(1 / (exp(-x) + 1));
}

float ActivationReLU(float x) {
	return (float)(x > 0 ? x : 0.0);
}

void ValueCheck(vector<vector<vector<vector<float>>>> ValueCheck_input, int offset = 1) {
	int ImageNum = 1;
	if (offset == 0) { ImageNum = ValueCheck_input.size(); }
	int FeatureCh = ValueCheck_input[0].size();
	int FeatureHeight = ValueCheck_input[0][0].size();
	int FeatureWidth = ValueCheck_input[0][0][0].size();
	for (int i = 0; i < ImageNum; i++)
	{
		for (int c = 0; c < FeatureCh; c++)
		{
			for (int y = 0; y < FeatureHeight; y++)
			{
				for (int x = 0; x < FeatureWidth; x++)
				{
					cout << setw(5) << ValueCheck_input[i][c][y][x];
				}cout << endl;
			}cout << endl; cout << endl;
		}
	}
}

vector<vector<vector<vector<float>>>> Convolution(
	int kernelSize,
	int stride,
	int ZeroPaddingSize,
	int ImageNum,
	int FeatureCh,
	int FeatureHeight,
	int FeatureWidth,
	vector<vector<vector<vector<float>>>> kernel,
	vector<vector<vector<vector<float>>>> input_vec) {

	int OutputSize = ((FeatureHeight - kernelSize + ZeroPaddingSize * 2) / stride) + 1;
	vector<vector<vector<vector<float>>>> output_vec; // output 값
	vector<vector<vector<vector<float>>>> input_vec_zeropadding;  // zeropadding

	cout << "===== ZeroPadding ===== \n";
	int FeatureHeightWithZeroPadding = FeatureHeight + 2 * ZeroPaddingSize;
	int FeatureWidthWithZeroPadding = FeatureWidth + 2 * ZeroPaddingSize;
	input_vec_zeropadding.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		input_vec_zeropadding[i].resize(FeatureCh);
		for (int c = 0; c < FeatureCh; c++)
		{
			input_vec_zeropadding[i][c].resize(FeatureHeightWithZeroPadding);
			for (int p = 0; p < ZeroPaddingSize; p++) {
				input_vec_zeropadding[i][c][p].resize(FeatureWidthWithZeroPadding);
				input_vec_zeropadding[i][c][FeatureHeightWithZeroPadding - 1 - p].resize(FeatureWidthWithZeroPadding);
			}
			for (int y = ZeroPaddingSize; y < FeatureHeight + ZeroPaddingSize; y++)
			{
				input_vec_zeropadding[i][c][y].resize(FeatureWidthWithZeroPadding);
				for (int x = ZeroPaddingSize; x < FeatureWidth+ ZeroPaddingSize; x++)
				{
					input_vec_zeropadding[i][c][y][x] = input_vec[i][c][y- ZeroPaddingSize][x- ZeroPaddingSize];
				}
			}
		}
	}

	cout << "===== Zeropadding Value check  ===== \n";

	//ValueCheck(input_vec_zeropadding);

	output_vec.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		output_vec[i].resize(2);
		for (int k = 0; k < 2; k++)
		{
			output_vec[i][k].resize(OutputSize);
			for (int y = 0; y < OutputSize; y++)
			{
				output_vec[i][k][y].resize(OutputSize);
			}
		}
	}
	cout << "===== Convolution ===== \n";
	for (int i = 0; i < 1; i++)
	{
		for (int k = 0; k < 2; k++)
		{
			for (int c = 0; c < FeatureCh; c++)
			{
				for (int rowStride = 0; rowStride < OutputSize; rowStride++) {
					for (int colStride = 0; colStride < OutputSize; colStride++) {
						int sum = 0;
						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								sum += input_vec_zeropadding[i][c][x][y] * kernel[k][c][x-stride*rowStride][y-stride*colStride];
							}
						}
						output_vec[i][k][rowStride][colStride] += sum;
					}
				}
			}
		}
	}

	/*
	cout << "===== 3. Activation Function  ===== \n";
	for (int i = 0; i < 1; i++)
	{
		for (int k = 0; k < 2; k++)
		{
			for (int y = 0; y < output_vec[0][0].size(); y++)
			{
				for (int x = 0; x < output_vec[0][0].size(); x++)
				{
					 output_vec[i][k][y][x] = ActivationSigmoid(output_vec[i][k][y][x]);
				}
			}
		}
	}*/
	return output_vec;
}


vector<vector<vector<vector<float>>>> MaxPooling(
	int PoolingWindow,
	int PoolingStride,
	vector<vector<vector<vector<float>>>> MaxPooling_input) {
	cout << "===== MaxPooling ===== \n";
	int ImageNum = MaxPooling_input.size();
	int FeatureCh = MaxPooling_input[0].size() ;
	int PoolingInputSize = MaxPooling_input[0][0].size();

	vector<vector<vector<vector<float>>>> PoolingOutput;
	int PoolingOutputSize = ((PoolingInputSize - PoolingWindow) / PoolingStride) + 1;//16

	PoolingOutput.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		PoolingOutput[i].resize(FeatureCh);
		for (int k = 0; k < FeatureCh; k++)
		{
			PoolingOutput[i][k].resize(PoolingOutputSize);
			for (int y = 0; y < PoolingOutputSize; y++)
			{
				PoolingOutput[i][k][y].resize(PoolingOutputSize);
			}
		}
	}

	for (int i = 0; i < ImageNum; i++)
	{
		for (int k = 0; k < FeatureCh; k++)
		{
			for (int rowStride = 0; rowStride < PoolingOutputSize; rowStride++) {
				for (int colStride = 0; colStride < PoolingOutputSize; colStride++) {
					int Max = 0;
					for (int x = rowStride * PoolingStride; x < rowStride * PoolingStride + PoolingWindow; x++) {
						for (int y = colStride * PoolingStride; y < colStride * PoolingStride + PoolingWindow; y++) {
							if (Max < MaxPooling_input[i][k][x][y])
							{
								Max = MaxPooling_input[i][k][x][y];
							}
						}
					}
					PoolingOutput[i][k][rowStride][colStride] = Max;
				}
			}
		}
	}
	
	return PoolingOutput;
}

vector<vector<vector<vector<float>>>> AvgPooling(
	int PoolingWindow,
	int PoolingStride,
	vector<vector<vector<vector<float>>>> AvgPooling_input) {
	cout << "===== AvgPooling ===== \n";
	int ImageNum = AvgPooling_input.size();
	int FeatureCh = AvgPooling_input[0].size();
	int PoolingInputSize = AvgPooling_input[0][0].size();

	vector<vector<vector<vector<float>>>> PoolingOutput;
	int PoolingOutputSize = ((PoolingInputSize - PoolingWindow) / PoolingStride) + 1;//16

	PoolingOutput.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		PoolingOutput[i].resize(FeatureCh);
		for (int k = 0; k < FeatureCh; k++)
		{
			PoolingOutput[i][k].resize(PoolingOutputSize);
			for (int y = 0; y < PoolingOutputSize; y++)
			{
				PoolingOutput[i][k][y].resize(PoolingOutputSize);
			}
		}
	}

	float PoolingWindowAreaInverse = 1 / (float)(PoolingWindow * PoolingWindow);
	for (int i = 0; i < ImageNum; i++)
	{
		for (int k = 0; k < 2; k++)
		{
			for (int rowStride = 0; rowStride < PoolingOutputSize; rowStride++) {
				for (int colStride = 0; colStride < PoolingOutputSize; colStride++) {
					int sum = 0;
					for (int x = rowStride * PoolingStride; x < rowStride * PoolingStride + PoolingWindow; x++) {
						for (int y = colStride * PoolingStride; y < colStride * PoolingStride + PoolingWindow; y++) {
							sum += AvgPooling_input[i][k][x][y];
						}
					}
					PoolingOutput[i][k][rowStride][colStride] = sum * PoolingWindowAreaInverse;
				}
			}
		}
	}

	return PoolingOutput;
}


vector<float> FullyConnected(int FullyConnectedSize,
	vector<vector<vector<vector<float>>>> fullyConnectedInput,
	vector<vector<vector<vector<vector<float>>>>> fullyConnectedWeight) {

	int ImageNumf = fullyConnectedInput.size();
	int FeatureChf = fullyConnectedInput[0].size();
	int FeatureHeightf = fullyConnectedInput[0][0].size();
	int FeatureWidthf = fullyConnectedInput[0][0][0].size();
	vector<float> FullyConnectedOutput;
	FullyConnectedOutput.resize(FullyConnectedSize);

	for (int f = 0; f < FullyConnectedSize; f++)
	{
		int sum = 0;
		for (int i = 0; i < ImageNumf; i++)
		{
			for (int c = 0; c < FeatureChf; c++)
			{
				for (int y = 0; y < FeatureHeightf; y++)
				{
					for (int x = 0; x < FeatureWidthf; x++)
					{
						sum += fullyConnectedInput[i][c][y][x] * fullyConnectedWeight[f][i][c][y][x];
					}
				}
			}
		}
		FullyConnectedOutput[f] = sum;
	}
	return FullyConnectedOutput;
}

vector<float> FullyConnected(int FullyConnectedSize, vector<float> fullyConnectedInput, vector<vector<float>> fullyConnectedWeight) {

	int ImageNumf = fullyConnectedInput.size();

	vector<float> FullyConnectedOutput;

	FullyConnectedOutput.resize(FullyConnectedSize);

	for (int f = 0; f < FullyConnectedSize; f++)
	{
		int sum = 0;
		for (int i = 0; i < ImageNumf; i++)
		{
			sum += fullyConnectedInput[i] * fullyConnectedWeight[f][i];
		}
		FullyConnectedOutput[f] = sum;
	}
	return FullyConnectedOutput;
}


vector<vector<vector<vector<vector<float>>>>> GeneratorFullyConnectedWeight(vector<vector<vector<vector<float>>>> FullyConnectedInput, int FullyConnectedSize) {
	vector<vector<vector<vector<vector<float>>>>> fullyConnectedWeight;

	int ImageNumf = FullyConnectedInput.size();
	int FeatureChf = FullyConnectedInput[0].size();
	int FeatureHeightf = FullyConnectedInput[0][0].size();
	int FeatureWidthf = FullyConnectedInput[0][0][0].size();

	fullyConnectedWeight.resize(FullyConnectedSize);
	for (int f = 0; f < FullyConnectedSize; f++)
	{
		fullyConnectedWeight[f].resize(ImageNumf);
		for (int i = 0; i < ImageNumf; i++)
		{
			fullyConnectedWeight[f][i].resize(FeatureChf);
			for (int c = 0; c < FeatureChf; c++)
			{
				fullyConnectedWeight[f][i][c].resize(FeatureHeightf);
				for (int y = 0; y < FeatureHeightf; y++)
				{
					fullyConnectedWeight[f][i][c][y].resize(FeatureWidthf);
					for (int x = 0; x < FeatureWidthf; x++)
					{
						fullyConnectedWeight[f][i][c][y][x] = 1.0;
					}
				}
			}
		}
	}
	return fullyConnectedWeight;
}


vector<vector<float>>  GeneratorFullyConnectedWeight(vector<float> FullyConnectedInput, int FullyConnectedSize) {
	vector<vector<float>> fullyConnectedWeight;

	int ImageNumf = FullyConnectedInput.size();

	fullyConnectedWeight.resize(FullyConnectedSize);
	for (int f = 0; f < FullyConnectedSize; f++)
	{
		fullyConnectedWeight[f].resize(ImageNumf);
		for(int i = 0; i < ImageNumf; i++){
			fullyConnectedWeight[f][i] = 1.0;
		}
	}

	return fullyConnectedWeight;
}


int main()
{
	cout << "4D([N][C][W][H]) Convolutions and Pooling(Avg & Max) ! \n\n";

	const int batch_size = 10; // 사용 할 이미지 갯수
	int ZeroPaddingSize = 1;
	int stride = 1;

	cout << "===== 1. Image loading ===== \n";

	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS("C:\\cifar\\test10");// 이미지가 저장되어 있는 폴더 경로

	vector<vector<vector<vector<float>>>> input_vec;  // 4D 입력 이미지 데이터
	vector<vector<vector<vector<float>>>> output_1; // 한번 Convolution 후 4D 출력 이미지 데이터

	//입력변수
	int ImageNum = batch_size;
	int FeatureCh = ImgBox[0].first.channels();
	int FeatureHeight = ImgBox[0].first.rows;
	int FeatureWidth = ImgBox[0].first.cols;

	cout << "===== 2. zeropadding and mat -> vector  ===== \n";

	// mat 형식 - > 4차 행렬
	input_vec.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		unsigned char* temp = ImgBox[i].first.data;
		input_vec[i].resize(FeatureCh);
		for (int c = 0; c < FeatureCh; c++)
		{
			input_vec[i][c].resize(FeatureHeight);
			for (int y = 0; y < FeatureHeight; y++)
			{
				input_vec[i][c][y].resize(FeatureWidth);
				for (int x = 0; x < FeatureWidth; x++)
				{
					input_vec[i][c][y][x] = temp[FeatureCh * FeatureWidth * y + FeatureCh * x + c];
				}
			}
		}
	}

	cout << "===== 3. Input Value check  ===== \n";

	//ValueCheck(input_vec);

	// 커널, window, weight
	vector<vector<vector<vector<float>>>> kernel; //[k][c][w][h]
	kernel = { {
	{ {1, 0, 1 },
	{  0, 1, 0 },
	{  1, 0, 1 } },

	{ {1, 0, 1 },
	{  0, 1, 0 },
	{  1, 0, 1 } },

	{ {1, 0, 1 },
	{  0, 1, 0 },
	{  1, 0, 1 } } },

{
	{ {0, 0, 1 },
	{  0, 1, 0 },
	{  0, 0, 1 } },

	{ {1, 0, 1 },
	{  0, 0, 0 },
	{  1, 0, 1 } },

	{ {1, 0, 0 },
	{  0, 1, 0 },
	{  1, 0, 0 } } } };

	cout << "===== 4. Convolution ===== \n";

	output_1 = Convolution(3, stride, ZeroPaddingSize, ImageNum, FeatureCh, FeatureHeight, FeatureWidth, kernel, input_vec);

	cout << "===== 5. Convolution Output (Feature map) check  ===== \n";

	//ValueCheck(output_1);
	
	cout << "===== 6. Pooling ===== \n";
	//vector<vector<vector<vector<float>>>> PoolingOutput = AvgPooling(2, 2, output_1);
	vector<vector<vector<vector<float>>>> PoolingOutput = MaxPooling(2, 2, output_1);

	cout << "===== 7.Pooling check  ===== \n";

	//ValueCheck(PoolingOutput);


	// 이전 레이어의 모든 파라미터수 * fullyConnected layer의 파라미터 수 =   fully connected weight 수 
	cout << "===== 8.FullyConnected  ===== \n";

	int FullyConnectedSize1 = 10;
	vector<vector<vector<vector<vector<float>>>>> fullyConnectedWeight1 = GeneratorFullyConnectedWeight(PoolingOutput,  FullyConnectedSize1);
	vector<float> FullyConnectedOutput1 = FullyConnected( FullyConnectedSize1, PoolingOutput ,fullyConnectedWeight1);

	cout << "===== 8.FullyConnectedOutput1 check  ===== \n";
	for (int i = 0; i < FullyConnectedSize1; i++) {
		cout << setw(8) << FullyConnectedOutput1[i];
	}cout << endl;

	int FullyConnectedSize2 = 5;
	vector<vector<float>> fullyConnectedWeight2 = GeneratorFullyConnectedWeight(FullyConnectedOutput1, FullyConnectedSize2);
	vector<float> FullyConnectedOutput2 = FullyConnected(FullyConnectedSize2, FullyConnectedOutput1, fullyConnectedWeight2);

	cout << "===== 9.FullyConnectedOutput2 check  ===== \n";
	for (int i = 0; i < FullyConnectedSize2; i++) {
		cout << setw(11) << FullyConnectedOutput2[i]<< " ";
	}cout << endl;


	return 0;
}