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
				temp4o = (⁠h_idx + TopPadingSize)*(Input_W + LeftPadingSize + RightPadingSize)+ LeftPadingSize + temp3o;
				
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

/*
vector<vector<vector<vector<float>>>> Convolution(
	int kernelSize,
	int stride,
	vector<vector<vector<vector<float>>>> kernel,
	vector<vector<vector<vector<float>>>> input_vec) {

	int KernelCh = kernel.size();
	int ImageNum = input_vec.size();
	int FeatureCh = input_vec[0].size();
	int FeatureHeight = input_vec[0][0].size();
	int FeatureWidth = input_vec[0][0][0].size();

	int OutputHeightSize = ((FeatureHeight - kernelSize) / stride) + 1;
	int OutputWidthSize = ((FeatureWidth - kernelSize) / stride) + 1;

	vector<vector<vector<vector<float>>>> output_vec; // output 값

	output_vec.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		output_vec[i].resize(KernelCh);
		for (int k = 0; k < KernelCh; k++)
		{
			output_vec[i][k].resize(OutputHeightSize);
			for (int y = 0; y < OutputHeightSize; y++)
			{
				output_vec[i][k][y].resize(OutputWidthSize);
			}
		}
	}

	//cout << "===== Convolution ===== \n";
	for (int i = 0; i < ImageNum; i++)
	{
		for (int k = 0; k < KernelCh; k++)
		{
			for (int c = 0; c < FeatureCh; c++)
			{
				for (int rowStride = 0; rowStride < OutputWidthSize; rowStride++) {
					for (int colStride = 0; colStride < OutputHeightSize; colStride++) {
						float sum = 0;
						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								sum += input_vec[i][c][x][y] * kernel[k][c][x - stride * rowStride][y - stride * colStride];
							}
						}
						output_vec[i][k][rowStride][colStride] += sum;
					}
				}
			}
		}
	}


	return output_vec;
}

vector<vector<vector<vector<float>>>> ConvolutionWithZeroPadding(
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

	cout << "===== Zero Padding ===== \n";
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
				for (int x = ZeroPaddingSize; x < FeatureWidth + ZeroPaddingSize; x++)
				{
					input_vec_zeropadding[i][c][y][x] = input_vec[i][c][y - ZeroPaddingSize][x - ZeroPaddingSize];
				}
			}
		}
	}

	//cout << "===== Zeropadding Value check  ===== \n";
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

	//cout << "===== Convolution ===== \n";
	for (int i = 0; i < ImageNum; i++)
	{
		for (int k = 0; k < 2; k++)
		{
			for (int c = 0; c < FeatureCh; c++)
			{
				for (int rowStride = 0; rowStride < OutputSize; rowStride++) {
					for (int colStride = 0; colStride < OutputSize; colStride++) {
						float sum = 0;
						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								sum += input_vec_zeropadding[i][c][x][y] * kernel[k][c][x - stride * rowStride][y - stride * colStride];
							}
						}
						output_vec[i][k][rowStride][colStride] += sum;
					}
				}
			}
		}
	}


	return output_vec;
}




vector<vector<vector<vector<float>>>> MaxPooling(int PoolingWindow, int PoolingStride, vector<vector<vector<vector<float>>>> MaxPooling_input) {
	cout << "===== MaxPooling ===== \n";
	int ImageNum = MaxPooling_input.size();
	int FeatureCh = MaxPooling_input[0].size();
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
					float Max = 0;
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

vector<vector<vector<vector<float>>>> AvgPooling(int PoolingWindow, int PoolingStride, vector<vector<vector<vector<float>>>> AvgPooling_input) {
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
		for (int k = 0; k < FeatureCh; k++)
		{
			for (int rowStride = 0; rowStride < PoolingOutputSize; rowStride++) {
				for (int colStride = 0; colStride < PoolingOutputSize; colStride++) {
					float sum = 0;
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

vector<vector<float>> FullyConnected(int FullyConnectedSize, vector<vector<vector<vector<float>>>> fullyConnectedInput, vector<vector<vector<vector<float>>>> fullyConnectedWeight) {

	int ImageNumf = fullyConnectedInput.size();
	int FeatureChf = fullyConnectedInput[0].size();
	int FeatureHeightf = fullyConnectedInput[0][0].size();
	int FeatureWidthf = fullyConnectedInput[0][0][0].size();
	vector<vector<float>> FullyConnectedOutput;
	FullyConnectedOutput.resize(ImageNumf);
	for (int i = 0; i < ImageNumf; i++)
	{
		FullyConnectedOutput[i].resize(FullyConnectedSize);
		for (int f = 0; f < FullyConnectedSize; f++)
		{
			float sum = 0;

			for (int c = 0; c < FeatureChf; c++)
			{
				for (int y = 0; y < FeatureHeightf; y++)
				{
					for (int x = 0; x < FeatureWidthf; x++)
					{
						sum += fullyConnectedInput[i][c][y][x] * fullyConnectedWeight[f][c][y][x];
					}
				}
			}
			FullyConnectedOutput[i][f] = sum;
		}

	}
	return FullyConnectedOutput;
}

vector<vector<float>>  FullyConnected(int FullyConnectedSize, vector<vector<float>> fullyConnectedInput, vector<vector<float>> fullyConnectedWeight) {

	int ImageNumf = fullyConnectedInput.size();
	int PrevFullySize = fullyConnectedInput[0].size();
	int NextFullySize = FullyConnectedSize;

	vector<vector<float>> FullyConnectedOutput;
	FullyConnectedOutput.resize(ImageNumf);

	for (int n = 0; n < ImageNumf; n++)
	{
		FullyConnectedOutput[n].resize(NextFullySize);
		for (int a = 0; a < NextFullySize; a++)
		{
			float sum = 0;
			for (int f = 0; f < PrevFullySize; f++)
			{
				sum += fullyConnectedInput[n][f] * fullyConnectedWeight[a][f];
			}
			FullyConnectedOutput[n][a] = sum;
		}
	}
	return FullyConnectedOutput;
}

vector<vector<vector<vector<float>>>> GeneratorFullyConnectedWeight(vector<vector<vector<vector<float>>>> FullyConnectedInput, int FullyConnectedSize) {
	vector<vector<vector<vector<float>>>> fullyConnectedWeight;

	int FeatureChf = FullyConnectedInput[0].size();
	int FeatureHeightf = FullyConnectedInput[0][0].size();
	int FeatureWidthf = FullyConnectedInput[0][0][0].size();

	fullyConnectedWeight.resize(FullyConnectedSize);
	for (int f = 0; f < FullyConnectedSize; f++)
	{
		fullyConnectedWeight[f].resize(FeatureChf);
		for (int c = 0; c < FeatureChf; c++)
		{
			fullyConnectedWeight[f][c].resize(FeatureHeightf);
			for (int y = 0; y < FeatureHeightf; y++)
			{
				fullyConnectedWeight[f][c][y].resize(FeatureWidthf);
				for (int x = 0; x < FeatureWidthf; x++)
				{
					fullyConnectedWeight[f][c][y][x] = 1.0;
				}
			}
		}

	}
	return fullyConnectedWeight;
}

vector<vector<float>>  GeneratorFullyConnectedWeight(vector<vector<float>> FullyConnectedInput, int NextFullySize) {
	vector<vector<float>> fullyConnectedWeight;
	int PrevFullySize = FullyConnectedInput[0].size();
	random_device rd;
	mt19937 gen(rd());
	float sigma = sqrt(6.0f / static_cast<float>(PrevFullySize + NextFullySize));
	uniform_real_distribution<float> d(-sigma, sigma);

	fullyConnectedWeight.resize(NextFullySize);
	for (int f = 0; f < NextFullySize; f++)
	{
		fullyConnectedWeight[f].resize(PrevFullySize);
		for (int i = 0; i < PrevFullySize; i++) {
			fullyConnectedWeight[f][i] = static_cast<float>(d(gen));
		}
	}
	return fullyConnectedWeight;
}

vector<vector<vector<vector<float>>>> InitWeightsXavier(int PrevLayerChannel, int NextLayerNodeNumber, int Height, int Width)
{
	random_device rd;
	mt19937 gen(rd());
	float sigma = sqrt(6.0f / static_cast<float>((NextLayerNodeNumber + PrevLayerChannel) * Height * Width));
	uniform_real_distribution<float> d(-sigma, sigma);

	vector<vector<vector<vector<float>>>> kernel;

	//Weights initializer
	kernel.resize(NextLayerNodeNumber);
	for (int och = 0; och < NextLayerNodeNumber; och++)
	{
		kernel[och].resize(PrevLayerChannel);
		for (int ch = 0; ch < PrevLayerChannel; ch++)
		{
			kernel[och][ch].resize(Height);
			for (int row = 0; row < Height; row++)
			{
				kernel[och][ch][row].resize(Width);
				for (int col = 0; col < Width; col++)
				{
					kernel[och][ch][row][col] = static_cast<float>(d(gen));
				}
			}
		}
	}
	return kernel;
}

void Softmax(vector<vector<float>> &Expect)
{
	int ImageNumf = Expect.size();
	int LastNodeNumber = Expect[0].size();

	for (int n = 0; n < ImageNumf; n++) {
		float sum = 0.0;

		for (int i = 0; i < LastNodeNumber; i++)
		{
			sum += exp(Expect[n][i]);
		}

		for (int i = 0; i < LastNodeNumber; i++)
		{
			Expect[n][i] = exp(Expect[n][i]) / sum;
		}
	}
}

void OneHotEncoding(vector<vector<float>> &testvalue) {

	for (int i = 0; i < testvalue.size(); i++) {

		float onehot = testvalue[i][0];
		int onehotindex = 0;
		for (int j = 1; j < testvalue[0].size(); j++) {

			if (onehot > testvalue[i][j]) {
				testvalue[i][j] = 0;
			}
			else {
				testvalue[i][onehotindex] = 0;
				onehot = testvalue[i][j];
				onehotindex = j;
			}
		}
		testvalue[i][onehotindex] = 1;
	}
}
*/
int main()
{
	cout << "4D([N][C][H][W]) Convolutions ! \n\n";
	int stride = 1;
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
		for (⁠c_idx= 0; ⁠c_idx < Input_C; ⁠c_idx++)
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
	ValueCheck(Input_Vec, Input_N, Input_C , Input_H , Input_W , 1);



	// 커널, window, weight
	//cout << "===== 3. weight(filter) generation ===== \n";
	//vector<float> kernel = InitWeightsXavier(3, 2, 3, 3);
	
	
	//cout << "===== weight check ===== \n";
	//ValueCheck(kernel, 0);
	vector<float> kernel;
	Activation(kernel);
	//ValueCheck(kernel, 0);

	cout << "===== 4. Convolution ===== \n";
	//ValueCheck(input_vec);

	int TopPadingSize = 1;
	int BottomPadingSize = 2;
	int LeftPadingSize = 4;
	int RightPadingSize = 3;

	vector<float> Input_Vec_ZeroPading(Input_N*Input_C*(Input_H + TopPadingSize + BottomPadingSize)*(Input_W + LeftPadingSize + RightPadingSize));

	ZeroPadding(Input_Vec_ZeroPading, Input_Vec, Input_N, Input_C, Input_H, Input_W, TopPadingSize, BottomPadingSize, LeftPadingSize, RightPadingSize);

	ValueCheck(Input_Vec_ZeroPading, Input_N, Input_C, Input_H + TopPadingSize + BottomPadingSize, Input_W + LeftPadingSize + RightPadingSize, 1);




	/*
	//ValueCheck(input_vec, 0);
	vector<vector<vector<vector<float>>>> output_1 = Convolution(3, stride, kernel, input_vec);

	//cout << "===== Convolution Output (Feature map) check  ===== \n";
	//ValueCheck(output_1, 0);

	cout << "===== 5. Activation  ===== \n";
	Activation(output_1);

	cout << "===== Actvation check ===== \n";
	//ValueCheck(output_1, 0);

	cout << "===== 6. Pooling ===== \n";
	//vector<vector<vector<vector<float>>>> PoolingOutput = AvgPooling(2, 2, output_1);
	vector<vector<vector<vector<float>>>> PoolingOutput = MaxPooling(2, 2, output_1);

	cout << "===== Pooling check  ===== \n";
	//ValueCheck(PoolingOutput, 0);

	cout << "===== 7. FullyConnected_1  ===== \n";
	int FullyConnectedSize1 = 10;
	vector<vector<vector<vector<float>>>> fullyConnectedWeight1 = GeneratorFullyConnectedWeight(PoolingOutput, FullyConnectedSize1);
	vector<vector<float>> FullyConnectedOutput1 = FullyConnected(FullyConnectedSize1, PoolingOutput, fullyConnectedWeight1);
	Activation(FullyConnectedOutput1);
	ValueCheck(FullyConnectedOutput1, 0);

	cout << "===== 8. FullyConnected_2  ===== \n";
	int FullyConnectedSize2 = 5;
	vector<vector<float>> fullyConnectedWeight2 = GeneratorFullyConnectedWeight(FullyConnectedOutput1, FullyConnectedSize2);
	vector<vector<float>> FullyConnectedOutput2 = FullyConnected(FullyConnectedSize2, FullyConnectedOutput1, fullyConnectedWeight2);

	//cout << "===== FullyConnectedOutput check  ===== \n";
	ValueCheck(FullyConnectedOutput2, 0);

	cout << "===== 9. Softmax  ===== \n";
	Softmax(FullyConnectedOutput2);
	ValueCheck(FullyConnectedOutput2, 0);

	cout << "===== 10. OneHotEncoding  ===== \n";
	OneHotEncoding(FullyConnectedOutput2);
	ValueCheck(FullyConnectedOutput2, 0);
	*/
	return 0;
}