#include<io.h>
#include<iostream>
#include<string>
#include<vector>
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

vector<vector<vector<vector<float>>>> Convolution(
	int kernelSize,
	int inputSize,
	int stride,
	int n,
	int ch,
	vector<vector<vector<float>>> kernel,
	vector<vector<vector<vector<float>>>> input_vec) {

	int OutputSize = ((inputSize - kernelSize) / stride) + 1;
	vector<vector<vector<vector<float>>>> output_vec;

	output_vec.resize(n);
	for (int i = 0; i < n; i++)
	{
		output_vec[i].resize(ch);
		for (int c = 0; c < ch; c++)
		{
			output_vec[i][c].resize(OutputSize);
			for (int y = 0; y < OutputSize; y++)
			{
				output_vec[i][c][y].resize(OutputSize);
			}
		}
	}

	for (int i = 0; i < 1; i++)
	{
		for (int c = 0; c < ch; c++)
		{
			for (int rowStride = 0; rowStride < OutputSize; rowStride++) {
				for (int colStride = 0; colStride < OutputSize; colStride++) {
					int sum = 0;
					for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
						for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
							sum += input_vec[i][c][x][y] * kernel[c][x - stride * rowStride][y - stride * colStride];
						}
					}
					output_vec[i][c][rowStride][colStride] = sum;
				}
			}
		}
	}

	return output_vec;
}




int main()
{
	std::cout << "4D([N][C][W][H]) Convolutions ! \n\n";

	std::cout << "===== 1. Image loading ===== \n";

	const int batch_size = 10; // 이미지 총 갯수
	int ZeroPaddingSize = 1;
	int stride = 2;

	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS("C:\\cifar\\test10");// 이미지가 저장되어 있는 폴더 경로

	std::cout << "===== 2. zeropadding and mat -> vector  ===== \n";

	vector<vector<vector<vector<float>>>> input_vec;  // 4D 입력 이미지 데이터
	vector<vector<vector<vector<float>>>> output_vec; // 한번 Convolution 후 4D 출력 이미지 데이터

	//입력변수

	int ImageNum = batch_size;
	int FeatureCh = ImgBox[0].first.channels();
	int FeatureHeight = ImgBox[0].first.rows + ZeroPaddingSize * 2;
	int FeatureWidth = ImgBox[0].first.cols + ZeroPaddingSize * 2;

	// mat 형식 - > 4차 행렬 , ZeroPadding 
	input_vec.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		unsigned char* temp = ImgBox[i].first.data;
		input_vec[i].resize(FeatureCh);
		for (int c = 0; c < FeatureCh; c++)
		{
			input_vec[i][c].resize(FeatureHeight);

			for (int p = 0; p < ZeroPaddingSize; p++) {
				input_vec[i][c][p].resize(FeatureWidth);
				input_vec[i][c][FeatureHeight - 1 - p].resize(FeatureWidth);
			}

			for (int y = ZeroPaddingSize; y < FeatureHeight - ZeroPaddingSize; y++)
			{
				input_vec[i][c][y].resize(FeatureWidth);
				for (int x = ZeroPaddingSize; x < FeatureWidth - ZeroPaddingSize; x++)
				{
					input_vec[i][c][y][x] = temp[3 * 32 * (y - ZeroPaddingSize) + 3 * (x - ZeroPaddingSize) + c];
				}
			}
		}
	}

	std::cout << "===== 3. Input Value check  ===== \n";

	for (int i = 0; i < 1; i++)
	{
		for (int c = 0; c < FeatureCh; c++)
		{
			for (int y = 0; y < FeatureHeight; y++)
			{
				for (int x = 0; x < FeatureWidth; x++)
				{
					std::cout << setw(4) << input_vec[i][c][y][x];
				}std::cout << std::endl;
			}std::cout << std::endl; std::cout << std::endl;
		}
	}

	std::cout << "===== 4. Convolution  ===== \n";


	vector<vector<vector<float>>> kernel;
	kernel = {
	{ {1, 0, 1},
	{  0, 1, 0},
	{  1, 0, 1} },

	{ {1, 0, 1},
	{  0, 1, 0 },
	{  1, 0, 1 } },

	{ {1, 0, 1},
	{  0, 1, 0 },
	{  1, 0, 1 } } };


	vector<vector<vector<vector<float>>>> output_1 = Convolution( 3, FeatureHeight, stride, ImageNum, FeatureCh, kernel, input_vec);

	std::cout << "===== 5. Output (Feature map) check  ===== \n";

	for (int i = 0; i < 1; i++)
	{
		for (int c = 0; c < FeatureCh; c++)
		{
			for (int y = 0; y < output_1[0][0].size(); y++)
			{
				for (int x = 0; x < output_1[0][0].size(); x++)
				{
					std::cout << setw(4) << output_1[i][c][y][x];
				}std::cout << std::endl;
			}std::cout << std::endl; std::cout << std::endl;
		}
	}



	return 0;
}