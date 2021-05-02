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
	int stride,
	int ZeroPaddingSize,
	int ImageNum,
	int FeatureCh,
	int FeatureHeight,
	int FeatureWidth,
	vector<vector<vector<float>>> kernel,
	vector<vector<vector<vector<float>>>> input_vec) {

	int OutputSize = ((FeatureHeight - kernelSize + (ZeroPaddingSize * 2)) / stride) + 1;

	// OutputSize = (N-K+2P)/S + 1 
	vector<vector<vector<vector<float>>>> output_vec; // output 값
	vector<vector<vector<vector<float>>>> input_vec_zeropadding;  // zeropadding
	
	input_vec_zeropadding.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		input_vec_zeropadding[i].resize(FeatureCh);
		for (int c = 0; c < FeatureCh; c++)
		{
			input_vec_zeropadding[i][c].resize(FeatureHeight + 2 * ZeroPaddingSize);
			for (int p = 0; p < ZeroPaddingSize; p++) {
				input_vec_zeropadding[i][c][p].resize(FeatureWidth + 2 * ZeroPaddingSize);
				input_vec_zeropadding[i][c][FeatureHeight + 2 * ZeroPaddingSize - 1 - p].resize(FeatureWidth + 2 * ZeroPaddingSize);
			}
			for (int y = 0; y < FeatureHeight; y++)
			{
				input_vec_zeropadding[i][c][y + ZeroPaddingSize].resize(FeatureWidth + 2 * ZeroPaddingSize);
				for (int x = 0; x < FeatureWidth; x++)
				{
					input_vec_zeropadding[i][c][y + ZeroPaddingSize][x + ZeroPaddingSize] = input_vec[i][c][y][x];
				}
			}
		}
	}

	/*
	for (int i = 0; i < 1; i++)
	{
		for (int c = 0; c < FeatureCh; c++)
		{
			for (int y = 0; y < FeatureHeight + 2 * ZeroPaddingSize; y++)
			{
				for (int x = 0; x < FeatureWidth + 2 * ZeroPaddingSize; x++)
				{
					std::cout << setw(4) << input_vec_zeropadding[i][c][y][x];
				}std::cout << std::endl;
			}std::cout << std::endl; std::cout << std::endl;
		}
	}
	*/
	
	output_vec.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		output_vec[i].resize(FeatureCh);
		for (int c = 0; c < FeatureCh; c++)
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
		for (int c = 0; c < FeatureCh; c++)
		{
			for (int rowStride = 0; rowStride < OutputSize ; rowStride++) {

				for (int colStride = 0; colStride < OutputSize; colStride++) {

					int sum = 0;

					for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {

						for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {

							sum += input_vec_zeropadding[i][c][x][y] * kernel[c][x - stride * rowStride][y - stride * colStride];

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

	const int batch_size = 10; // 이미지 총 갯수
	int ZeroPaddingSize = 1;
	int stride = 1;
	int kernelSize = 3; 

	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS("C:\\cifar\\test10");// 이미지가 저장되어 있는 폴더 경로

	vector<vector<vector<vector<float>>>> input_vec;  // 4D 입력 이미지 데이터
	vector<vector<vector<vector<float>>>> output_1; // 한번 Convolution 후 4D 출력 이미지 데이터



	//입력변수
	int ImageNum = batch_size;
	int FeatureCh = ImgBox[0].first.channels();
	int FeatureHeight = ImgBox[0].first.rows;
	int FeatureWidth = ImgBox[0].first.cols;

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
					input_vec[i][c][y][x] = temp[3 * 32 * y + 3 * x + c];
				}
			}
		}
	}

	// 커널, window, weight
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

	
	output_1 = Convolution(kernelSize, stride, ZeroPaddingSize, ImageNum, FeatureCh, FeatureHeight, FeatureWidth, kernel, input_vec);

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