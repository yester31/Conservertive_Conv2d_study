#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;
using namespace std;

/***************************************************************************
	winograd convolution -> vector 1개 구조로 변경 시도
*****************************************************************************/


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

void matMult(vector<vector<float>>& Y_output, vector<vector<float>> Prev_matrix, vector<vector<float>> Post_matrix, int prev_rows, int prev_colsAndPost_rows, int post_cols) {

	for (int i = 0; i < prev_rows; i++) {
		for (int j = 0; j < post_cols; j++) {
			float x = 0;
			for (int k = 0; k < prev_colsAndPost_rows; k++) {
				x += Prev_matrix[i][k] * Post_matrix[k][j];
			}
			Y_output[i][j] = x;
		}
	}
}

void matrixMul(vector<float>& Y_output, vector<float> Prev_matrix, vector<float> Post_matrix, int prev_rows, int prev_colsAndPost_rows, int post_cols)
{

	for (int i = 0; i < prev_rows; ++i){
		int temp1 = i * prev_colsAndPost_rows;
		int temp2 = i * post_cols;
		for (int j = 0; j < post_cols; ++j)
		{
			float sum = 0;
			for (int k = 0; k < prev_colsAndPost_rows; ++k)
			{
				sum += Prev_matrix[temp1+ k] * Post_matrix[k* post_cols + j];
			}
			Y_output[temp2+ j] = (float)sum;
		}
	}
}

void matSum(vector<vector<float>>& Y_output, vector<vector<float>> Prev_matrix, vector<vector<float>> Post_matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Y_output[i][j] = Prev_matrix[i][j] + Post_matrix[i][j];
		}
	}
}

void matrixSum(vector<float>& Y_output, vector<float> Prev_matrix, vector<float> Post_matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		int temp1 = i * cols;
		for (int j = 0; j < cols; j++) {
			int gidx = temp1 + j;
			Y_output[gidx] = Prev_matrix[gidx] + Post_matrix[gidx];
		}
	}
}
// 수정 해야함.
void winograd(vector<vector<float>>& Y, vector<vector<float>> d/*input*/, vector<vector<float>> h/*filter*/) {

	vector<vector<float>> B
		= { { 1,  0,  0,  0},
				 { 0,  1, -1,  1},
				 {-1,  1,  1,  0},
				 { 0,  0,  0, -1} };

	vector<vector<float>> Bt
		= { {1,  0, -1,  0},
				 {0,  1,  1,  0},
				 {0, -1,  1,  0},
				 {0,  1,  0, -1} };

	vector<vector<float>> G
		= { {  1,     0,    0},
			  {0.5f,  0.5f, 0.5f},
			  {0.5f, -0.5f, 0.5f},
			   {  0,     0,    1} };

	vector<vector<float>> Gt
		= { {   1,  0.5f,  0.5f,  0},
			   {0,  0.5f, -0.5f,  0},
			   {0,  0.5f,  0.5f,  1} };

	vector<vector<float>> A
		= { {1,  0},
			{1,  1},
			{1, -1},
			{0, -1} };

	vector<vector<float>> At
		= { {1, 1,  1,  0},
			{0, 1, -1, -1} };

	vector<vector<float>> Gh;
	Gh.resize(4);
	for (int i = 0; i < 4; i++) Gh[i].resize(3);

	vector<vector<float>> Btd;
	Btd.resize(4);
	for (int i = 0; i < 4; i++) Btd[i].resize(4);

	vector<vector<float>> U;
	U.resize(4);
	for (int i = 0; i < 4; i++) U[i].resize(4);

	vector<vector<float>> V;
	V.resize(4);
	for (int i = 0; i < 4; i++) V[i].resize(4);

	vector<vector<float>> UV;
	UV.resize(4);
	for (int i = 0; i < 4; i++) UV[i].resize(4);

	vector<vector<float>> AtUV;
	AtUV.resize(2);
	for (int i = 0; i < 2; i++) AtUV[i].resize(4);


	matMult(Gh, G, h, 4, 3, 3);
	matMult(U, Gh, Gt, 4, 3, 4); //	U , filter transform
	matMult(Btd, Bt, d, 4, 4, 4);
	matMult(V, Btd, B, 4, 4, 4); // V , input transform

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			UV[i][j] = U[i][j] * V[i][j]; // U * V , element wise matrix multiplication 
		}
	}

	matMult(AtUV, At, UV, 2, 4, 4);
	matMult(Y, AtUV, A, 2, 4, 2); // Y, result transform

}

void loadingTile(vector<vector<float>>& tile, vector<vector<vector<float>>> input, int rows, int cols, int inch) {

	for (int i = rows; i < rows + 4; i++) {
		for (int j = cols; j < cols + 4; j++) {
			tile[i-rows][j-cols] = input[inch][i][j];
		}
	}

}

void loadingTileFromInput(vector<float>& tile, vector<vector<vector<float>>> input, int rows, int cols, int inch) {

	//int temp3 = batchsize * inch
	for (int i = rows; i < rows + 4; i++) {

		int temp1 = (i - rows) * 4;
		//int temp2 = 
		for (int j = cols; j < cols + 4; j++) {

			int gidx = temp1 + (j - cols);

			tile[gidx] = input[inch][i][j];
		}
	}

}

void InsertResult(vector<vector<vector<float>>>& Output, vector<vector<vector<float>>>  OutTile, int row, int col, int out_ch) {

	for (int c = 0; c < out_ch; c++) {
		for (int i = row; i < row + 2; i++) {
			for (int j = col; j < col + 2; j++) {
				Output[c][i][j] = OutTile[c][i-row][j-col];
			}
		}
	}
}


int main()
{
	//Winograd 
	//F(2,3) 1D problem - >      Y = At((G h)*(Bt d))
	//F(2x2, 3x3) 2d problem - > Y = At((G h Gt)*(Bt d B))A (4*4 input에 대해서 3*3 필터를 사용해서 2*2 아웃풋으로 출력)
	//d = input, h = filter, (A, G and B is given value)

	// Calc Process 
	// 1. h = filter 3x3, d = input 4x4 준비
	// 2. U = G h Gt, V = Bt d B 계산
	// 3. At( U*V )A 계산
	// 4. 2x2 계산 결과를 Output에 입력

	clock_t start, finish;
	double  duration;
	const int batch_size = 1; // 사용 할 이미지 갯수

	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS("C:\\cifar\\test10");// 이미지가 저장되어 있는 폴더 경로

	vector<vector<vector<vector<float>>>> input_vec;  // 4D 입력 이미지 데이터
	vector<vector<vector<vector<float>>>> output_1; // 한번 Convolution 후 4D 출력 이미지 데이터

	//입력변수
	int ImageNum = batch_size;
	int FeatureCh = ImgBox[0].first.channels();
	int FeatureHeight = ImgBox[0].first.rows;
	int FeatureWidth = ImgBox[0].first.cols;
	int output_size = FeatureHeight - 2;
	int out_ch = 4;
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

	cout << "Winograd Convolutions 4d! \n\n";
	//ln_Ch  = 3
	//O_Ch = 4
	/*
	vector<vector<vector<vector<float>>>> input;
	int count = 1;
	input.resize(batch_size);
	for (int n = 0; n < batch_size; n++) {
		input[n].resize(3);
		for (int c = 0; c < 3; c++) {
			input[n][c].resize(8);
			for (int i = 0; i < 8; i++) {
				input[n][c][i].resize(8);
				for (int j = 0; j < 8; j++) {
					input[n][c][i][j] = count;
					count++;
				}
			}
		}
	}*/

	vector<vector<vector<float>>> d;
	d.resize(3);
	for (int c = 0; c < 3; c++) {
		d[c].resize(4);
		for (int i = 0; i < 4; i++) {
			d[c][i].resize(4);
		}
	}
	// h[O_Ch][ln_Ch][KernelSize_h][KernelSize_w] 
	// h[4][3][3][3]
	vector<vector<vector<vector<float>>>> h;

	h.resize(out_ch);
	for (int oc = 0; oc < out_ch; oc++) {
		h[oc].resize(FeatureCh);
		for (int ic = 0; ic < FeatureCh; ic++) {
			h[oc][ic].resize(3);
			for (int i = 0; i < 3; i++) {
				h[oc][ic][i].resize(3);
				for (int j = 0; j < 3; j++) {
					h[oc][ic][i][j] = 1;
				}
			}
		}
	}


	vector<vector<float>> tile;
	tile.resize(4);
	for (int i = 0; i < 4; i++) tile[i].resize(4);

	vector<vector<float>> Y;
	Y.resize(2);
	for (int i = 0; i < 2; i++) Y[i].resize(2);


	vector<vector<vector<float>>> OutTile;
	OutTile.resize(out_ch);
	for (int c = 0; c < out_ch; c++) {
		OutTile[c].resize(2);
		for (int i = 0; i < 2; i++) {
			OutTile[c][i].resize(2);
		}
	}

	vector<vector<vector<vector<float>>>> Output;
	Output.resize(batch_size);
	for (int n = 0; n < batch_size; n++) {
		Output[n].resize(out_ch);
		for (int c = 0; c < out_ch; c++) {
			Output[n][c].resize(output_size);
			for (int i = 0; i < output_size; i++) {
				Output[n][c][i].resize(output_size);
			}
		}
	}

	int input_H = FeatureHeight;
	int input_W = FeatureWidth;

	start = clock();
	for (int n = 0; n < batch_size; n++)
	{
		for (int row = 0; row < input_H - 2; row += 2)
		{
			for (int col = 0; col < input_W - 2; col += 2)
			{

				OutTile.resize(out_ch);
				for (int c = 0; c < out_ch; c++) {
					OutTile[c].resize(2);
					for (int i = 0; i < 2; i++) {
						OutTile[c][i].resize(2);
						for (int j = 0; j < 2; j++) {
							OutTile[c][i][j] = 0;
						}
					}
				}

				for (int outch = 0; outch < out_ch; outch++)
				{
					for (int inch = 0; inch < 3; inch++)
					{
						loadingTile(tile, input_vec[n], row, col, inch);// 계산할 tile 부분만 input에서 가져온다.
						winograd(Y, tile, h[outch][inch]); // winograd convolution 수행
						matSum(OutTile[outch], OutTile[outch], Y, 2, 2); // 각 outch에 inch 수만큼 계산한 결과 타이틀 누적 계산
					}
				}
				InsertResult(Output[n], OutTile, row, col, out_ch); // 각 outch 에 대한 모든 inch 계산 후 최종 outTile을 output에 채워 넣어준다. 
			}
		}
	}
	finish = clock();

	duration = (double)(finish-start) / CLOCKS_PER_SEC;


	std::cout << "Winograd Covlution result" << endl;;

	for (int n = 0; n < batch_size; n++) {
		for (int och = 0; och < out_ch; och++) {
			for (int i = 0; i < output_size; i++) {
				for (int j = 0; j < output_size; j++) {
					cout << setw(5) << Output[n][och][i][j] << "  ";
				}cout << endl;
			}cout << endl;
		}cout << endl;
	}cout << endl; cout << endl;


	printf("%2.5f seconds\n", duration);

	return 0;
}