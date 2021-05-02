#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;
using namespace std;

void matMult(vector<vector<float>>& Y_output,
	vector<vector<float>> Prev_matrix,
	vector<vector<float>> Post_matrix,
	int prev_rows, int prev_colsAndPost_rows, int post_cols) {

	for (int i = 0; i < prev_rows; i++) {
		for (int j = 0; j < post_cols; j++) {
			float x = 0;
			for (int k = 0; k < prev_colsAndPost_rows; k++) {
				x = x + Prev_matrix[i][k] * Post_matrix[k][j];
			}
			Y_output[i][j] = x;
		}
	}
}

void matSum(vector<vector<float>>& Y_output,
	vector<vector<float>> Prev_matrix,
	vector<vector<float>> Post_matrix,
	int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			Y_output[i][j] = Prev_matrix[i][j] + Post_matrix[i][j];
		}
	}
}

void winograd(vector<vector<float>>& Y,
	vector<vector<float>> d/*input*/,
	vector<vector<float>> h/*filter*/) {

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
	matMult(U, Gh, Gt, 4, 3, 4);
	matMult(Btd, Bt, d, 4, 4, 4);
	matMult(V, Btd, B, 4, 4, 4);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			UV[i][j] = U[i][j] * V[i][j];
		}
	}
	matMult(AtUV, At, UV, 2, 4, 4);
	matMult(Y, AtUV, A, 2, 4, 2);

}

void loadingTile(vector<vector<float>>& tile, vector<vector<vector<float>>> input, int rows, int cols, int inch) {

	for (int i = rows; i < rows + 4; i++) {
		for (int j = cols; j < cols + 4; j++) {
			tile[i - rows][j - cols] = input[inch][i][j];
		}
	}

}

void InsertResult(vector<vector<vector<float>>>& Output, vector<vector<vector<float>>>  OutTile, int row, int col) {

	for (int c = 0; c < 4; c++) {
		for (int i = row; i < row + 2; i++) {
			for (int j = col; j < col + 2; j++) {
				Output[c][i][j] = OutTile[c][i - row][j - col];
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


	cout << "Winograd Convolutions 4d! \n\n";
	//ln_Ch  = 3
	//O_Ch = 4
	int batch_size = 2;
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
	}

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
	vector<vector<vector<vector<float>>>> h
		= {
			{
			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} },

			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} },

			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} }
			},

			{
			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} },

			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} },

			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} }
			},

			{
			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} },

			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} },

			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} }
			},

			{
			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} },

			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} },

			{ {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} }
			}
	};


	vector<vector<float>> filter;
	filter.resize(3);
	for (int i = 0; i < 3; i++) filter[i].resize(3);

	vector<vector<float>> tile;
	tile.resize(4);
	for (int i = 0; i < 4; i++) tile[i].resize(4);

	vector<vector<float>> Y;
	Y.resize(2);
	for (int i = 0; i < 2; i++) Y[i].resize(2);


	vector<vector<vector<float>>> OutTile;
	OutTile.resize(4);
	for (int c = 0; c < 4; c++) {
		OutTile[c].resize(2);
		for (int i = 0; i < 2; i++) {
			OutTile[c][i].resize(2);
		}
	}

	vector<vector<vector<vector<float>>>> Output;

	Output.resize(batch_size);
	for (int n = 0; n < batch_size; n++) {
		Output[n].resize(4);
		for (int c = 0; c < 4; c++) {
			Output[n][c].resize(6);
			for (int i = 0; i < 6; i++) {
				Output[n][c][i].resize(6);
			}
		}
	}

	int input_H = 8;
	int input_W = 8;

	for (int n = 0; n < batch_size; n++) 
	{
		for (int row = 0; row < input_H-2; row += 2) 
		{
			for (int col = 0; col < input_W-2; col += 2) 
			{
				for (int outch = 0; outch < 4; outch++) 
				{
					for (int inch = 0; inch < 3; inch++) 
					{
						loadingTile(tile, input[n], row, col, inch);// 계산할 tile 부분만 input에서 가져온다.
						winograd(Y, tile, h[outch][inch]); // winograd convolution 수행
						matSum(OutTile[outch], OutTile[outch], Y, 2, 2);
					}
				}
				InsertResult(Output[n], OutTile, row, col); // 각 outch 에 대한 모든 inch 계산 후 최종 outTile을 output에 채워 넣어준다. 
			}
		}
	}


	std::cout << "Winograd Covlution result" << endl;;
	for (int n = 0; n < batch_size; n++) {
	for (int och = 0; och < 4; och++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				cout << setw(5) << Output[n][och][i][j] << "  ";
			}cout << endl;
		}cout << endl;
	}cout << endl;
	}cout << endl;




	return 0;
}