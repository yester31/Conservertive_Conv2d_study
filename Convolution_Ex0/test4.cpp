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
	int rows,  int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			Y_output[i][j] = Prev_matrix[i][j] + Post_matrix[i][j];
		}
	}
}

void winograd(vector<vector<float>>& Y,
	vector<vector<float>> d/*input*/,
	vector<vector<float>> h/*filter*/ ) {

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

	cout << "Winograd Convolutions 3d! \n\n";
	//ln_Ch  = 3
	//O_Ch = 4
	vector<vector<vector<float>>> d;
	int count = 1;
	d.resize(3);
	for(int c = 0; c< 3; c++){
		d[c].resize(4);
		for (int i = 0; i < 4; i++) {
			d[c][i].resize(4);
			for (int j = 0; j < 4; j++) {
				d[c][i][j] = count;
				count++;
			}
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

	vector<vector<float>> Y;
	Y.resize(2);
	for (int i = 0; i < 2; i++) Y[i].resize(2);
	
	vector<vector<vector<float>>> Output;
	Output.resize(4);
	for (int c = 0; c < 4; c++) {
		Output[c].resize(2);
		for (int i = 0; i < 2; i++) {
			Output[c][i].resize(2);
		}
	}

	for (int outch = 0; outch < 4; outch++) {
		for (int inch = 0; inch < 3; inch++) {
		
			winograd(Y, d[inch], h[outch][inch]);
			matSum(Output[outch], Output[outch], Y, 2, 2);
		}
	}
	std::cout << "Winograd Covlution result" << endl;;
	for (int och = 0; och < 4; och++) {
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			cout << setw(5) << Output[och][i][j] << "  ";
		}cout << endl;
	}cout << endl;
	}cout << endl;




	
	return 0;
}