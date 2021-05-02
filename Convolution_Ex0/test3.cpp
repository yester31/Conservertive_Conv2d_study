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
	 clock_t start2, finish2;
	 double  duration;
	 double  duration2;
	cout << "Winograd Convolutions 2d ! \n\n";


	vector<vector<float>> d;
	int count = 1;
	d.resize(4);
	for (int i = 0; i < 4; i++) {
		d[i].resize(4);
		for (int j = 0; j < 4; j++) {
			d[i][j] = count;
			count++;
		}
	}

	vector<vector<float>> h
		= { {1, 1, 1},
			{1, 1, 1},
			{1, 1, 1} };

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

	vector<vector<float>> Y;
	Y.resize(2);
	for (int i = 0; i < 2; i++) Y[i].resize(2);

	start = clock();

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


	finish = clock();

	std::cout << "Winograd Covlution result" << endl;;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			cout << setw(5) << Y[i][j] << "  ";
		}cout << endl;
	}cout << endl;
	duration = (double)(finish - start) ;

	printf("%le seconds\n", duration);



	// =========2D Convolultion ===========


	vector<vector<float>> output;

	int OutputSize = 2;
	int stride = 1;
	int kernelSize = 3;
	output.resize(OutputSize);
	for (int i = 0; i < OutputSize; i++) output[i].resize(OutputSize);
	
	start2 = clock();

	for (int rowStride = 0; rowStride < OutputSize; rowStride++) {
		for (int colStride = 0; colStride < OutputSize; colStride++) {
			int sum = 0;
			for (int i = rowStride * stride; i < rowStride * stride + kernelSize; i++) {
				for (int j = colStride * stride; j < colStride * stride + kernelSize; j++) {
					sum += d[i][j] * h[i - stride * rowStride][j - stride * colStride];
				}
			}
			output[rowStride][colStride] = sum;
		}
	}

	finish2 = clock();

	std::cout << "===== navie Convolution Result  ===== \n";

	for (int i = 0; i < OutputSize; i++) {
		for (int j = 0; j < OutputSize; j++) {
			cout << setw(5) << output[i][j] << "  ";
		}cout << endl;
	}
	duration2 = (double)(finish2 - start2) / CLOCKS_PER_SEC;
	printf("%2.9f seconds\n", duration2);

	return 0;
}