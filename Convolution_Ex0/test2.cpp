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

void activation(vector<float>& activationOutput, vector<float>& activationInput) {
	cout << "===== activation ===== \n";

	activationOutput.resize(activationInput.size());
	for (int i = 0; i < activationInput.size(); i++)
	{
		//activationInput[i] = activationSigmoid(activationInput[i]);
		activationOutput[i] = activationTanh(activationInput[i]);
		//activationInput[i] = activationReLU(activationInput[i]);
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





void matMult(vector<vector<float>> &Y_output, 
	         vector<vector<float>> Prev_matrix, 
	         vector<vector<float>> Post_matrix, 
	         int prev_rows, int prev_colsAndPost_rows, int post_cols) {

	for (int i = 0; i < prev_rows; i++) {
		for (int j = 0; j < post_cols; j++) {
			float x = 0;
			for (int k = 0; k < prev_colsAndPost_rows; k++){
				x = x + Prev_matrix[i][k] * Post_matrix[k][j];
			}
			Y_output[i][j] = x;
		}
	}
}


int main()
{

	cout << "4D([N][C][H][W]) Convolutions ! \n\n";
	cout << "===== 1. Image loading ===== \n";
	
	/*
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


	cout << "===== 4. zeroPadding ===== \n";

	int topPadingSize = 1;
	int bottomPadingSize = 1;
	int leftPadingSize = 1;
	int rightPadingSize = 1;
	int zeroPaddingOutputSize = input_n * input_c * (input_h + topPadingSize + bottomPadingSize) * (input_w + leftPadingSize + rightPadingSize);

	vector<float> inputVecWithZeroPadding(zeroPaddingOutputSize);
	zeroPadding(inputVecWithZeroPadding, inputVec, input_n, input_c, input_h, input_w, topPadingSize, bottomPadingSize, leftPadingSize, rightPadingSize);
	valueCheck(inputVecWithZeroPadding, input_n, input_c, input_h + topPadingSize + bottomPadingSize, input_w + leftPadingSize + rightPadingSize, 1);

	//cout << "===== activation ===== \n";
	//activation(inputVec_ZeroPading);
	//valueCheck(inputVec_ZeroPading, input_n, input_c, input_h + topPadingSize + bottomPadingSize, input_w + leftPadingSize + rightPadingSize, 1);


	cout << "===== 5. Convolution ===== \n";
	int outputCh = 4;
	int kernelSize = 3;
	int stride = 1;
	int outputHeight = ((input_h + topPadingSize + bottomPadingSize - kernelSize) / stride) + 1;
	int outputWidth = ((input_w + leftPadingSize + rightPadingSize - kernelSize) / stride) + 1;
	int convOutputSize = input_n * outputCh * outputHeight * outputWidth;

	vector<float> conv1Output(convOutputSize);

	convolution(conv1Output, inputVecWithZeroPadding, kernel, kernelSize, stride, input_n, input_c, input_h + topPadingSize + bottomPadingSize, input_w + leftPadingSize + rightPadingSize, outputCh);

	valueCheck(conv1Output, input_n, outputCh, outputHeight, outputWidth, 1);
	*/

	vector<vector<float>> d;
	int count = 1;
	d.resize(4);
	for (int i = 0; i < 4; i++) {
		d[i].resize(4);
		for (int j = 0 ; j < 4 ; j ++){
			d[i][j] = count;
			count++;
		}
	}



	vector<vector<float>> h
	 = { {1, 1, 1},
	     {1, 1, 1},
	     {1, 1, 1} };



	vector<vector<float>> B
	 = {      { 1,  0,  0,  0},
			  { 0,  1, -1,  1},
			  {-1,  1,  1,  0},
			  { 0,  0,  0, -1} };

	vector<vector<float>> Bt
	 =      { {1,  0, -1,  0},
			  {0,  1,  1,  0},
			  {0, -1,  1,  0},
			  {0,  1,  0, -1} };


	vector<vector<float>> G
	 = {    {  1,     0,    0},
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

	//Winograd 
	//F(2,3) 1D problem - >      Y = At((G h)*(Bt d))
	//F(2x2, 3x3) 2d problem - > Y = At((G h Gt)*(Bt d B))A (4*4 input에 대해서 3*3 필터를 사용해서 2*2 아웃풋으로 출력)
	//d = input, h = filter, (A, G and B is given value)

	// Calc Process 
	// 1. h = filter 3x3, d = input 4x4 준비
	// 2. U = G h Gt, V = Bt d B 계산
	// 3. At( U*V )A 계산
	// 4. 2x2 계산 결과를 Output에 입력


	matMult(Gh,G,h,4, 3, 3);

	std::cout << "Gh" << endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++) {
			cout<< setw(5) << Gh[i][j] << "  ";
		}cout << endl;
	}cout << endl;

	matMult(U, Gh, Gt, 4, 3, 4);

	std::cout << "U = GhGt" << endl;;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << setw(5) << U[i][j] << "  ";
		}cout << endl;
	}cout << endl;

	//U 준비 끝

	matMult(Btd, Bt, d, 4, 4, 4);
	matMult(V, Btd, B, 4, 4, 4);

	//V 준비 끝
	std::cout << "V = BtdB" << endl;;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << setw(5) << V[i][j] << "  ";
		}cout << endl;
	}cout << endl;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			UV[i][j] = U[i][j] * V[i][j];
		}
	}
	// point wise calc , UV 끝


	matMult(AtUV, At, UV, 2, 4, 4);
	matMult(Y, AtUV, A, 2, 4, 2);

	std::cout << "INPUT(d)" << endl;;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << setw(5) << d[i][j] << "  ";
		}cout << endl;
	}cout << endl;

	std::cout << "Y" << endl;;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			cout << setw(5) << Y[i][j] << "  ";
		}cout << endl;
	}cout << endl;




	// =========2D Convolultion ===========


	vector<vector<float>> output;

	int OutputSize = 2;
	int stride = 1;
	int kernelSize = 3;
	output.resize(OutputSize);
	for (int i = 0; i < OutputSize; i++) output[i].resize(OutputSize);


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
	std::cout << "===== Result check ===== \n";

	for (int i = 0; i < OutputSize; i++) {
		for (int j = 0; j < OutputSize; j++) {
			cout << output[i][j] << "  ";
		}cout << endl;
	}


	return 0;
}