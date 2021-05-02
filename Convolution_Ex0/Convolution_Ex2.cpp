#include <iostream>
#include <vector>

using namespace std;

int main()
{

	std::cout << "2D Convolution(Correlation) with stride + zeroPading ! \n\n";

	int kernelSize = 3;//ex
	int inputSize = 5; //ex
	int stride = 2;

	int LeftPadding = 1;
	int RightPadding = 1;
	int TopPadding = 1;
	int BottomPadding = 1;
	
	int OutputSize = ((inputSize - kernelSize + 2) / stride) + 1;
	// OutputSize = (N-K+2P)/S + 1
	// (5-3)/1 + 1 = 3  < - stride 1 일때
	// (5-3)/2 + 1 = 2  < - stride 2 일때

	vector<vector<int>> kernel;
	kernel = { {1, 0, 1},
			   {0, 1, 0},
			   {1, 0, 1} };

	vector<vector<int>> input;
	input = { {1,1,1,0,0,},
			  {0,1,1,1,0},
			  {0,0,1,1,1},
			  {0,0,1,1,0},
			  {0,1,1,0,0} };

	vector<vector<float>> output;


	output.resize(OutputSize);
	for (int i = 0; i < OutputSize; i++) output[i].resize(OutputSize);



	std::cout << "===== Value check ===== \n";

	std::cout << "* kernel value \n";
	for (int i = 0; i < kernelSize; i++) {
		for (int j = 0; j < kernelSize; j++) {
			cout << kernel[i][j] << "  ";
		}cout << endl;
	}

	std::cout << "* input value \n";
	for (int i = 0; i < inputSize; i++) {
		for (int j = 0; j < inputSize; j++) {
			cout << input[i][j] << "  ";
		}cout << endl;
	}

	// Adding Left Zero Padding 
	for (int i = 0; i < inputSize; i++) {
		vector<int> temp = input[i];
		input[i].resize(inputSize+1);
		input[i][0] = 0;
		for (int j = 0; j < inputSize; j++) {
			input[i][j + 1] = temp[j];
		}
	}

	std::cout << "* Left Zero Padding value \n";
	for (int i = 0; i < inputSize; i++) {
		for (int j = 0; j < inputSize+1; j++) {
			cout << input[i][j] << "  ";
		}cout << endl;
	}

	// Adding Right Zero Padding 

	for (int i = 0; i < inputSize; i++) {
		input[i].push_back(0);
	}

	std::cout << "* Right Zero Padding value \n";
	for (int i = 0; i < inputSize; i++) {
		for (int j = 0; j < inputSize + 2; j++) {
			cout << input[i][j] << "  ";
		}cout << endl;
	}

	// Adding Bottom Zero Padding 
	
	std::cout << "* Bottom Zero Padding value \n";
	input.resize(inputSize+1);
	input[inputSize].resize(inputSize+2);

	for (int i = 0; i < inputSize+1; i++) {
		for (int j = 0; j < inputSize + 2; j++) {
			cout << input[i][j] << "  ";
		}cout << endl;
	}

	// Adding Top Zero Padding 
	std::cout << "* Top Zero Padding value \n";

	vector<vector<int>> input_TopPadding;

	input_TopPadding.resize(inputSize + 2);
	for (int i = 0; i < inputSize + 2; i++) {
	input_TopPadding[i].resize(inputSize + 2);
	}

	for (int i = 0; i < inputSize + 1; i++) {
		for (int j = 0; j < inputSize + 2; j++) {
			input_TopPadding[i+1][j] = input[i][j];
		}
	}


	input.resize(inputSize + 2);
	for (int i = 0; i < inputSize + 2; i++) {
		input[i].resize(inputSize + 2);
	}


	for (int i = 0; i < inputSize + 2; i++) {
		for (int j = 0; j < inputSize + 2; j++) {
			cout << input_TopPadding[i][j] << "  ";
			input[i][j] = input_TopPadding[i][j];
		}cout << endl;
	}
	
	
	// =========2D Convolultion ===========

	for (int rowStride = 0; rowStride < OutputSize; rowStride++) {
		for (int colStride = 0; colStride < OutputSize; colStride++) {
			int sum = 0;
			for (int i = rowStride * stride; i < rowStride * stride + kernelSize; i++) {
				for (int j = colStride * stride; j < colStride * stride + kernelSize; j++) {
					sum += input[i][j] * kernel[i - stride * rowStride][j - stride * colStride];
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

	// stride = 1일때 결과값 
	//  2  2  3  1  1
	//	1  4  3  4  1
	//	1  2  4  3  3
	//	1  2  3  4  1
	//	0  2  2  1  1

	// stride = 2일때 결과값 
	//  2  3  1
	//	1  4  3
	//	0  2  1
	
	return 0;
}
