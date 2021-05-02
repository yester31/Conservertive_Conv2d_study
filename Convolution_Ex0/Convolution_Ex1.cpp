#include <iostream>
#include <vector>

using namespace std;

int main()
{

	std::cout << "2D Convolution(Correlation) with stride ! \n\n";

	int kernelSize = 3;//ex
	int inputSize = 5;//ex
	int stride = 2;

	int OutputSize = ((inputSize - kernelSize) / stride) + 1;
	// OutputSize = (N-K+2P)/S + 1 
	// (5-3)/1 + 1 = 3  < - stride 1 일때
	// (5-3)/2 + 1 = 2  < - stride 2 일때

	vector<vector<float>> kernel;
	kernel = { {1, 0, 1},
			   {0, 1, 0},
			   {1, 0, 1} };

	vector<vector<float>> input;
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

	// =========2D Convolultion ===========

	for (int rowStride = 0; rowStride < OutputSize; rowStride++) {
		for (int colStride = 0; colStride < OutputSize; colStride++) {
			int sum = 0;
			for (int i = rowStride * stride; i < rowStride * stride + kernelSize; i++) {
				for (int j = colStride * stride; j < colStride * stride + kernelSize; j++) {
					sum += input[i][j] * kernel[i - stride* rowStride][j - stride* colStride];
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
	//4  4
	//2  4

	// stride = 2일때 결과값 
	//4  3  4
	//2  4  3
	//2  3  4

	return 0;
}
