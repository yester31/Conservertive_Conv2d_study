#include <iostream>
#include <vector>

using namespace std;

int main()
{

    std::cout << "2D Convolution(Correlation) ! \n\n";

	int kernelSize = 3;//ex
	int inputSize = 5;//ex

	vector<vector<float>> kernel;
	kernel = { {1, 0, 1},
			   {0, 1, 0},
	           {1, 0, 1}};

	vector<vector<float>> input;
	input = { {1,1,1,0,0,}, 
			  {0,1,1,1,0}, 
			  {0,0,1,1,1}, 
			  {0,0,1,1,0}, 
			  {0,1,1,0,0}};

	vector<vector<float>> output;
	output.resize(3);
	for (int i = 0; i < 3; i++) output[i].resize(3);



	std::cout << "===== Value check ===== \n";

	std::cout << "* kernel value \n";
	for (int i = 0; i < kernelSize; i++) {
		for (int j = 0 ; j < kernelSize; j++) {
			cout << kernel[i][j] << "  " ;
		}cout<<endl;
	}

	std::cout << "* input value \n";
	for (int i = 0; i < inputSize; i++) {
		for (int j = 0; j < inputSize; j++) {
			cout << input[i][j] << "  ";
		}cout << endl;
	}

	// =========2D Convolultion ===========

	for (int rowStride = 0; rowStride < 3 ; rowStride++){
		for (int colStride = 0; colStride < 3; colStride++) {
			int sum = 0;
		for (int i = rowStride; i < rowStride + 3; i++) {
			for (int j = colStride; j < colStride + 3; j++) {
				sum += input[i][j] * kernel[i-rowStride][j-colStride];
			}
		}
		output[rowStride][colStride] = sum;
	}
	}

	// =========2D Convolultion ===========

	std::cout << "===== Result check ===== \n";

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cout << output[i][j] << "  ";
		}cout << endl;
	}

	// 결과값 
	//4  3  4
	//2  4  3
	//2  3  4

	return 0;
}
