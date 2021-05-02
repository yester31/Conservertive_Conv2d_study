#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


void ZeroPadding(int LeftPadingSize, int RightPadingSize, int TopPadingSize, int BottomPadingSize, vector<vector<vector<vector<float>>>>& ZeroPaddingInput) {

	cout << "===== Zero Padding ===== \n";

	int ImageNum = ZeroPaddingInput.size();
	int FeatureCh = ZeroPaddingInput[0].size();
	int FeatureHeight = ZeroPaddingInput[0][0].size();
	int FeatureWidth = ZeroPaddingInput[0][0][0].size();

	int FeatureHeightWithZeroPadding = FeatureHeight + TopPadingSize + BottomPadingSize;
	int FeatureWidthWithZeroPadding = FeatureWidth + LeftPadingSize + RightPadingSize;
	

	ZeroPaddingInput.resize(ImageNum);
	for (int i = 0; i < ImageNum; i++)
	{
		ZeroPaddingInput[i].resize(FeatureCh);
		for (int c = 0; c < FeatureCh; c++)
		{
			ZeroPaddingInput[i][c].resize(FeatureHeightWithZeroPadding);
			for (int y = FeatureHeight + TopPadingSize - 1; y >= TopPadingSize; y--)
			{
				ZeroPaddingInput[i][c][y].resize(FeatureWidthWithZeroPadding);

				for (int x = FeatureWidth + LeftPadingSize - 1; x >= LeftPadingSize; x--)
				{
					ZeroPaddingInput[i][c][y][x] = ZeroPaddingInput[i][c][y - TopPadingSize][x - LeftPadingSize];
				}
				for (int x = LeftPadingSize-1; x >= 0; x--)
				{
					ZeroPaddingInput[i][c][y][x] = 0;
				}
			}
			for (int p = 0; p < TopPadingSize; p++) {
				ZeroPaddingInput[i][c][p].resize(FeatureWidthWithZeroPadding);
				for (int x = 0; x < FeatureWidth; x++)
				{
					ZeroPaddingInput[i][c][p][x] = 0;
				}
			}
			for (int p = 0; p < BottomPadingSize; p++) {
				ZeroPaddingInput[i][c][FeatureHeightWithZeroPadding - 1 - p].resize(FeatureWidthWithZeroPadding);
			}

		}
	}
}



vector<vector<vector<vector<float>>>> InitWeights(int PrevLayerChannel, int NextLayerNodeNumber, int Height, int Width)
{

	vector<vector<vector<vector<float>>>> kernel;
	//Weights initializer
	int count = 0;
	kernel.resize(NextLayerNodeNumber);
	for (int och = 0; och < NextLayerNodeNumber; och++)
	{
		kernel[och].resize(PrevLayerChannel);
		for (int ch = 0; ch < PrevLayerChannel; ch++)
		{
			kernel[och][ch].resize(Height);
			for (int row = 0; row < Height; row++)
			{
				kernel[och][ch][row].resize(Width);
				for (int col = 0; col < Width; col++)
				{
					kernel[och][ch][row][col] = count;
					count++;
				}
			}
		}
	}
	return kernel;
}

void ValueCheck(vector<vector<vector<vector<float>>>> ValueCheck_input, int offset = 1) {
	int ImageNum = 1;
	if (offset == 0) { ImageNum = ValueCheck_input.size(); }
	int FeatureCh = ValueCheck_input[0].size();
	int FeatureHeight = ValueCheck_input[0][0].size();
	int FeatureWidth = ValueCheck_input[0][0][0].size();
	for (int i = 0; i < ImageNum; i++)
	{
		for (int c = 0; c < FeatureCh; c++)
		{
			for (int y = 0; y < FeatureHeight; y++)
			{
				for (int x = 0; x < FeatureWidth; x++)
				{
					cout << setw(5) << ValueCheck_input[i][c][y][x] << " ";
				}cout << endl;
			}cout << endl; cout << endl;
		}
	}
}

int main() {

	vector<vector<vector<vector<float>>>> kernel = InitWeights(3,3,3,3);

	ValueCheck(kernel, 0);

	ZeroPadding(1, 1, 0, 3, kernel);

	ValueCheck(kernel, 0);


cout << "===== done ===== \n";


return 0;
}