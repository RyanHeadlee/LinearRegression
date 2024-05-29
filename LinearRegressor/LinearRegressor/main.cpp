#include "LinearRegressor.h"
using namespace std;

int main() {
	LinearRegressor model;

	vector<vector<double>> X = { 
		{0, 1, 2, 3, 4},
		{2, 3, 4, 7, 8},
		{1, 3, 5, 6, 7},
		{8, 9, 10, 3, 5},
		{1, 3, 0, 0, 7}
	};
	vector<double> y = { 0, 2, 25, 4, 9 };
	vector<double> initW = { 3, 2, 20, 1, 0.5 };

	//int cost = model.computeCost(X, y, initW, 0);
	vector<double> dj_dw;
	double dj_db;

	tie(dj_dw, dj_db) = model.computeGradient(X, y, initW, 0);

	return 0;
}