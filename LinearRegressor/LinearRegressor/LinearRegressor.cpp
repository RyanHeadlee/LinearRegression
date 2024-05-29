#include "LinearRegressor.h"

LinearRegressor::LinearRegressor() {
	this->w = 0;
	this->b = 0;
}

double LinearRegressor::computeCost(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& w, double b) {
	int m = X.size();
	double totalCost = 0.0;

	double cost = 0;
	for (int i = 0; i < m; ++i) {
		double f_wb = std::inner_product(std::begin(X[i]), std::end(X[i]), std::begin(w), 0.0) + b;
		cost += pow(f_wb - y[i], 2);
	}

	cost /= (2 * m);
	return cost;
}

std::tuple<std::vector<double>, double> LinearRegressor::computeGradient(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& w, double b) {
	int m = X.size();
	int n = X[0].size();

	std::vector<double> dj_dw(n, 0);
	double dj_db = 0;

	for (int i = 0; i < m; ++i) {
		double err = (std::inner_product(std::begin(X[i]), std::end(X[i]), std::begin(w), 0.0) + b) - y[i];
		for (int j = 0; j < n; ++j) {
			dj_dw[j] += err * X[i][j];
		}
		dj_db += err;
	}
	std::transform(dj_dw.begin(), dj_dw.end(), dj_dw.begin(),
		std::bind(std::divides<double>(), std::placeholders::_1, m));
	dj_db /= double(m);

	return std::make_tuple(dj_dw, dj_db);
}