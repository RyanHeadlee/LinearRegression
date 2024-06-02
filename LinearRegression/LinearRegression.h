#ifndef LINEAR_REGRESSION
#define LINEAR_REGRESSION

#include <vector>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <limits>
#include <omp.h>

class LinearRegression {
private:
	std::vector<double> bW, fitHist;
	double bB, lambda, alpha;
	int numIter, degree;

	// Private methods
	double computeCost(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& w, double b);
	std::tuple<std::vector<double>, double> computeGradient(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& w, double b);
	std::tuple<std::vector<double>, double, std::vector<double>> gradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& wInit, double bInit);
	std::vector<std::vector<double>> transformToPolynomial(const std::vector<std::vector<double>>& X);
public:
	// Constructor
	LinearRegression();

	void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double alpha = 0.01, int numIter = 1000, double lambda = 0, int degree = 1);
	std::vector<double> predict(const std::vector<std::vector<double>>& X);

	// Evaluation methods
	double meanSquaredError(const std::vector<double>& yTrue, const std::vector<double>& predictions);
	double rSquared(const std::vector<double>& yTrue, const std::vector<double>& predictions);
	double rootMSError(const std::vector<double>& yTrue, const std::vector<double>& predictions);

	// Getters
	std::vector<double> getFitHistory() { return fitHist; }
	std::tuple<std::vector<double>, double> getBestWeights() { return { bW, bB }; }
};

#endif