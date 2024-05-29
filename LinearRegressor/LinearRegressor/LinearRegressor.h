#ifndef LINEAR_REGRESSOR
#define LINEAR_REGRESSOR

#include <vector>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <functional>

class LinearRegressor {
private:
	double w, b;
public:
	LinearRegressor();

	double computeCost(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& w, double b);

	std::tuple<std::vector<double>, double> computeGradient(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& w, double b);
};

#endif