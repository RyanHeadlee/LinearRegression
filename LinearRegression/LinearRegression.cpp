#include "LinearRegression.h"

LinearRegression::LinearRegression() : bB(0), lambda(0), alpha(0), numIter(0), degree(1) {}

double LinearRegression::computeCost(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& w, double b) {
	int m = static_cast<int>(X.size());

	double cost = 0;

	#pragma omp parallel for reduction(+:cost)
	for (int i = 0; i < m; ++i) {
		// Get the dot product of w and X[i] and add b. (f_wb = w * X[i] + b)
		double f_wb = std::inner_product(std::begin(X[i]), std::end(X[i]), std::begin(w), 0.0) + b;
		cost += pow(f_wb - y[i], 2);
	}

	cost /= (2 * m);


	// L2 Rregularization if lambda is non-default
	double regTerm = 0.0;
	if (this->lambda != 0) {
		#pragma omp parallel for reduction(+:regTerm)
		for (int i = 0; i < w.size(); ++i) {
			regTerm += w[i] * w[i];
		}
		regTerm *= lambda / (2 * X.size());
	}

	return cost + regTerm;
}

std::tuple<std::vector<double>, double> LinearRegression::computeGradient(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& w, double b) {
	int m = static_cast<int>(X.size());
	int n = static_cast<int>(X[0].size());

	std::vector<double> dj_dw(n, 0);
	double dj_db = 0;

	# pragma omp parallel for reduction(+:dj_db)
	for (int i = 0; i < m; ++i) {
		// Get the dot product of X[i] and w, add b and subtract by the y[i]. (err = f_wb - y[i])
		double err = (std::inner_product(std::begin(X[i]), std::end(X[i]), std::begin(w), 0.0) + b) - y[i];
		#pragma omp parallel for
		for (int j = 0; j < n; ++j) {
			dj_dw[j] += err * X[i][j];
		}
		dj_db += err;
	}

	// Divide dj_dw by m
	std::transform(dj_dw.begin(), dj_dw.end(), dj_dw.begin(), std::bind(std::divides<double>(), std::placeholders::_1, m));

	dj_db /= double(m);

	// L2 Regularization if lambda is non-default
	if (this->lambda != 0) {
		#pragma omp parallel for
		for (int i = 0; i < w.size(); ++i) {
			dj_dw[i] += lambda * w[i] / X.size();
		}
	}

	return { dj_dw, dj_db };
}

std::tuple<std::vector<double>, double, std::vector<double>> LinearRegression::gradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<double>& wInit, double bInit) {
	std::vector<double> w = wInit;
	double b = bInit;
	std::vector<double> jHist;

	for (int i = 0; i < numIter; ++i) {
		auto [dj_dw, dj_db] = this->computeGradient(X, y, w, b);

		// Multiply dj_dw by alpha. (alpha * dj_dw)
		std::transform(dj_dw.begin(), dj_dw.end(), dj_dw.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, alpha));

		// Subtract (alpha * dj_dw) from w, and set w to this value. w = w - (alpha * dj_dw)
		std::transform(w.begin(), w.end(), dj_dw.begin(), w.begin(), std::minus<double>());

		// Set new b value
		b -= alpha * dj_db;

		if (i < 100000)
			jHist.emplace_back(this->computeCost(X, y, w, b));
	}

	return { w, b, jHist };
}

std::vector<std::vector<double>> LinearRegression::transformToPolynomial(const std::vector<std::vector<double>>& X) {
	std::vector<std::vector<double>> X_poly;

	for (const auto& row : X) {
		std::vector<double> newRow;
		for (double val : row) {
			for (int d = 0; d < degree; ++d) {
				newRow.emplace_back(pow(val, d + 1));
			}
		}
		X_poly.emplace_back(newRow);
	}

	return X_poly;
}

void LinearRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double alpha, int numIter, double lambda, int degree) {
	try {
		if (X.size() != y.size()) throw std::invalid_argument("Number of rows in X must match the number of target values in y");
		if ((X.empty() || X[0].empty()) || y.empty()) throw std::invalid_argument("X and y must be a valid Matrix");
		if (degree < 1) throw std::invalid_argument("Degree must be greater than 0");
		if (alpha <= 0) throw std::invalid_argument("Alpha cannot be non-positive");
		if (numIter < 1) throw std::invalid_argument("Number of iterations must be greater than 0");

		// Set private data members to specified values or default
		this->lambda = lambda;
		this->alpha = alpha;
		this->numIter = numIter;
		this->degree = degree;

		// If degree greater than 1, transform matrix into polynomial based on degree
		std::vector<std::vector<double>> copyX = X;
		if (this->degree > 1) copyX = this->transformToPolynomial(X);

		// Get # of cols, set wInit to [0_1, 0_2, ..., 0_n], set bInit to 0
		int n = static_cast<int>(copyX[0].size());
		std::vector<double> wInit(n, 0);
		double bInit = 0;

		// Call gradient descent to get best weights
		auto [w, b, jHist] = this->gradientDescent(copyX, y, wInit, bInit);

		// Set best weights
		bW = w;
		bB = b;
		fitHist = jHist;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
	}
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) {
	try {
		if (X.empty() || X[0].empty()) throw std::invalid_argument("X cannot be an empty vector");

		// If degree greater than 1, transform matrix into polynomial based on degree
		std::vector<std::vector<double>> copyX = X;
		if (this->degree > 1) copyX = this->transformToPolynomial(X);

		int m = static_cast<int>(copyX.size());
		std::vector<double> predictions(m);

		#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			double f_wb = std::inner_product(std::begin(copyX[i]), std::end(copyX[i]), std::begin(bW), 0.0) + bB;
			predictions[i] = f_wb;
		}

		return predictions;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return std::vector<double>();
	}
}

double LinearRegression::meanSquaredError(const std::vector<double>& yTrue, const std::vector<double>& predictions) {
	try {
		if (yTrue.empty() || predictions.empty()) throw std::invalid_argument("yTrue or predictions cannot be an empty vector");
		if (yTrue.size() != predictions.size()) throw std::invalid_argument("yTrue and predictions must be the same size");

		double error = 0;
		int m = static_cast<int>(yTrue.size());

		#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			error += pow((predictions[i] - yTrue[i]), 2);
		}

		error /= m;

		return error;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return std::numeric_limits<double>::quiet_NaN();
	}
}

// R^2 = 1 - (RSS / TSS); 
// RSS = sum((y_i - prediction_i)^2) for all i; 
// TSS = sum((y_i - mean(y))^2) for all i;
double LinearRegression::rSquared(const std::vector<double>& yTrue, const std::vector<double>& predictions) {
	try {
		if (yTrue.empty() || predictions.empty()) throw std::invalid_argument("yTrue or predictions cannot be an empty vector");
		if (yTrue.size() != predictions.size()) throw std::invalid_argument("yTrue and predictions must be the same size");

		double rss = 0;
		int m = static_cast<int>(yTrue.size());

		// Calculate RSS
		#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			rss += pow(yTrue[i] - predictions[i], 2);
		}

		// Calculate TSS
		double mean = std::reduce(std::begin(yTrue), std::end(yTrue)) / double(m);
		double tss = 0;
		#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			tss += pow(yTrue[i] - mean, 2);
		}

		if (tss == 0) throw std::runtime_error("Total sum of squares (TSS) is zero, R^2 is undefined");

		return 1 - (rss / tss);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return std::numeric_limits<double>::quiet_NaN();
	}
}

// Root Mean Squared Error: rmse = sqrt(sum((prediction_i - y_i)^2) / # of rows)
double LinearRegression::rootMSError(const std::vector<double>& yTrue, const std::vector<double>& predictions) {
	try {
		if (yTrue.empty() || predictions.empty()) throw std::invalid_argument("yTrue or predictions cannot be an empty vector");
		if (yTrue.size() != predictions.size()) throw std::invalid_argument("yTrue and predictions must be the same size");

		int m = static_cast<int>(yTrue.size());
		double rmse = 0;
		#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			rmse += pow(predictions[i] - yTrue[i], 2);
		}
		rmse = sqrt(rmse / m);

		return rmse;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return std::numeric_limits<double>::quiet_NaN();
	}
}