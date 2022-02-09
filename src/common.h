#ifndef GUARD_common_h
#define GUARD_common_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <algorithm>
#include <omp.h>

#include "RcppArmadillo.h"
#include "Rcpp.h"
#include "omp.h"

using namespace std;
using namespace arma;
using namespace Rcpp;

#define LTPI 1.83787706640934536

std::ostream &operator<<(std::ostream &out, const std::vector<double> &v);
std::ostream &operator<<(std::ostream &out, const std::vector<size_t> &v);
std::ostream &operator<<(std::ostream &out, const std::vector<bool> &v);
std::ostream &operator<<(std::ostream &out, const std::vector<std::vector<double>> &v);
std::ostream &operator<<(std::ostream &out, const std::vector<std::vector<size_t>> &v);

double fastLm(const arma::vec &y, const arma::mat &X);

double fastLm_weighted(const arma::vec &y, const arma::mat &X, const arma::vec &weight);

bool sum(std::vector<bool> &v);

class leaf_data
{
public:
    std::vector<double> R;
    std::vector<size_t> months;
    std::vector<size_t> stocks;
    std::vector<double> weight;

    leaf_data(size_t N) : R(N, 0.0), months(N, 0), stocks(N, 0), weight(N, 0) {}
};

struct node_info
{
    std::size_t id; //node id
    std::size_t v;  //variable
    double c;       //cut point // different from BART
    std::vector<double> theta;
};

double log_normal_density(arma::vec &R, arma::mat &cov);

// functions below are for Lasso regression
double soft_c(double a, double lambda);

double lasso_loss(const arma::mat &X, const arma::mat &Y, const arma::vec &beta, double lambda);

arma::vec lasso_fit_standardized(const arma::mat &X, const arma::mat &Y, double lambda, const arma::vec &beta_ini, double eps);

// indepenent sampler of univariate regression model with conjugate prior
Rcpp::List runireg_rcpp_loop(arma::vec const& y, arma::mat const& X, arma::vec const& betabar, arma::mat const& A, double nu, double ssq, size_t R, size_t keep);

void int_to_bin(size_t num, std::vector<size_t> &s);
#endif