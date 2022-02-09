#ifndef GUARD_state_h
#define GUARD_state_h
#include "common.h"

class State
{

public:
    arma::mat *X; // pointer to the charateristics matrix
    arma::vec *Y;
    arma::vec *R; // pointer to the return vector
    arma::mat *R_mat; // pointer to the return matrix, for TSTree only
    arma::mat *Z; // placeholder
    arma::mat *F; // for Bayes tree
    arma::mat *regressor; // for Bayes tree
    arma::mat *H; // placeholder
    arma::vec *weight;
    arma::vec *loss_weight;
    arma::vec *stocks; // pointer to the index of stocks, same number of rows as X
    arma::vec *months; // months indicator
    arma::vec *first_split_var;
    arma::vec *second_split_var;
    // the two vectors below are for the APTree model 2, first cut at Macro variable
    arma::vec *third_split_var;
    arma::vec *deep_split_var;
    arma::mat *first_split_mat; // for APTree model 2 only
    arma::mat *split_candidate_mat;
    std::map<size_t, size_t> *months_list; // list of UNIQUE months

    size_t num_obs_all;
    size_t num_stocks;
    size_t num_months;
    size_t min_leaf_size;
    size_t max_depth;
    size_t num_cutpoints;
    size_t num_regressors; // for Bayes tree
    size_t p;              // number of charateristics
    std::vector<double> split_candidates;
    bool equal_weight;
    bool no_H;
    bool abs_normalize;
    bool weighted_loss;
    bool stop_no_gain;
    double overall_loss;
    double sigma;
    double tau;
    double lambda;
    double lambda_mean;
    double lambda_cov;
    double eta;
    bool flag_first_cut;

    // prior parameters for the Bayes tree
    double a;
    double b;
    double xi_normal;
    double xi_spike;
    double xi_slab;
    size_t p_normal_prior;
    size_t p_spike_slab;

    // state for APTree model
    State(arma::mat &X, arma::vec &Y, arma::vec &R, arma::mat &Z, arma::mat &H, arma::vec &portfolio_weight, arma::vec &loss_weight, arma::vec &stocks, arma::vec &months, arma::vec &first_split_var, arma::vec &second_split_var, size_t &num_months, std::map<size_t, size_t> &months_list, size_t &num_stocks, size_t &min_leaf_size, size_t &max_depth, size_t &num_cutpoints, bool &equal_weight, bool &no_H, bool &abs_normalize, bool &weighted_loss, bool &stop_no_gain, double &eta, double &lambda_mean, double &lambda_cov)
    {
        this->X = &X;
        this->Y = &Y;
        this->R = &R;
        this->Z = &Z;
        this->H = &H;
        this->weight = &portfolio_weight;
        this->loss_weight = &loss_weight;
        this->stocks = &stocks;
        this->months = &months;
        this->months_list = &months_list;
        this->first_split_var = &first_split_var;
        this->second_split_var = &second_split_var;
        this->third_split_var = 0;
        this->deep_split_var = 0;
        this->num_months = num_months;
        this->num_stocks = num_stocks;
        this->min_leaf_size = min_leaf_size;
        this->max_depth = max_depth;
        this->num_cutpoints = num_cutpoints;
        this->split_candidates.resize(num_cutpoints);
        this->p = X.n_cols;
        this->num_obs_all = X.n_rows;
        this->equal_weight = equal_weight;
        this->no_H = no_H;
        this->abs_normalize = abs_normalize;
        this->weighted_loss = weighted_loss;
        this->stop_no_gain = stop_no_gain;
        this->overall_loss = std::numeric_limits<double>::max();
        this->sigma = 0.0;
        this->tau = 0.0;
        this->lambda = 0.0;
        this->eta = eta;
        this->first_split_mat = 0;
        this->num_regressors = 0;
        this->lambda_mean = lambda_mean;
        this->lambda_cov = lambda_cov;
        for (size_t i = 0; i < num_cutpoints; i++)
        {
            split_candidates[i] = 2.0 / (num_cutpoints + 1) * (i + 1) - 1;
        }

        cout << "The split value candidates are " << split_candidates << endl;
    }

    // state for APTree model2
    State(arma::mat &X, arma::vec &Y, arma::vec &R, arma::mat &Z, arma::mat &H, arma::vec &portfolio_weight, arma::vec &loss_weight, arma::vec &stocks, arma::vec &months, arma::vec &first_split_var, arma::vec &second_split_var, arma::vec &third_split_var, arma::vec &deep_split_var, size_t &num_months, std::map<size_t, size_t> &months_list, size_t &num_stocks, size_t &min_leaf_size, size_t &max_depth, size_t &num_cutpoints, bool &equal_weight, bool &no_H, bool &abs_normalize, bool &weighted_loss, bool &stop_no_gain, double &lambda, size_t &num_obs_all, arma::mat &first_split_mat)
    {
        this->X = &X;
        this->Y = &Y;
        this->R = &R;
        this->Z = &Z;
        this->H = &H;
        this->weight = &portfolio_weight;
        this->loss_weight = &loss_weight;
        this->stocks = &stocks;
        this->months = &months;
        this->months_list = &months_list;
        this->first_split_var = &first_split_var;
        this->second_split_var = &second_split_var;
        this->third_split_var = &third_split_var;
        this->deep_split_var = &deep_split_var;
        this->num_months = num_months;
        this->num_stocks = num_stocks;
        this->min_leaf_size = min_leaf_size;
        this->max_depth = max_depth;
        this->num_cutpoints = num_cutpoints;
        this->split_candidates.resize(num_cutpoints);
        this->p = X.n_cols;
        this->equal_weight = equal_weight;
        this->no_H = no_H;
        this->abs_normalize = abs_normalize;
        this->weighted_loss = weighted_loss;
        this->stop_no_gain = stop_no_gain;
        this->overall_loss = std::numeric_limits<double>::max();
        this->sigma = 0.0;
        this->tau = 0.0;
        this->lambda = lambda;
        this->eta = 0.0;
        this->num_obs_all = num_obs_all;
        this->first_split_mat = &first_split_mat;
        this->num_regressors = 0;
        for (size_t i = 0; i < num_cutpoints; i++)
        {
            split_candidates[i] = 2.0 / (num_cutpoints + 1) * (i + 1) - 1;
        }

        cout << "The split value candidates are " << split_candidates << endl;
    }

};

#endif