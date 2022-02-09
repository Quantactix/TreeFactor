#include "common.h"
#include "state.h"
#include "APTree.h"
#include "model.h"
#include "json_io.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List TreeFactor_APTree_cpp(arma::vec R, arma::vec Y, arma::mat X, arma::mat Z, arma::mat H, arma::vec portfolio_weight, arma::vec loss_weight, arma::vec stocks, arma::vec months, arma::vec unique_months, arma::vec first_split_var, arma::vec second_split_var, size_t num_stocks, size_t num_months, size_t min_leaf_size = 100, size_t max_depth = 5, size_t num_iter = 30, size_t num_cutpoints = 4, double eta = 1.0, bool equal_weight = false, bool no_H = false, bool abs_normalize = false, bool weighted_loss = false, bool stop_no_gain = false, double lambda_mean = 0, double lambda_cov = 0)
{
    // we assume the number of months is continuous
    std::map<size_t, size_t> months_list;
    assert(num_months == unique_months.n_elem);
    // a mapping from month to index from zero to num_months - 1
    // it is not necessary to normalize months, adjust from zero in the input
    for (size_t i = 0; i < num_months; i++)
    {
        // count from zero
        months_list[unique_months(i)] = i;
    }

    // initialize state class to save data objects
    State state(X, Y, R, Z, H, portfolio_weight, loss_weight, stocks, months, first_split_var, second_split_var, num_months, months_list, num_stocks, min_leaf_size, max_depth, num_cutpoints, equal_weight, no_H, abs_normalize, weighted_loss, stop_no_gain, eta, lambda_mean, lambda_cov);

    APTreeModel model(lambda_cov);

    // calculate Xorder matrix, each index is row index of the data in the X matrix, but sorted from low to high
    arma::umat Xorder(X.n_rows, X.n_cols, arma::fill::zeros);
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }

    // initialize tree class
    APTree root(state.num_months, 1, state.num_obs_all, 1, 0, &Xorder);

    root.setN(X.n_rows);

    // initialize the portfolio at the root node
    model.initialize_portfolio(state, &root);

    // initialize the proper regressor matrix for the criterion
    // Rt ~ Zt * Ft + Ht
    // create a matrix of Zt * Ft + Ht
    model.initialize_regressor_matrix(state);

    bool break_flag = false;

    std::vector<double> criterion_values;

    // out put of split criterion evaluations
    // a list (number of iters), each one has length of all possible candidates
    Rcpp::List all_criterion = Rcpp::List::create();

    arma::vec temp_vec;

    for (size_t iter = 0; iter < num_iter; iter++)
    {
        // main function that grows the tree
        root.grow(break_flag, model, state, iter, criterion_values);

        temp_vec.set_size(criterion_values.size());

        for (size_t i = 0; i < criterion_values.size(); i++)
        {
            temp_vec(i) = criterion_values[i];
        }

        // output vector of all split criterion for debugging
        all_criterion.push_back(temp_vec, to_string(iter));

        if (break_flag)
        {
            break;
        }
    }

    arma::vec leaf_node_index;
    arma::mat all_leaf_portfolio, leaf_weight, ft;

    model.calculate_factor(root, leaf_node_index, all_leaf_portfolio, leaf_weight, ft, state);

    cout << "fitted tree " << endl;
    cout.precision(3);
    cout << root << endl;

    std::stringstream trees;
    Rcpp::StringVector output_tree(1);
    trees.precision(10);
    trees.str(std::string());
    trees << root;
    output_tree(0) = trees.str();

    // return pointer to the tree structure, cannot be restored if saving the environment in R
    // APTree *root_pnt = &root;
    // Rcpp::XPtr<APTree> tree_pnt(root_pnt, true);
    Rcpp::StringVector json_output(1);
    json j = tree_to_json(root);
    json_output[0] = j.dump(4);

    // calculating the pricing error of the factor, run regression
    double loss = model.calculate_R2(state, ft);

    return Rcpp::List::create(
        Rcpp::Named("R") = R,
        Rcpp::Named("X") = X,
        Rcpp::Named("Xorder") = Xorder,
        Rcpp::Named("tree") = output_tree,
        Rcpp::Named("leaf_weight") = leaf_weight,
        Rcpp::Named("leaf_id") = leaf_node_index,
        Rcpp::Named("ft") = ft,
        Rcpp::Named("portfolio") = all_leaf_portfolio,
        Rcpp::Named("json") = json_output,
        Rcpp::Named("R2") = loss,
        Rcpp::Named("all_criterion") = all_criterion);
}