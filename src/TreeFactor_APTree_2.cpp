#include "common.h"
#include "state.h"
#include "APTree.h"
#include "model.h"
#include "json_io.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List TreeFactor_APTree_2_cpp(arma::vec R, arma::vec Y, arma::mat X, arma::mat Z, arma::mat H, arma::vec portfolio_weight, arma::vec loss_weight, arma::vec stocks, arma::vec months, arma::vec unique_months, arma::vec first_split_var, arma::mat first_split_mat, arma::vec second_split_var, arma::vec third_split_var, arma::vec deep_split_var, size_t num_stocks, size_t num_months, size_t min_leaf_size = 100, size_t max_depth = 5, size_t num_iter = 30, size_t num_cutpoints = 4, double lambda = 0.0001, bool equal_weight = false, bool no_H = false, bool abs_normalize = false, bool weighted_loss = false, bool stop_no_gain = false)
{
    // for the first cut, time is continuous
    std::map<size_t, size_t> months_list_root;
    assert(num_months == unique_months.n_elem);
    for (size_t i = 0; i < num_months; i++)
    {
        // count from zero
        months_list_root[unique_months(i)] = i;
    }

    size_t num_obs_all = X.n_rows;

    // initialize state class to save data objects
    State state(X, Y, R, Z, H, portfolio_weight, loss_weight, stocks, months, first_split_var, second_split_var, third_split_var, deep_split_var, num_months, months_list_root, num_stocks, min_leaf_size, max_depth, num_cutpoints, equal_weight, no_H, abs_normalize, weighted_loss, stop_no_gain, lambda, num_obs_all, first_split_mat);

    APTreeModel model(lambda);

    // calculate Xorder matrix, each index is row index of the data in the X matrix, but sorted from low to high
    arma::umat Xorder(X.n_rows, X.n_cols, arma::fill::zeros);
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }

    // initialize tree class
    APTree root(state.num_months, 1, state.num_obs_all, 1, 0, &Xorder);

    root.setN(X.n_rows);

    model.initialize_portfolio(state, &root);

    model.initialize_regressor_matrix(state);

    bool break_flag = false;

    // grow the first cut with macro variable
    root.grow_APTree_TS(break_flag, model, state);

    // search the tree, find months on the left / right child
    arma::vec leaf_index(X.n_rows);
    model.predict_AP(X, root, months, leaf_index);
    
    // cout << "leaf index " << leaf_index << endl;

    // size_t count_left = 0;
    // size_t count_right = 0;
    // for (size_t i = 0; i < X.n_rows; i++)
    // {
    //     if (leaf_index(i) == 2)
    //     {
    //         count_left++;
    //     }
    //     else
    //     {
    //         count_right++;
    //     }
    // }

    // arma::vec months_left(count_left);
    // arma::vec months_right(count_right);
    // arma::vec index_left(count_left);
    // arma::vec index_right(count_right);
    // size_t temp_ind_left = 0;
    // size_t temp_ind_right = 0;
    // for (size_t i = 0; i < X.n_rows; i++)
    // {
    //     if (leaf_index(i) == 2)
    //     {
    //         months_left(temp_ind_left) = months(i);
    //         index_left(temp_ind_left) = i;
    //         temp_ind_left++;
    //     }
    //     else
    //     {
    //         months_right(temp_ind_right) = months(i);
    //         index_right(temp_ind_right) = i;
    //         temp_ind_right++;
    //     }
    // }

    // arma::vec unique_months_right = arma::sort(arma::unique(months_right));
    // arma::vec unique_months_left = arma::sort(arma::unique(months_left));

    // std::map<size_t, size_t> months_list_left;
    // std::map<size_t, size_t> months_list_right;

    // for (size_t i = 0; i < unique_months_left.n_elem; i++)
    // {
    //     // count from zero
    //     months_list_left[unique_months_left(i)] = i;
    // }

    // for (size_t i = 0; i < unique_months_right.n_elem; i++)
    // {
    //     // count from zero
    //     months_list_right[unique_months_right(i)] = i;
    // }

    // size_t num_months_left = unique_months_left.n_elem;
    // size_t num_months_right = unique_months_right.n_elem;

    // root.getl()->theta.resize(num_months_left);
    // root.getr()->theta.resize(num_months_right);

    // cout << "number of data on left and right " << count_left << " " << count_right << endl;

    // cout << "number of months on left and right " << num_months_left << " " << num_months_right << endl;

    // State state_left(X, Y, R, Z, H, portfolio_weight, loss_weight, stocks, months, first_split_var, second_split_var, third_split_var, deep_split_var, index_left, num_months_left, months_list_left, num_stocks, min_leaf_size, max_depth, num_cutpoints, equal_weight, no_H, abs_normalize, weighted_loss, stop_no_gain, lambda, count_left, first_split_mat);

    // State state_right(X, Y, R, Z, H, portfolio_weight, loss_weight, stocks, months, first_split_var, second_split_var, third_split_var, deep_split_var, index_right, num_months_right, months_list_right, num_stocks, min_leaf_size, max_depth, num_cutpoints, equal_weight, no_H, abs_normalize, weighted_loss, stop_no_gain, lambda, count_right, first_split_mat);

    // state.flag_first_cut = false;

    // state.split_candidates = split_candidates_backup;


    double cutpoint = root.getv();
    double cutvalue = root.getc();


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
    // double loss = model.calculate_R2(state, ft);
    double loss = 0;

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
        Rcpp::Named("cutpoint") = cutpoint,
        Rcpp::Named("cutvalue") = cutvalue);
}