# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

TreeFactor_APTree_cpp <- function(R, Y, X, Z, H, portfolio_weight, loss_weight, stocks, months, unique_months, first_split_var, second_split_var, num_stocks, num_months, min_leaf_size = 100L, max_depth = 5L, num_iter = 30L, num_cutpoints = 4L, eta = 1.0, equal_weight = FALSE, no_H = FALSE, abs_normalize = FALSE, weighted_loss = FALSE, stop_no_gain = FALSE, lambda_mean = 0, lambda_cov = 0) {
    .Call(`_TreeFactor_TreeFactor_APTree_cpp`, R, Y, X, Z, H, portfolio_weight, loss_weight, stocks, months, unique_months, first_split_var, second_split_var, num_stocks, num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, no_H, abs_normalize, weighted_loss, stop_no_gain, lambda_mean, lambda_cov)
}

TreeFactor_APTree_2_cpp <- function(R, Y, X, Z, H, portfolio_weight, loss_weight, stocks, months, unique_months, first_split_var, first_split_mat, second_split_var, third_split_var, deep_split_var, num_stocks, num_months, min_leaf_size = 100L, max_depth = 5L, num_iter = 30L, num_cutpoints = 4L, lambda = 0.0001, equal_weight = FALSE, no_H = FALSE, abs_normalize = FALSE, weighted_loss = FALSE, stop_no_gain = FALSE) {
    .Call(`_TreeFactor_TreeFactor_APTree_2_cpp`, R, Y, X, Z, H, portfolio_weight, loss_weight, stocks, months, unique_months, first_split_var, first_split_mat, second_split_var, third_split_var, deep_split_var, num_stocks, num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, lambda, equal_weight, no_H, abs_normalize, weighted_loss, stop_no_gain)
}

predict_APTree_cpp <- function(X, json_string, months) {
    .Call(`_TreeFactor_predict_APTree_cpp`, X, json_string, months)
}

