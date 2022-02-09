
TreeFactor_APTree_2 <- function(R, Y, X, Z, H, portfolio_weight, loss_weight, stocks, months, first_split_var, first_split_point, second_split_var, third_split_var, deep_split_var, num_stocks, num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, lambda = 0.0001, equal_weight = FALSE, no_H = FALSE, abs_normalize = FALSE, weighted_loss = FALSE, stop_no_gain = FALSE) {
    R = as.matrix(R)
    Y = as.matrix(Y)
    X = as.matrix(X)
    Z = as.matrix(Z)
    H = as.matrix(H)
    
    unique_months = sort(unique(months))

    output = .Call(`_TreeFactor_TreeFactor_APTree_2_cpp`, R, Y, X, Z, H, portfolio_weight, loss_weight, stocks, months, unique_months, first_split_var, first_split_point, second_split_var, third_split_var, deep_split_var, num_stocks, num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, lambda, equal_weight, no_H, abs_normalize, weighted_loss, stop_no_gain)

    class(output) = "APTree"

    return(output)
}



