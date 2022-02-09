predict.APTree = function(model, X, R, months, weight = NULL)
{
    # make sure that months begin with zero

    X = as.matrix(X)

    N = dim(X)[1]

    unique_months = sort(unique(months))

    output = .Call(`_TreeFactor_predict_APTree_cpp`, X, model$json, months)

    # the C++ function returns ID of leaves that the observation is in
    # next calculate corresponding portfolio in R

    if(is.null(weight))
    {
        weight = rep(1, N)
    }

    num_months = length(unique_months)

    num_leafs = length(model$leaf_id)

    portfolio = matrix(0, num_months, num_leafs)

    weight_portfolio = matrix(0, num_months, num_leafs)

    for(i in 1:N)
    {
        # loop over data
        # add one to start from one rather than zero
        temp_month = months[i]

        temp_month_index = which(unique_months == temp_month)
        
        temp_leaf = which(model$leaf_id == output$leaf_index[i])

        portfolio[temp_month_index, temp_leaf] = portfolio[temp_month_index, temp_leaf] + weight[i] * R[i]

        weight_portfolio[temp_month_index, temp_leaf] = weight_portfolio[temp_month_index, temp_leaf] + weight[i]
    }

    for(i in 1:length(portfolio))
    {
        if(weight_portfolio[i]!=0)
        {
            portfolio[i] = portfolio[i] / weight_portfolio[i]
        }else{
            portfolio[i] = 0
        }
    }

    output$portfolio = portfolio

    output$ft = portfolio %*% model$leaf_weight

    return(output)
}