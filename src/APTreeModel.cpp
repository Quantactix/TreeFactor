#include "model.h"
#include "APTree.h"
////////////////////////////
//
//
//      APTree global split criterion
//
//
////////////////////////////

void APTreeModel::check_node_splitability(State &state, std::vector<APTree *> &bottom_nodes_vec, std::vector<bool> &node_splitability)
{
    APTree::APTree_p node;

    // check node depth and number of data observations
    for (size_t i = 0; i < bottom_nodes_vec.size(); i++)
    {
        node = bottom_nodes_vec[i];

        if (node->getdepth() >= state.max_depth)
        {
            node_splitability[i] = false;
        }
        else if (node->getN() <= state.min_leaf_size)
        {
            node_splitability[i] = false;
        }
        else
        {
            node_splitability[i] = true;
        }
    }

    return;
}

void APTreeModel::calculate_criterion(State &state, std::vector<APTree *> &bottom_nodes_vec, std::vector<bool> &node_splitability, size_t &split_node, size_t &split_var, size_t &split_point, bool &splitable, std::vector<double> &criterion_values)
{
    size_t num_nodes = bottom_nodes_vec.size();
    size_t num_candidates = state.num_cutpoints * state.p;

    // a vector to save split criterion valuation of all nodes, all candidates
    // initialized at infinity
    // the first num_cutpoints * p is for the first node, etc
    criterion_values.resize(num_nodes * num_candidates);
    std::fill(criterion_values.begin(), criterion_values.end(), std::numeric_limits<double>::max());
    // std::vector<double> criterion_values(num_nodes * num_candidates, std::numeric_limits<double>::max());

    // temp_vector stores criterion evaluation of ONE variable
    std::vector<double> temp_vector(state.num_cutpoints);

    // three major sufficient statistics, calculate one for each month
    // sum of weighted returns, sum(w * R)
    arma::vec weighted_return_all(state.num_months, arma::fill::zeros);
    // sum of weights, sum(w)
    arma::vec cumu_weight_all(state.num_months, arma::fill::zeros);
    // number of stocks
    arma::vec num_stocks_all(state.num_months, arma::fill::zeros);

    size_t temp_index;

    // loop over all current leaf nodes
    for (size_t i = 0; i < num_nodes; i++)
    {
        if (!node_splitability[i])
        {
            // if cannot split here, do nothing, the split criterion value will remain infinite
        }
        else
        {
            // this node is splitable, checkout split candidates
            // calculate sufficient statistics for a node
            node_sufficient_stat(state, *(bottom_nodes_vec[i]->Xorder), weighted_return_all, cumu_weight_all, num_stocks_all);

            if (bottom_nodes_vec[i]->getdepth() == 1)
            {
                // depth 1, this is the root
                for (size_t var = 0; var < state.first_split_var->n_elem; var++)
                {
                    // loop over variables, note the constraint on variables for the root
                    temp_index = (size_t)(*state.first_split_var)(var);
                    this->calculate_criterion_one_variable(state, temp_index, bottom_nodes_vec, i, temp_vector, weighted_return_all, cumu_weight_all, num_stocks_all);
                    for (size_t ind = 0; ind < state.num_cutpoints; ind++)
                    {
                        criterion_values[num_candidates * i + temp_index * state.num_cutpoints + ind] = temp_vector[ind];
                    }
                }
            }
            else if (bottom_nodes_vec[i]->getdepth() == 2)
            {
                // depth 2
                for (size_t var = 0; var < state.second_split_var->n_elem; var++)
                {
                    // loop over variables, note the constraint on variables for depth 2
                    temp_index = (size_t)(*state.second_split_var)(var);
                    this->calculate_criterion_one_variable(state, temp_index, bottom_nodes_vec, i, temp_vector, weighted_return_all, cumu_weight_all, num_stocks_all);
                    for (size_t ind = 0; ind < state.num_cutpoints; ind++)
                    {
                        criterion_values[num_candidates * i + temp_index * state.num_cutpoints + ind] = temp_vector[ind];
                    }
                }
            }
            else
            {
                // all other following nodes
                for (size_t var = 0; var < state.p; var++)
                {
                    // loop over variables, there is no constraint, loop over all variables
                    this->calculate_criterion_one_variable(state, var, bottom_nodes_vec, i, temp_vector, weighted_return_all, cumu_weight_all, num_stocks_all);
                    for (size_t ind = 0; ind < state.num_cutpoints; ind++)
                    {
                        criterion_values[num_candidates * i + var * state.num_cutpoints + ind] = temp_vector[ind];
                    }
                }
            }
        }
    }

    // find the lowest split criterion
    size_t lowest_index = 0;
    double temp = criterion_values[0];
    for (size_t i = 1; i < criterion_values.size(); i++)
    {
        if (criterion_values[i] <= temp)
        {
            temp = criterion_values[i];
            lowest_index = i;
        }
    }

    if (temp == std::numeric_limits<double>::max())
    {
        // if all cutpoints have loss infinite, stop split
        splitable = false;
        return;
    }

    state.overall_loss = temp;

    // restore corresponding index of node, cutpoint variable and data index
    size_t temp2;

    split_node = lowest_index / num_candidates;
    temp2 = lowest_index % num_candidates;
    split_var = temp2 / state.num_cutpoints;
    split_point = temp2 % state.num_cutpoints;

    return;
}

void APTreeModel::calculate_criterion_APTree_TS(State &state, std::vector<APTree *> &bottom_nodes_vec, std::vector<bool> &node_splitability, size_t &split_node, size_t &split_var, size_t &split_point, bool &splitable)
{
    size_t num_nodes = bottom_nodes_vec.size();
    size_t num_candidates = state.num_cutpoints * state.p;

    // a vector to save split criterion valuation of all nodes, all candidates
    // initialized at infinity
    // the first num_cutpoints * p is for the first node, etc
    std::vector<double> criterion_values(num_nodes * num_candidates, std::numeric_limits<double>::max());

    std::vector<double> temp_vector(state.num_cutpoints);

    arma::vec weighted_return_all(state.num_months, arma::fill::zeros);
    arma::vec cumu_weight_all(state.num_months, arma::fill::zeros);
    arma::vec num_stocks_all(state.num_months, arma::fill::zeros);

    size_t temp_index;

    for (size_t i = 0; i < num_nodes; i++)
    {
        if (!node_splitability[i])
        {
            // if cannot split here, do nothing, the split criterion value will remained infinite
        }
        else
        {
            // this node is splitable, checkout split candidates
            // calculate sufficient statistics for a node

            node_sufficient_stat(state, *(bottom_nodes_vec[i]->Xorder), weighted_return_all, cumu_weight_all, num_stocks_all);

            // depth 1, root
            for (size_t var = 0; var < state.first_split_mat->n_cols; var++)
            {
                // loop over variables
                temp_index = (size_t)(*state.first_split_var)(var);
                this->calculate_criterion_one_variable_APTree_TS(state, temp_index, bottom_nodes_vec, i, temp_vector, weighted_return_all, cumu_weight_all, num_stocks_all, var);
                for (size_t ind = 0; ind < state.num_cutpoints; ind++)
                {
                    criterion_values[num_candidates * i + temp_index * state.num_cutpoints + ind] = temp_vector[ind];
                }
            }
        }
    }

    // find the lowest split criterion
    size_t lowest_index = 0;
    double temp = criterion_values[0];
    for (size_t i = 1; i < criterion_values.size(); i++)
    {
        if (criterion_values[i] <= temp)
        {
            temp = criterion_values[i];
            lowest_index = i;
        }
    }

    if (temp == std::numeric_limits<double>::max())
    {
        // if all cutpoints have loss infinite, stop split
        splitable = false;
        return;
    }

    state.overall_loss = temp;

    // restore corresponding index of node, cutpoint variable and data index
    size_t temp2;

    split_node = lowest_index / num_candidates;
    temp2 = lowest_index % num_candidates;
    split_var = temp2 / state.num_cutpoints;
    split_point = temp2 % state.num_cutpoints;

    return;
}

void APTreeModel::node_sufficient_stat(State &state, arma::umat &Xorder, arma::vec &weighted_return_all, arma::vec &cumu_weight_all, arma::vec &num_stocks_all)
{
    // This function create basis portfolio for the node
    // Use R not Y
    size_t num_obs = Xorder.n_rows;
    size_t temp_index;
    size_t temp_month;
    size_t temp_month_index;

    // three sufficient statistics
    // weighted return, w * Rt
    weighted_return_all.fill(0.0);
    // cumulative weight, sum of w
    cumu_weight_all.fill(0.0);
    // number of stocks
    num_stocks_all.fill(0.0);

    for (size_t i = 0; i < num_obs; i++)
    {
        temp_index = Xorder(i, 0);
        temp_month = (*state.months)(temp_index);
        temp_month_index = state.months_list->at(temp_month);
        weighted_return_all(temp_month_index) += (*state.R)(temp_index) * (*state.weight)(temp_index);
        cumu_weight_all(temp_month_index) += (*state.weight)(temp_index);
        num_stocks_all(temp_month_index) += 1.0;
    }

    return;
}

void APTreeModel::calculate_criterion_one_variable(State &state, size_t var, std::vector<APTree *> &bottom_nodes_vec, size_t node_ind, std::vector<double> &output, arma::vec &weighted_return_all, arma::vec &cumu_weight_all, arma::vec &num_stocks_all)
{
    // calculate split criterion for one variable at a specific node
    APTree *node = bottom_nodes_vec[node_ind];

    // initialize split criterion, start from infinity
    std::fill(output.begin(), output.end(), std::numeric_limits<double>::max());

    // essentially, the sufficient statistics are two vectors with length num_months;
    // first vector: weight * return
    // second vector: cumulative weight
    // the portfolio is just elementwise ratio of the two vectors
    size_t num_nodes = bottom_nodes_vec.size();
    arma::umat *Xorder = node->Xorder;

    // calculate sufficient statistics of all data here
    size_t temp_index = 0;
    size_t temp_month = 0;
    size_t temp_month_index = 0;

    // next loop over cutpoints, calculate sufficient statistics on left / right side
    // basis porfolio, use R not Y
    arma::vec weighted_return_left(state.num_months, arma::fill::zeros);
    arma::vec cumu_weight_left(state.num_months, arma::fill::zeros);
    arma::vec num_stocks_left(state.num_months, arma::fill::zeros);

    arma::vec weighted_return_right(state.num_months, arma::fill::zeros);
    arma::vec cumu_weight_right(state.num_months, arma::fill::zeros);
    arma::vec num_stocks_right(state.num_months, arma::fill::zeros);

    double cutpoint;
    size_t loop_index = 0;
    arma::mat mu;
    arma::mat sigma;
    arma::mat weight;
    arma::mat ft;
    double weight_sum;

    arma::mat all_portfolio(state.num_months, num_nodes + 1, arma::fill::zeros);
    temp_index = 2; // the FIRST two columns for the candidate split
    for (size_t i = 0; i < num_nodes; i++)
    {
        if (i != node_ind)
        {
            // if it is not the current node
            // copy portfolio return from the leaf directly
            for (size_t ind = 0; ind < state.num_months; ind++)
            {
                all_portfolio(ind, temp_index) = (bottom_nodes_vec[i]->theta)[ind];
            }
            temp_index++;
        }
    }

    // next calculate portfolio returns for current candidate
    for (size_t i = 0; i < state.num_cutpoints; i++)
    {
        // reset all vectors for a new cutpoint
        weighted_return_left.fill(0.0);
        weighted_return_right.fill(0.0);
        cumu_weight_left.fill(0.0);
        cumu_weight_right.fill(0.0);
        num_stocks_left.fill(0.0);
        num_stocks_right.fill(0.0);

        // cout << "dim of all portfolio " << all_portfolio.n_rows << " " << all_portfolio.n_cols << endl;

        // loop over candidates
        cutpoint = state.split_candidates[i];

        // while ((*state.X)((*Xorder)(loop_index, var), var) <= cutpoint)
        // {
        //     // the observation is on the left side
        //     temp_index = (*Xorder)(loop_index, var);              // convert from sorted index (rank) to the original index
        //     temp_month = (*state.months)(temp_index);             // find corresponding month
        //     temp_month_index = state.months_list->at(temp_month); // index of the month in the month_list
        //     // update weighted return, cumulative weight and count of stocks
        //     weighted_return_left(temp_month_index) += (*state.R)(temp_index) * (*state.weight)(temp_index);
        //     cumu_weight_left(temp_month_index) += (*state.weight)(temp_index);
        //     num_stocks_left(temp_month_index) += 1.0;
        //     loop_index++; // index of the current obs in the original Xorder matrix, will be used in the next round until it reaches total number of obs
        //     if (loop_index == (*Xorder).n_rows)
        //     {
        //         // terminating condition, avoid overflow
        //         break;
        //     }
        // }

        // weighted_return_right = weighted_return_all - weighted_return_left;
        // cumu_weight_right = cumu_weight_all - weighted_return_left;
        // num_stocks_right = num_stocks_all - num_stocks_left;


        for(size_t jj = 0; jj < (*Xorder).n_rows; jj ++)
        {
            if ((*state.X)((*Xorder)(jj, var), var) <= cutpoint)
            {
                temp_index = (*Xorder)(jj, var);                      // convert from sorted index (rank) to the original index
                temp_month = (*state.months)(temp_index);             // find corresponding month
                temp_month_index = state.months_list->at(temp_month); // index of the month in the month_list
                // update weighted return, cumulative weight and count of stocks
                weighted_return_left(temp_month_index) += (*state.R)(temp_index) * (*state.weight)(temp_index);
                cumu_weight_left(temp_month_index) += (*state.weight)(temp_index);
                num_stocks_left(temp_month_index) += 1.0;
            }
            else
            {
                temp_index = (*Xorder)(jj, var);                      // convert from sorted index (rank) to the original index
                temp_month = (*state.months)(temp_index);             // find corresponding month
                temp_month_index = state.months_list->at(temp_month); // index of the month in the month_list
                // update weighted return, cumulative weight and count of stocks
                weighted_return_right(temp_month_index) += (*state.R)(temp_index) * (*state.weight)(temp_index);
                cumu_weight_right(temp_month_index) += (*state.weight)(temp_index);
                num_stocks_right(temp_month_index) += 1.0;
            }
        }


        // cout << " ---- " << endl;
        // cout << arma::join_rows(num_stocks_all - num_stocks_left - num_stocks_right, weighted_return_all - weighted_return_left - weighted_return_right, cumu_weight_all - cumu_weight_left - cumu_weight_right) << endl;

        // check stopping conditions such as minimal leaf size, number of stocks
        if (num_stocks_right.min() < state.min_leaf_size || num_stocks_left.min() < state.min_leaf_size || arma::accu(num_stocks_right) == 0 || arma::accu(num_stocks_left) == 0)
        {
            // too few data in the leaf, set criterion as infinity
            output[i] = std::numeric_limits<double>::max();
        }
        else
        {
            // if this candidate is splitable, calculate split criterion
            for (size_t ind = 0; ind < state.num_months; ind++)
            {
                // calculate weighted return for the candidate left / right child leaves
                all_portfolio(ind, 0) = (num_stocks_left(ind) == 0) ? 0 : weighted_return_left(ind) / cumu_weight_left(ind);
                all_portfolio(ind, 1) = (num_stocks_right(ind) == 0) ? 0 : weighted_return_right(ind) / cumu_weight_right(ind);
            }

            mu = arma::mean(all_portfolio, 0); // 0 for column mean
            mu = arma::trans(mu);              // transpose to column vectors
            sigma = arma::cov(all_portfolio);

            size_t n_leafs = mu.n_elem;

            // mean variance efficient weight
            weight = arma::inv(sigma + state.lambda_cov * arma::eye(n_leafs, n_leafs)) * (mu + state.lambda_mean * arma::ones(mu.n_rows, mu.n_cols));

            arma::vec equal_weight(n_leafs);

            equal_weight.fill(1.0 / n_leafs);

            weight = weight * state.eta + (1.0 - state.eta) * equal_weight;

            if (state.abs_normalize)
            {
                weight_sum = arma::accu(arma::abs(weight));
            }
            else
            {
                weight_sum = arma::accu((weight));
            }

            weight = weight / weight_sum;

            // mean variance efficient portfolio
            ft = all_portfolio * weight;

            for (size_t i = 0; i < state.num_obs_all; i++)
            {
                for (size_t j = 0; j < (*state.Z).n_cols; j++)
                {
                    // interaction term, Z_{it} * ft
                    temp_month_index = state.months_list->at((*state.months)(i));
                    this->regressor(i, j) = (*state.Z)(i, j) * ft(temp_month_index, 0);
                }
            }

            if (state.weighted_loss)
            {
                // Loss function, Use Y instead of R
                // pricing error of Y
                output[i] = fastLm_weighted((*state.Y), this->regressor, (*state.loss_weight));
            }
            else
            {
                // no weight on loss function, standard regression
                output[i] = fastLm((*state.Y), this->regressor);
            }

            if (state.stop_no_gain)
            {
                // compare with overall loss, stop split if no gain
                if (output[i] >= state.overall_loss)
                {
                    // if cannot improve overall pricing error, discard this split candidate
                    output[i] = std::numeric_limits<double>::max();
                }
            }
        }

        if (loop_index == (*Xorder).n_rows)
        {
            // if loop_index = number of data, means that all observations belongs to left side
            // not necessary to loop over the next larger cutpoint
            break;
        }
    }

    return;
}

void APTreeModel::calculate_criterion_one_variable_APTree_TS(State &state, size_t var, std::vector<APTree *> &bottom_nodes_vec, size_t node_ind, std::vector<double> &output, arma::vec &weighted_return_all, arma::vec &cumu_weight_all, arma::vec &num_stocks_all, size_t var_ind)
{
    // calculate split criterion for one variable at a specific node
    APTree *node = bottom_nodes_vec[node_ind];

    // initialize split criterion, start from infinity
    std::fill(output.begin(), output.end(), std::numeric_limits<double>::max());

    // essentially, the sufficient statistics are two vectors with length num_months;
    // first vector: weight * return
    // second vector: cumulative weight
    // the portfolio is just elementwise ratio of the two vectors
    size_t num_nodes = bottom_nodes_vec.size();
    arma::umat *Xorder = node->Xorder;

    // // calculate sufficient statistics of all data here
    size_t temp_index;
    size_t temp_month;
    size_t temp_month_index;

    // next loop over cutpoints, calculate sufficient statistics on left / right side
    arma::vec weighted_return_left(state.num_months, arma::fill::zeros);
    arma::vec cumu_weight_left(state.num_months, arma::fill::zeros);
    arma::vec num_stocks_left(state.num_months, arma::fill::zeros);

    arma::vec weighted_return_right(state.num_months, arma::fill::zeros);
    arma::vec cumu_weight_right(state.num_months, arma::fill::zeros);
    arma::vec num_stocks_right(state.num_months, arma::fill::zeros);

    double cutpoint;
    size_t loop_index = 0;
    arma::mat mu;
    arma::mat sigma;
    arma::mat weight;
    arma::mat ft;
    double weight_sum;

    // for time series split, the months on the left / right sides are not the same
    size_t num_months_left = 0;
    size_t num_months_right = 0;

    // number of original data observations on the left / right side
    size_t num_obs_left = 0;
    size_t num_obs_right = 0;

    arma::mat all_portfolio(state.num_months, num_nodes + 1, arma::fill::zeros);
    temp_index = 2; // first two columns for the candidate split

    for (size_t i = 0; i < num_nodes; i++)
    {
        if (i != node_ind)
        {
            for (size_t ind = 0; ind < state.num_months; ind++)
            {
                all_portfolio(ind, temp_index) = (bottom_nodes_vec[i]->theta)[ind];
            }
            temp_index++;
        }
    }

    for (size_t i = 0; i < state.num_cutpoints; i++)
    {
        // reset all vectors for a new cutpoint
        weighted_return_left.fill(0.0);
        weighted_return_right.fill(0.0);
        cumu_weight_left.fill(0.0);
        cumu_weight_right.fill(0.0);
        num_stocks_left.fill(0.0);
        num_stocks_right.fill(0.0);

        cutpoint = (*state.first_split_mat)(i, var_ind);

        while ((*state.X)((*Xorder)(loop_index, var), var) <= cutpoint)
        {
            // the observation is on the left side
            temp_index = (*Xorder)(loop_index, var);
            temp_month = (*state.months)(temp_index);
            temp_month_index = state.months_list->at(temp_month);
            weighted_return_left(temp_month_index) += (*state.R)(temp_index) * (*state.weight)(temp_index);
            cumu_weight_left(temp_month_index) += (*state.weight)(temp_index);
            num_stocks_left(temp_month_index) += 1.0;
            loop_index++;
            if (loop_index == (*Xorder).n_rows)
            {
                // terminating condition, avoid overflow
                break;
            }
        }

        weighted_return_right = weighted_return_all - weighted_return_left;
        cumu_weight_right = cumu_weight_all - cumu_weight_left;
        num_stocks_right = num_stocks_all - num_stocks_left;

        // cout << "some conditions " << endl;
        // cout << (!state.flag_first_cut) << endl;
        // cout << (num_stocks_right.min() < state.min_leaf_size) << endl;
        // cout << (num_stocks_left.min() < state.min_leaf_size) << endl;
        // cout << (arma::accu(num_stocks_right) == 0) << endl;
        // cout << (arma::accu(num_stocks_left) == 0) << endl;

        if ((!state.flag_first_cut) && (num_stocks_right.min() < state.min_leaf_size || num_stocks_left.min() < state.min_leaf_size || arma::accu(num_stocks_right) == 0 || arma::accu(num_stocks_left) == 0))
        {
            // too few data in the leaf
            output[i] = std::numeric_limits<double>::max();
        }
        else
        {

            for (size_t ind = 0; ind < state.num_months; ind++)
            {
                // first column for the left portfolio
                all_portfolio(ind, 0) = (cumu_weight_left(ind) == 0) ? 0 : weighted_return_left(ind) / cumu_weight_left(ind);

                // second column for the right portfolio
                all_portfolio(ind, 1) = (cumu_weight_right(ind) == 0) ? 0 : weighted_return_right(ind) / cumu_weight_right(ind);
            }

            // count how many months on the left
            num_months_left = 0;
            for (size_t tt = 0; tt < num_stocks_left.n_elem; tt++)
            {
                if (num_stocks_left[tt] == 0)
                {
                    num_months_left++;
                }
            }
            num_months_right = state.num_months - num_months_left;

            // cout << "number of months on left " << num_months_left << " " << num_months_right << endl;

            mu = arma::mean(all_portfolio, 0); // 0 for column mean
            mu = arma::trans(mu);              // transpose to column vectors
            sigma = arma::cov(all_portfolio);

            size_t n_leafs = mu.n_elem;

            weight = arma::inv(sigma + state.lambda * arma::eye(n_leafs, n_leafs)) * mu;

            arma::vec equal_weight(n_leafs);

            equal_weight.fill(1.0 / n_leafs);

            weight = weight * state.eta + (1.0 - state.eta) * equal_weight;

            if (state.abs_normalize)
            {
                weight_sum = arma::accu(arma::abs(weight));
            }
            else
            {
                weight_sum = arma::accu((weight));
            }
            weight = weight / weight_sum;

            ft = all_portfolio * weight;

            num_obs_left = num_obs_right = 0;

            for (size_t i = 0; i < state.num_obs_all; i++)
            {
                for (size_t j = 0; j < (*state.Z).n_cols; j++)
                {
                    temp_month_index = state.months_list->at((*state.months)(i));

                    if (num_stocks_left(temp_month_index) == 0)
                    {
                        // no stock in the left leaf, look at the right one
                        this->regressor(i, j) = (*state.Z)(i, j) * all_portfolio(temp_month_index, 1);
                        num_obs_right++;
                    }
                    else
                    {
                        // have stocks in the left leaf, empty at the right one
                        this->regressor(i, j) = (*state.Z)(i, j) * all_portfolio(temp_month_index, 0);
                        num_obs_left++;
                    }
                }
            }

            size_t num_obs_left = loop_index;
            size_t num_obs_right = state.num_obs_all - num_obs_left;
            size_t num_regressor_cols = this->regressor.n_cols;

            arma::vec Y_left(num_obs_left);
            arma::vec Y_right(num_obs_right);
            arma::mat regressor_left(num_obs_left, num_regressor_cols);
            arma::mat regressor_right(num_obs_right, num_regressor_cols);

            if (state.weighted_loss)
            {
                output[i] = fastLm_weighted((*state.Y), this->regressor, (*state.loss_weight));
            }
            else
            {
                output[i] = fastLm((*state.Y), this->regressor);
            }

            if (state.stop_no_gain)
            {
                // compare with overall loss, stop split if no gain
                if (output[i] >= state.overall_loss)
                {
                    // if cannot improve overall pricing error, discard this split candidate
                    output[i] = std::numeric_limits<double>::max();
                }
            }
        }

        if (loop_index == (*Xorder).n_rows)
        {
            // if loop_index = number of data, means that all observations belongs to left side
            // not necessary to loop over the next larger cutpoint
            break;
        }
    }

    return;
}

void APTreeModel::split_node_APTree_TS(State &state, APTree *node, size_t split_var, size_t split_point)
{
    // first, figure out how many are on the left side and right side
    arma::umat *Xorder = node->Xorder;
    size_t num_obs_left = 0;
    size_t num_obs_right = 0;

    // first find the corresponding index in the first_split_var vector
    size_t var = 999;
    for (size_t i = 0; i < state.first_split_var->n_elem; i++)
    {
        if (split_var == (*state.first_split_var)(i))
        {
            var = i;
        }
    }

    for (size_t i = 0; i < node->getN(); i++)
    {
        ((*state.X)((*Xorder)(i, split_var), split_var) <= (*state.first_split_mat)(split_point, var)) ? num_obs_left++ : num_obs_right++;
    }

    double temp_split = (*state.first_split_mat)(split_point, var);

    node->setv(split_var);
    node->setc_index(split_point);
    node->setc(temp_split);

    arma::umat *Xorder_left = new arma::umat(num_obs_left, state.p, arma::fill::zeros);
    arma::umat *Xorder_right = new arma::umat(num_obs_right, state.p, arma::fill::zeros);

    // node->split_Xorder((*Xorder_left), (*Xorder_right), (*Xorder), split_point, split_var, state);

    APTree::APTree_p lchild = new APTree(state.num_months, node->getdepth() + 1, num_obs_left, node->getID() * 2, node, Xorder_left);
    APTree::APTree_p rchild = new APTree(state.num_months, node->getdepth() + 1, num_obs_right, node->getID() * 2 + 1, node, Xorder_right);

    node->setl(lchild);
    node->setr(rchild);

    this->initialize_portfolio(state, lchild);
    this->initialize_portfolio(state, rchild);

    return;
}

void APTreeModel::split_node(State &state, APTree *node, size_t split_var, size_t split_point)
{
    // first, figure out how many are on the left side and right side
    arma::umat *Xorder = node->Xorder;
    size_t num_obs_left = 0;
    size_t num_obs_right = 0;
    for (size_t i = 0; i < node->getN(); i++)
    {
        ((*state.X)((*Xorder)(i, split_var), split_var) <= state.split_candidates[split_point]) ? num_obs_left++ : num_obs_right++;
    }

    double temp_split = state.split_candidates[split_point];

    node->setv(split_var);
    node->setc_index(split_point);
    node->setc(temp_split);

    arma::umat *Xorder_left = new arma::umat(num_obs_left, state.p, arma::fill::zeros);
    arma::umat *Xorder_right = new arma::umat(num_obs_right, state.p, arma::fill::zeros);

    node->split_Xorder((*Xorder_left), (*Xorder_right), (*Xorder), split_point, split_var, state);

    APTree::APTree_p lchild = new APTree(state.num_months, node->getdepth() + 1, num_obs_left, node->getID() * 2, node, Xorder_left);
    APTree::APTree_p rchild = new APTree(state.num_months, node->getdepth() + 1, num_obs_right, node->getID() * 2 + 1, node, Xorder_right);

    node->setl(lchild);
    node->setr(rchild);

    this->initialize_portfolio(state, lchild);
    this->initialize_portfolio(state, rchild);

    return;
}

void APTreeModel::initialize_portfolio(State &state, APTree *node)
{
    // initialize Rt at the given node
    // calculate equal weight / value weight portfolio return of a node
    size_t num_obs = (*node->Xorder).n_rows;
    size_t month;
    size_t row_ind;
    size_t temp_month_index;
    std::vector<double> weight_sum(state.num_months);

    if (state.equal_weight)
    {
        for (size_t i = 0; i < num_obs; i++)
        {
            row_ind = (*node->Xorder)(i, 0);
            month = (*state.months)[row_ind];
            temp_month_index = state.months_list->at(month);
            (node->theta)[temp_month_index] += (*state.R)[row_ind];
            weight_sum[temp_month_index] = weight_sum[temp_month_index] + 1;
        }
    }
    else
    {
        for (size_t i = 0; i < num_obs; i++)
        {
            row_ind = (*node->Xorder)(i, 0);
            month = (*state.months)[row_ind];
            temp_month_index = state.months_list->at(month);
            (node->theta)[temp_month_index] += (*state.R)[row_ind] * (*state.weight)[row_ind];
            weight_sum[temp_month_index] = weight_sum[temp_month_index] + (*state.weight)[row_ind];
        }
    }

    for (size_t i = 0; i < state.num_months; i++)
    {
        (node->theta)[i] = (weight_sum[i] == 0) ? 0.0 : (node->theta)[i] / weight_sum[i];
    }

    return;
}

void APTreeModel::initialize_regressor_matrix(State &state)
{
    // initialize the regressor matrix in the model class
    // used in calculating pricing error
    // pre allocate space to save computing time
    // regress Yt ~ Zt * Ft + Ht
    // Use Y instead of R

    size_t num_obs = state.num_obs_all;
    size_t num_H = (*state.H).n_cols;
    size_t num_Z = (*state.Z).n_cols;

    if (state.no_H)
    {
        this->regressor.resize(num_obs, num_Z);
        this->regressor.fill(arma::fill::zeros);
    }
    else
    {

        this->regressor.resize(num_obs, num_H + num_Z);
        this->regressor.fill(arma::fill::zeros);

        for (size_t i = 0; i < num_obs; i++)
        {
            for (size_t j = 0; j < num_H; j++)
            {
                // first columns leave for the SDF
                this->regressor(i, j + num_Z) = (*state.H)(i, j);
            }
        }
    }
    return;
}

void APTreeModel::predict_AP(arma::mat &X, APTree &root, arma::vec &months, arma::vec &leaf_index)
{
    APTree *leaf;
    for (size_t i = 0; i < X.n_rows; i++)
    {
        leaf = root.bn(X, i);
        leaf_index(i) = leaf->nid();
    }
    return;
}

void APTreeModel::calculate_factor(APTree &root, arma::vec &leaf_node_index, arma::mat &all_leaf_portfolio, arma::mat &leaf_weight, arma::mat &ft, State &state)
{
    std::vector<APTree *> bottom_nodes_vec;
    // once fitting is done, calculate weight of all leaf nodes
    bottom_nodes_vec.resize(0);
    root.getbots(bottom_nodes_vec);

    leaf_node_index.resize(bottom_nodes_vec.size());
    leaf_node_index.fill(arma::fill::zeros);
    all_leaf_portfolio.resize(state.num_months, bottom_nodes_vec.size());
    all_leaf_portfolio.fill(arma::fill::zeros);

    for (size_t i = 0; i < bottom_nodes_vec.size(); i++)
    {
        leaf_node_index(i) = bottom_nodes_vec[i]->nid();
        for (size_t j = 0; j < state.num_months; j++)
        {
            all_leaf_portfolio(j, i) = (bottom_nodes_vec[i]->theta)[j];
        }
    }

    arma::mat mu = arma::mean(all_leaf_portfolio, 0);
    mu = arma::trans(mu);
    size_t n_leafs = mu.n_elem;
    arma::mat sigma = arma::cov(all_leaf_portfolio);

    leaf_weight = arma::inv(sigma + state.lambda_cov * arma::eye(n_leafs, n_leafs)) * (mu + state.lambda_mean * arma::ones(mu.n_rows, mu.n_cols));

    arma::vec equal_weight(n_leafs);

    equal_weight.fill(1.0 / n_leafs);

    leaf_weight = leaf_weight * state.eta + (1.0 - state.eta) * equal_weight;

    double weight_sum;

    if (state.abs_normalize)
    {
        weight_sum = arma::accu(arma::abs(leaf_weight));
    }
    else
    {
        weight_sum = arma::accu((leaf_weight));
    }

    leaf_weight = leaf_weight / weight_sum;

    ft = all_leaf_portfolio * leaf_weight;

    if (arma::accu(ft) < 0)
    {
        // if the average return is negative, short it
        leaf_weight = leaf_weight * (-1.0);
        ft = all_leaf_portfolio * leaf_weight;
    }

    return;
}

double APTreeModel::calculate_R2(State &state, arma::mat &ft)
{
    arma::mat regressor;
    double loss = 0.0;
    size_t temp_month_index;

    if (!state.no_H)
    {
        regressor.resize(state.num_obs_all, (*state.Z).n_cols + (*state.H).n_cols);
        regressor.fill(arma::fill::zeros);

        for (size_t i = 0; i < state.num_obs_all; i++)
        {
            for (size_t j = 0; j < (*state.H).n_cols; j++)
            {
                regressor(i, j + (*state.Z).n_cols) = (*state.H)(i, j);
            }
        }
    }
    else
    {
        regressor.resize(state.num_obs_all, (*state.Z).n_cols);
        regressor.fill(arma::fill::zeros);
    }

    for (size_t i = 0; i < state.num_obs_all; i++)
    {
        for (size_t j = 0; j < (*state.Z).n_cols; j++)
        {
            temp_month_index = state.months_list->at((*state.months)(i));
            regressor(i, j) = (*state.Z)(i, j) * ft(temp_month_index, 0);
        }
    }

    if (state.weighted_loss)
    {
        loss = fastLm_weighted((*state.Y), regressor, (*state.loss_weight));
    }
    else
    {
        loss = fastLm((*state.Y), regressor);
    }

    loss = 1 - loss / arma::accu(pow(*state.Y, 2));

    return loss;
}