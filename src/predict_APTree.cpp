#include "common.h"
#include "APTree.h"
#include "model.h"
#include "state.h"
#include "json_io.h"
#include "json.h"

// [[Rcpp::export]]
Rcpp::List predict_APTree_cpp(arma::mat X, Rcpp::StringVector json_string, arma::vec months)
{
    size_t N = X.n_rows;

    arma::vec leaf_index(N);

    std::vector<std::string> j(json_string.size());
    j[0] = json_string(0);

    size_t dim_theta;
    auto temp = json::parse(j[0]);
    temp.at("dim_theta").get_to(dim_theta);

    APTree *root = new APTree(dim_theta);

    json_to_tree(j[0], *root);

    APTreeModel *model = new APTreeModel(1.0);

    model->predict_AP(X, *root, months, leaf_index);

    return Rcpp::List::create(
        Rcpp::Named("leaf_index") = leaf_index);
}
