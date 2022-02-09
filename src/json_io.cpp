#include "json_io.h"
// JSON

json tree_to_json(APTree &root)
{
    json output;
    output["dim_theta"] = root.theta.size();
    output["num_nodes"] = root.treesize();

    json tree_json = root.to_json();

    output["tree"] = tree_json;

    return output;
}

void json_to_tree(std::string &json_string, APTree &root)
{
    auto j3 = json::parse(json_string);

    size_t dim_theta;
    j3.at("dim_theta").get_to(dim_theta);

    size_t num_nodes;
    j3.at("num_nodes").get_to(num_nodes);

    root.from_json(j3["tree"], dim_theta);

    return;
}
