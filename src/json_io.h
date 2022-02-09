#ifndef GUARD_json_io_
#define GUARD_json_io_

#include "APTree.h"
#include "tree.h"

json tree_to_json(APTree &root);

void json_to_tree(std::string &json_string, APTree &root);

#endif