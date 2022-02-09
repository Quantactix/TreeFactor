#ifndef GUARD_tree_h
#define GUARD_tree_h

#include "common.h"
#include "state.h"
#include "model.h"
#include "json.h"

using json = nlohmann::json;

class tree
{
public:
    // define types
    typedef tree *tree_p;
    typedef const tree *tree_cp;
    typedef std::vector<tree_p> npv;
    typedef std::vector<tree_cp> cnpv;

    //leaf parameters and sufficient statistics
    std::vector<double> theta;
    arma::umat *Xorder;
    arma::mat beta;

    // constructors
    tree() : theta(1, 0.0), Xorder(0), N(0), ID(1), v(0), c_index(0), c(0.0), depth(0), p(0), l(0), r(0), beta(0) {}
    tree(size_t dim_theta) : theta(dim_theta, 0.0), Xorder(0), N(0), ID(1), v(0), c_index(0), c(0.0), depth(0), p(0), l(0), r(0) { (this->beta).resize(dim_theta, 1); }
    tree(size_t dim_theta, arma::umat *Xordermat) : theta(dim_theta, 0.0), Xorder(Xordermat), N(0), ID(1), v(0), c_index(0), c(0.0), depth(0), p(0), l(0), r(0) {}
    tree(size_t dim_theta, size_t depth, size_t N, size_t ID, tree_p p, arma::umat *Xordermat) : theta(dim_theta, 0.0), Xorder(Xordermat), N(N), ID(ID), v(0), c_index(0), c(0.0), depth(depth), p(p), l(0), r(0) { (this->beta).resize(dim_theta, 1); }

    // functions
    void settheta(std::vector<double> &theta) { this->theta = theta; }
    void setv(size_t v) { this->v = v; }
    void setc(double c) { this->c = c; }
    void setc_index(size_t c_index) { this->c_index = c_index; }
    void setN(size_t N) { this->N = N; }
    void setID(size_t ID) { this->ID = ID; }
    void setdepth(size_t depth) { this->depth = depth; }

    size_t getv() const { return v; }
    size_t getdepth() const { return depth; }
    std::vector<double> gettheta() { return theta; }
    double gettheta(size_t ind) const { return theta[ind]; }
    size_t getthetasize() const { return theta.size(); }
    size_t getc_index() const { return c_index; }
    double getc() const { return c; }
    size_t getN() const { return N; }
    tree_p getp() { return p; }
    tree_p getr() { return r; }
    tree_p getl() { return l; }
    size_t getID() { return ID; }

    tree &operator=(const tree &rhs);

    // tree operation functions
    void tonull();                              // delete the tree
    tree_p getptr(size_t nid);                  // get node pointer from node ID, 0 if not there;
    void pr(bool pc = true);                    // to screen, pc is "print child"
    size_t treesize();                          // return number of nodes in the tree
    size_t nnogs();                             // number of nog (no grandchildren nodes)
    size_t nbots();                             // number of leaf nodes
    void getbots(npv &v);                       // return a vector of bottom nodes
    void getnogs(npv &v);                       // get nog nodes
    void getnodes(npv &v);                      // get ALL nodes
    void getnodes(cnpv &v) const;               // get ALL nodes (const)
    tree_p gettop();                            // get pointer to the top node (root node) of the tree
    tree_p bn(arma::mat &x, size_t &row_index); // search tree, find bottom node of the data
    size_t nid() const;                         // nid of the node
    char ntype();                               // node type, t:top, b:bot, n:no grandchildren, i:interior (t can be b);
    bool isnog();
    void cp(tree_p n, tree_cp o);  // copy tree from o to n
    void copy_only_root(tree_p o); // copy tree, point new root to old structure
    friend std::istream &operator>>(std::istream &, tree &);

    // growing functions
    void grow(State &state, Model &model, arma::umat &Xorder);
    void split_Xorder(arma::umat &Xorder_left, arma::umat &Xorder_right, arma::umat &Xorder, size_t split_point, size_t split_var, State &state, Model &model);
    void predict(arma::mat X, arma::vec months, arma::vec &output);

    // input and output to json
    json to_json();
    void from_json(json &j3, size_t dim_theta);

private:
    size_t N;       // number of training data observation in this node
    size_t ID;      // ID of the node
    size_t v;       // index of the variable to split
    size_t c_index; // index of the value to split (index in the Xorder matrix)
    double c;       // raw value to split
    size_t depth;   // depth of the tree

    tree_p p; // pointer to the parent node
    tree_p l; // pointer to left child
    tree_p r; // pointer to right child
};

// io functions
std::istream &operator>>(std::istream &, tree &);
std::ostream &operator<<(std::ostream &, const tree &);

#endif