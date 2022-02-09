#ifndef GUARD_APTree_h
#define GUARD_APTree_h

#include "common.h"
#include "state.h"
#include "model.h"
#include "json.h"

using json = nlohmann::json;

class State;
class APTreeModel;

class APTree
{
public:
    // define types
    typedef APTree *APTree_p;
    typedef const APTree *APTree_cp;
    typedef std::vector<APTree_p> npv;
    typedef std::vector<APTree_cp> cnpv;

    //leaf parameters and sufficient statistics
    std::vector<double> theta;
    arma::umat *Xorder;

    // constructors
    APTree() : theta(1, 0.0), Xorder(0), N(0), ID(1), v(0), c_index(0), c(0.0), depth(0), p(0), l(0), r(0), iter(0) {}
    APTree(size_t dim_theta) : theta(dim_theta, 0.0), Xorder(0), N(0), ID(1), v(0), c_index(0), c(0.0), depth(0), p(0), l(0), r(0), iter(0) {}
    APTree(size_t dim_theta, arma::umat *Xordermat) : theta(dim_theta, 0.0), Xorder(Xordermat), N(0), ID(1), v(0), c_index(0), c(0.0), depth(0), p(0), l(0), r(0), iter(0) {}
    APTree(size_t dim_theta, size_t depth, size_t N, size_t ID, APTree_p p, arma::umat *Xordermat) : theta(dim_theta, 0.0), Xorder(Xordermat), N(N), ID(ID), v(0), c_index(0), c(0.0), depth(depth), p(p), l(0), r(0), iter(0) {}

    // functions
    void settheta(std::vector<double> &theta) { this->theta = theta; }
    void setv(size_t v) { this->v = v; }
    void setc(double c) { this->c = c; }
    void setc_index(size_t c_index) { this->c_index = c_index; }
    void setN(size_t N) { this->N = N; }
    void setID(size_t ID) { this->ID = ID; }
    void setdepth(size_t depth) { this->depth = depth; }
    void setl(APTree_p l) { this->l = l; }
    void setr(APTree_p r) { this->r = r; }
    void setp(APTree_p p) { this->p = p; }
    void setiter(size_t iter) { this->iter = iter; }

    size_t getv() const { return v; }
    size_t getdepth() const { return depth; }
    std::vector<double> gettheta() { return theta; }
    double gettheta(size_t ind) const { return theta[ind]; }
    size_t getthetasize() const { return theta.size(); }
    size_t getc_index() const { return c_index; }
    double getc() const { return c; }
    size_t getN() const { return N; }
    APTree_p getp() { return p; }
    APTree_p getr() { return r; }
    APTree_p getl() { return l; }
    size_t getID() { return ID; }
    size_t getiter() const { return iter; }

    APTree &operator=(const APTree &rhs);

    // tree operation functions
    void tonull();                                // delete the tree
    APTree_p getptr(size_t nid);                  // get node pointer from node ID, 0 if not there;
    void pr(bool pc = true);                      // to screen, pc is "print child"
    size_t treesize();                            // return number of nodes in the tree
    size_t nnogs();                               // number of nog (no grandchildren nodes)
    size_t nbots();                               // number of leaf nodes
    void getbots(npv &v);                         // return a vector of bottom nodes
    void getnogs(npv &v);                         // get nog nodes
    void getnodes(npv &v);                        // get ALL nodes
    void getnodes(cnpv &v) const;                 // get ALL nodes (const)
    APTree_p gettop();                            // get pointer to the top node (root node) of the tree
    APTree_p bn(arma::mat &x, size_t &row_index); // search tree, find bottom node of the data
    size_t nid() const;                           // nid of the node
    char ntype();                                 // node type, t:top, b:bot, n:no grandchildren, i:interior (t can be b);
    bool isnog();
    void cp(APTree_p n, APTree_cp o); // copy tree from o to n
    void copy_only_root(APTree_p o);  // copy tree, point new root to old structure
    friend std::istream &operator>>(std::istream &, APTree &);

    void split_Xorder(arma::umat &Xorder_left, arma::umat &Xorder_right, arma::umat &Xorder, size_t split_point, size_t split_var, State &state);
    void predict(arma::mat X, arma::vec months, arma::vec &output);

    void grow(bool &break_flag, APTreeModel &model, State &state, size_t &iter, std::vector<double> &criterion_values);
    void grow_APTree_TS(bool &break_flag, APTreeModel &model, State &state);

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
    size_t iter;    // which iteration this node splits

    APTree_p p; // pointer to the parent node
    APTree_p l; // pointer to left child
    APTree_p r; // pointer to right child
};

// io functions
std::istream &operator>>(std::istream &, APTree &);
std::ostream &operator<<(std::ostream &, const APTree &);

#endif