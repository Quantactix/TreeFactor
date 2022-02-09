#include "tree.h"
#include <chrono>
#include <ctime>

size_t tree::nid() const
{
    if (!p)
        return 1; //if you don't have a parent, you are the top
    if (this == p->l)
        return 2 * (p->nid()); //if you are a left child
    else
        return 2 * (p->nid()) + 1; //else you are a right child
}

tree::tree_p tree::getptr(size_t nid)
{
    if (this->nid() == nid)
        return this; //found it
    if (l == 0)
        return 0; //no children, did not find it
    tree_p lp = l->getptr(nid);
    if (lp)
        return lp; //found on left
    tree_p rp = r->getptr(nid);
    if (rp)
        return rp; //found on right
    return 0;      //never found it
}

size_t tree::treesize()
{
    if (l == 0)
        return 1; //if bottom node, tree size is 1
    else
        return (1 + l->treesize() + r->treesize());
}

char tree::ntype()
{
    //t:top, b:bottom, n:no grandchildren, i:internal
    if (!p)
        return 't';
    if (!l)
        return 'b';
    if (!(l->l) && !(r->l))
        return 'n';
    return 'i';
}

void tree::pr(bool pc)
{
    size_t d = this->depth;
    size_t id = nid();
    size_t pid;
    if (!p)
        pid = 0; //parent of top node
    else
        pid = p->nid();

    std::string pad(2 * d, ' ');
    std::string sp(", ");
    if (pc && (ntype() == 't'))
        std::cout << "tree size: " << treesize() << std::endl;
    std::cout << pad << "(id,parent): " << id << sp << pid;
    std::cout << sp << "(v,c): " << v << sp << c;
    // std::cout << sp << "theta: " << theta;
    std::cout << sp << "type: " << ntype();
    std::cout << sp << "depth: " << this->depth;
    std::cout << sp << "pointer: " << this << std::endl;

    if (pc)
    {
        if (l)
        {
            l->pr(pc);
            r->pr(pc);
        }
    }
}

bool tree::isnog()
{
    bool isnog = true;
    if (l)
    {
        if (l->l || r->l)
            isnog = false; //one of the children has children.
    }
    else
    {
        isnog = false; //no children
    }
    return isnog;
}

size_t tree::nnogs()
{
    if (!l)
        return 0; //bottom node
    if (l->l || r->l)
    { //not a nog
        return (l->nnogs() + r->nnogs());
    }
    else
    { //is a nog
        return 1;
    }
}

size_t tree::nbots()
{
    if (l == 0)
    { //if a bottom node
        return 1;
    }
    else
    {
        return l->nbots() + r->nbots();
    }
}

void tree::getbots(npv &bv)
{
    if (l)
    { //have children
        l->getbots(bv);
        r->getbots(bv);
    }
    else
    {
        bv.push_back(this);
    }
}

void tree::getnogs(npv &nv)
{
    if (l)
    { //have children
        if ((l->l) || (r->l))
        { //have grandchildren
            if (l->l)
                l->getnogs(nv);
            if (r->l)
                r->getnogs(nv);
        }
        else
        {
            nv.push_back(this);
        }
    }
}

tree::tree_p tree::gettop()
{
    if (!p)
    {
        return this;
    }
    else
    {
        return p->gettop();
    }
}

void tree::getnodes(npv &v)
{
    v.push_back(this);
    if (l)
    {
        l->getnodes(v);
        r->getnodes(v);
    }
}
void tree::getnodes(cnpv &v) const
{
    v.push_back(this);
    if (l)
    {
        l->getnodes(v);
        r->getnodes(v);
    }
}

tree::tree_p tree::bn(arma::mat &x, size_t &row_ind)
{
    // v is variable to split, c is raw value
    // not index in matrix<double>, so compare x[v] with c directly

    if (l == 0)
        return this;
    if (x(row_ind, v) <= c)
    {
        return l->bn(x, row_ind);
    }
    else
    {
        return r->bn(x, row_ind);
    }
}

void tree::tonull()
{
    size_t ts = treesize();
    //loop invariant: ts>=1
    while (ts > 1)
    { //if false ts=1
        npv nv;
        getnogs(nv);
        for (size_t i = 0; i < nv.size(); i++)
        {
            delete nv[i]->l;
            delete nv[i]->r;
            nv[i]->l = 0;
            nv[i]->r = 0;
        }
        ts = treesize(); //make invariant true
    }
    v = 0;
    c = 0;
    p = 0;
    l = 0;
    r = 0;
}

//copy tree tree o to tree n
void tree::cp(tree_p n, tree_cp o)
//assume n has no children (so we don't have to kill them)
//recursion down
// create a new copy of tree in NEW memory space
{
    if (n->l)
    {
        std::cout << "cp:error node has children\n";
        return;
    }

    n->v = o->v;
    n->c = o->c;
    n->theta = o->theta;

    if (o->l)
    { //if o has children
        n->l = new tree;
        (n->l)->p = n;
        cp(n->l, o->l);
        n->r = new tree;
        (n->r)->p = n;
        cp(n->r, o->r);
    }
}

void tree::copy_only_root(tree_p o)
//assume n has no children (so we don't have to kill them)
//NOT LIKE cp() function
//this function pointer new root to the OLD structure
{
    this->v = o->v;
    this->c = o->c;
    this->theta = o->theta;

    if (o->l)
    {
        // keep the following structure, rather than create a new tree in memory
        this->l = o->l;
        this->r = o->r;
        // also update pointers to parents
        this->l->p = this;
        this->r->p = this;
    }
    else
    {
        this->l = 0;
        this->r = 0;
    }
}

//--------------------------------------------------
//operators
tree &tree::operator=(const tree &rhs)
{
    if (&rhs != this)
    {
        tonull();       //kill left hand side (this)
        cp(this, &rhs); //copy right hand side to left hand side
    }
    return *this;
}
//--------------------------------------------------
std::ostream &operator<<(std::ostream &os, const tree &t)
{
    tree::cnpv nds;
    t.getnodes(nds);
    os << nds.size() << std::endl;
    size_t theta_length = nds[0]->getthetasize();
    // cout << "theta length is " << theta_length << endl;
    for (size_t i = 0; i < nds.size(); i++)
    {
        os << nds[i]->nid() << " ";
        os << nds[i]->getv() << " ";
        os << nds[i]->getc() << " ";
        os << nds[i]->getc_index() << " ";
        // for (size_t j = 0; j < theta_length; j++)
        // {
        //     os << " " << nds[i]->gettheta(j);
        // }
        os << nds[i]->beta << " ";
        os << std::endl;
    }
    return os;
}

std::istream &operator>>(std::istream &is, tree &t)
{
    size_t tid, pid;                    //tid: id of current node, pid: parent's id
    std::map<size_t, tree::tree_p> pts; //pointers to nodes indexed by node id
    size_t nn;                          //number of nodes

    t.tonull(); // obliterate old tree (if there)

    //read number of nodes----------
    is >> nn;
    if (!is)
    {
        return is;
    }

    // The idea is to dump string to a lot of node_info structure first, then link them as a tree, by nid

    //read in vector of node information----------
    std::vector<node_info> nv(nn);
    for (size_t i = 0; i != nn; i++)
    {
        is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].theta[0]; // Only works on first theta for now, fix latex if needed
        if (!is)
        {
            return is;
        }
    }

    //first node has to be the top one
    pts[1] = &t; //be careful! this is not the first pts, it is pointer of id 1.
    t.setv(nv[0].v);
    t.setc(nv[0].c);
    t.settheta(nv[0].theta);
    t.p = 0;

    //now loop through the rest of the nodes knowing parent is already there.
    for (size_t i = 1; i != nv.size(); i++)
    {
        tree::tree_p np = new tree;
        np->v = nv[i].v;
        np->c = nv[i].c;
        np->theta = nv[i].theta;
        tid = nv[i].id;
        pts[tid] = np;
        pid = tid / 2;
        if (tid % 2 == 0)
        { //left child has even id
            pts[pid]->l = np;
        }
        else
        {
            pts[pid]->r = np;
        }
        np->p = pts[pid];
    }
    return is;
}

void tree::grow(State &state, Model &model, arma::umat &Xorder)
{
    // this is main growing function of one recursion.
    // conditions to stop growing
    size_t num_obs = Xorder.n_rows; // number of observations in the current node

    bool stop_split = false;

    if (num_obs <= state.min_leaf_size)
    {
        cout << "not enough data" << endl;
        stop_split = true;
    }

    if (this->depth >= state.max_depth)
    {
        cout << "reach max depth " << endl;
        stop_split = true;
    }
    size_t split_var;
    size_t split_point; // index of the split variable and point in the X matrix
    size_t num_obs_left = 0;
    size_t num_obs_right = 0;

    bool splitable = true;

    if (stop_split)
    {
        this->l = 0;
        this->r = 0;

        // update leaf parameters (theta)
        // not necessary?
        model.update_leaf_theta(state, Xorder, this);
        return;
    }
    else
    {
        // split
        model.calculate_criterion(state, Xorder, split_var, split_point, num_obs_left, num_obs_right, this, splitable);

        if (splitable)
        {
            this->v = split_var;
            this->c_index = split_point;
            this->c = state.split_candidates[split_point];
        }
        else
        {
            cout << "break of no good candidate" << endl;
            this->l = 0;
            this->r = 0;

            model.update_leaf_theta(state, Xorder, this);
            return;
        }
    }

    // split to left and right side
    arma::umat Xorder_left(num_obs_left, state.p);
    arma::umat Xorder_right(num_obs_right, state.p);

    // Xorder matrix carries indexing of data in leaves
    split_Xorder(Xorder_left, Xorder_right, Xorder, split_point, split_var, state, model);

    tree::tree_p lchild = new tree((*state.Z).n_cols);
    tree::tree_p rchild = new tree((*state.Z).n_cols);

    this->l = lchild;
    this->r = rchild;

    lchild->depth = this->depth + 1;
    rchild->depth = this->depth + 1;

    // recursion for the next round
    this->l->grow(state, model, Xorder_left);
    this->r->grow(state, model, Xorder_right);
    return;
}

void tree::split_Xorder(arma::umat &Xorder_left, arma::umat &Xorder_right, arma::umat &Xorder, size_t split_point, size_t split_var, State &state, Model &model)
{
    size_t num_obs = Xorder.n_rows;

    double cutvalue = state.split_candidates[split_point];

    size_t left_index;
    size_t right_index;
    for (size_t i = 0; i < state.p; i++)
    {
        left_index = 0;
        right_index = 0;

        // loop over variables
        for (size_t j = 0; j < num_obs; j++)
        {

            // loop over observations
            if ((*state.X)(Xorder(j, i), split_var) <= cutvalue)
            {
                // left side
                Xorder_left(left_index, i) = Xorder(j, i);
                left_index++;
            }
            else
            {
                // right side
                Xorder_right(right_index, i) = Xorder(j, i);
                right_index++;
            }
        }
    }
    return;
}

void tree::predict(arma::mat X, arma::vec months, arma::vec &output)
{
    size_t N_test = X.n_rows;

    tree::tree_p bottom_pointer;
    for (size_t i = 0; i < N_test; i++)
    {
        bottom_pointer = this->bn(X, i);
        output(i) = bottom_pointer->theta[months(i)];
    }

    return;
}

json tree::to_json()
{
    json j;
    if (l == 0)
    {
        std::vector<double> beta_std((this->beta).n_rows);
        for (size_t i = 0; i < (this->beta).n_rows; i++)
        {
            beta_std[i] = this->beta(i, 0);
        }
        j = beta_std;
    }
    else
    {
        j["variable"] = this->v;
        j["cutpoint"] = this->c;
        j["cutpoint_index"] = this->c_index;
        j["nodeid"] = this->nid();
        j["depth"] = this->depth;
        j["left"] = this->l->to_json();
        j["right"] = this->r->to_json();
    }
    return j;
}

void tree::from_json(json &j3, size_t dim_theta)
{
    if (j3.is_array())
    {
        // this is the leaf
        std::vector<double> temp;
        j3.get_to(temp);
        for (size_t i = 0; i < temp.size(); i++)
        {
            this->beta(i, 0) = temp[i];
        }
    }
    else
    {
        // this is an intermediate node
        j3.at("variable").get_to(this->v);
        j3.at("cutpoint").get_to(this->c);
        j3.at("cutpoint_index").get_to(this->c_index);
        j3.at("depth").get_to(this->depth);

        tree *lchild = new tree(dim_theta);
        lchild->from_json(j3["left"], dim_theta);
        tree *rchild = new tree(dim_theta);
        rchild->from_json(j3["right"], dim_theta);

        lchild->p = this;
        rchild->p = this;
        this->l = lchild;
        this->r = rchild;
    }
}
