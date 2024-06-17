#include <iostream>
#include "matrix_base.h"
#include "matrix_read.h"
#include "matrix_qp.h"

int main(){
    matrix* P = read_csv("/home/longxin/qpdata2/P.csv", ',');
    //matrix* q = read_csv("/home/longxin/qpdata/q.csv", '');
    matrix* G0 = read_csv("/home/longxin/qpdata2/G0.csv", ',');
    matrix* h0 = read_csv("/home/longxin/qpdata2/h0.csv", ',');
    matrix* G2 = read_csv("/home/longxin/qpdata2/G2.csv", ',');
    matrix* h2 = read_csv("/home/longxin/qpdata2/h2.csv", ',');
    matrix* lb = read_csv("/home/longxin/qpdata2/lb.csv", ',');
    matrix* rb = read_csv("/home/longxin/qpdata2/rb.csv", ',');
    // solver 1
    qp2 solver1(P, nullptr, nullptr, nullptr, G2, h2);
    matrix* x1 = solver1.solve();
    x1->_display(10);
    /* solver 2
    qp2 solver2(P, nullptr, lb, rb, G0, h0);
    matrix* x2 = solver2.solve();
    x2->_display(10);
    /*
    std::cout << solver.n << std::endl;
    std::cout << solver.lbdim << std::endl;
    std::cout << solver.rbdim << std::endl;
    std::cout << solver.gdim << std::endl;
    std::cout << solver.cdim << std::endl;
    std::cout << "------" << std::endl;
    */
}
