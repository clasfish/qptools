#include <pybind11/pybind11.h>
#include <iostream>
#include <string>
#include "matrix_base.h"
#include "qp.h"
namespace py = pybind11;


py::buffer_info matrix_toBuffer(const matrix* A){
    return py::buffer_info(
        A->begin,         // point to buffer
        sizeof(double),  // item size
        py::format_descriptor<double>::format(),  // python struct-style format descriptor
        2,               // number of dimensions
        {A->nrows, A->ncols}, // buffer dimensions
        {sizeof(double), sizeof(double)*A->nrows}  // strides
    );
}

matrix* matrix_fromBuffer(py::buffer buf){
    py::buffer_info info = buf.request();
    if(info.format != py::format_descriptor<double>::format())
        throw std::runtime_error("Incompatible format: expected a double array");
    if(info.ndim == 1){
        const int size = info.shape[0];
        matrix *A = new matrix(size, 1);
        double *iter=A->begin, *end=iter+size;
        double *iter0 = static_cast<double *>(info.ptr);
        for(; iter<end; iter++, iter0++) *iter = *iter0;
        return A;
    }else if(info.ndim == 2){
        int i, j;
        const int nrows=info.shape[0], ncols=info.shape[1], stride0=info.strides[0]/sizeof(double), stride1=info.strides[1]/sizeof(double);
        matrix *A = new matrix(nrows, ncols);
        double *itera = A->begin;
        const double *B=static_cast<double*>(info.ptr), *iterb;
        for(j=0;j<ncols;j++){
            iterb = B + j * stride1;
            for(i=0;i<nrows;i++,itera++,iterb+=stride0) *itera = *iterb;
        }
        return A;
    }else throw std::runtime_error("Incompatible buffer dimension: " + std::to_string(info.ndim));
}

std::string matrix_repr(const matrix* a){
    return "<qptools.matrix of size (" +
        std::to_string(a->nrows) + "," + std::to_string(a->ncols) + 
        ") at " + 
        std::to_string(reinterpret_cast<uintptr_t>(a->begin)) +
        + ">";
}

PYBIND11_MODULE(core, m){
    m.def("matrix_fromBuffer", &matrix_fromBuffer);

    py::class_<matrix>(m, "matrix", py::buffer_protocol())
        .def(py::init<int, int>())
        .def(py::init<int, int, double>())
        .def(py::init<const matrix&>())
        .def_readonly("nrows", &matrix::nrows)
        .def_readonly("ncols", &matrix::ncols)
        .def_readonly("size", &matrix::size)
        .def("display", &matrix::display)
        .def("__repr__", &matrix_repr)
        .def_buffer(&matrix_toBuffer);

    py::class_<qp1>(m, "qp1")
        .def(
            py::init<const matrix*, const matrix*, const matrix*, const matrix*, const matrix*, const matrix*>(),
            py::arg("P"),
            py::arg("q") = nullptr,
            py::arg("lb") = nullptr,
            py::arg("rb") = nullptr,
            py::arg("G") = nullptr,
            py::arg("h") = nullptr
        )
        .def("solve", &qp1::solve);
        
    py::class_<qp2>(m, "qp2")
        .def(
            py::init<const matrix*, const matrix*, const matrix*, const matrix*, const matrix*, const matrix*>(),
            py::arg("P"),
            py::arg("q") = nullptr,
            py::arg("lb") = nullptr,
            py::arg("rb") = nullptr,
            py::arg("G") = nullptr,
            py::arg("h") = nullptr
        )
        .def("solve", &qp2::solve);
}