#include <pybind11/pybind11.h>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "matrix_base.cuh"
#include "matrix_util.cuh"
#include "qp.cuh"
namespace py = pybind11;


py::buffer_info matrix_toBuffer(const matrix* A){
    double* a = new double[A->size];
    cudaMemcpy(a, A->begin, A->size * sizeof(double), cudaMemcpyDeviceToHost);
    return py::buffer_info(
        a,
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {A->nrows, A->ncols},
        {sizeof(double), sizeof(double)*A->nrows}
    );
}

matrix* matrix_fromBuffer(py::buffer buf){
    py::buffer_info info = buf.request();
    if(info.format != py::format_descriptor<double>::format())
        throw std::runtime_error("Incompatible format: expected a double array");
    if(info.ndim == 1){
        const int size=info.shape[0];
        std::cout << size << std::endl;
        matrix *A = new matrix(size, 1);
        double *a = new double[size], *iter=a, *end=a+size;
        double *iter0 = static_cast<double *>(info.ptr);
        for(; iter<end; iter++, iter0++) *iter = *iter0;
        cudaMemcpy(A->begin, a, size*sizeof(double), cudaMemcpyHostToDevice);
        delete[] a;
        return A;
    }else if(info.ndim == 2){
        int i, j;
        const int nrows=info.shape[0], ncols=info.shape[1], size=nrows*ncols, stride0=info.strides[0]/sizeof(double), stride1=info.strides[1]/sizeof(double);
        matrix *A = new matrix(nrows, ncols);
        double *a = new double[size], *itera=a;
        const double *b = static_cast<double *>(info.ptr), *iterb;
        for(j=0;j<ncols;j++){
            iterb = b + j * stride1;
            for(i=0;i<nrows;i++,itera++,iterb+=stride0) *itera = *iterb;
        }
        cudaMemcpy(A->begin, a, size*sizeof(double), cudaMemcpyHostToDevice);
        delete[] a;
        return A;
    }else throw std::runtime_error("Incompatible buffer dimension: " + std::to_string(info.ndim));
}

std::string matrix_repr(const matrix* a){
    return "<qptools.matrix of size (" +
        std::to_string(a->nrows) + "," + std::to_string(a->ncols) + 
        ")>";
}


PYBIND11_MODULE(cudacore, m){
    m.def("matrix_fromBuffer", &matrix_fromBuffer);
    py::class_<matrix>(m, "matrix", py::buffer_protocol())
        .def(py::init<int, int>())
        .def(py::init<int, int, double>())
        .def_readonly("nrows", &matrix::nrows)
        .def_readonly("ncols", &matrix::ncols)
        .def_readonly("size", &matrix::size)
        .def("display", &matrix::display)
        .def("__repr__", &matrix_repr)
        .def_buffer(&matrix_toBuffer);
    py::class_<CublasHandle>(m, "CublasHandle")
        .def(py::init<>());
    py::class_<CusolverHandle>(m, "CusolverHandle")
        .def(py::init<>());
    py::class_<qp1>(m, "qp1")
        .def(
            py::init<const CublasHandle&, const CusolverHandle&, const matrix*, const matrix*, const matrix*, const matrix*, const matrix*, const matrix*>(),
            py::arg("_cublas_handle"),
            py::arg("_cusolver_handle"),
            py::arg("P"),
            py::arg("q") = nullptr,
            py::arg("lb") = nullptr,
            py::arg("rb") = nullptr,
            py::arg("G") = nullptr,
            py::arg("h") = nullptr
        )
        .def("solve", &qp1::solve);
}