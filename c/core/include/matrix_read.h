#pragma once
#include "matrix_base.h"

matrix* read_csv(const std::string& path, char sep);
matrix* random_matrix(int nrows, int ncols, int seed);