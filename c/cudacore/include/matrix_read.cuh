#pragma once
#include <string>
#include "matrix_base.cuh"

cumatrix* read_csv(const std::string& path, char sep);
