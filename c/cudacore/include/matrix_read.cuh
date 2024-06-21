#pragma once
#include <string>
#include "matrix_base.cuh"

matrix* read_csv(const std::string& path, char sep);
