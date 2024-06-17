#include "util.h"

std::chrono::time_point<std::chrono::high_resolution_clock> get_time(){
    return std::chrono::high_resolution_clock::now();
}
void print_elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> begin){
    auto end = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "Elapsed:" << duration << std::endl;
}

void print_ok(){
    std::cout << "ok" << std::endl;
}

void print_endl(){
    std::cout << std::endl;
}