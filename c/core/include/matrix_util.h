#pragma once

#include <iostream>
#include <chrono>

std::chrono::time_point<std::chrono::high_resolution_clock> get_time();
void print_elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> begin);
void print_ok();
void print_endl();

// display single element
template<typename T>
std::ostream &_display(std::ostream& os, const T& t){
    return os << t << std::endl;
}

// display multiple elements
template<typename T, typename... Args>
std::ostream &_display(std::ostream& os, const T& t, const Args&... args){
    os << t << ",";
    return _display(os, args...);
}

// display
template<typename... Args>
void display(const Args&... args){
    std::cout << "Printing " << sizeof...(Args) << " arguments: ";
    _display(std::cout, args...);
}

