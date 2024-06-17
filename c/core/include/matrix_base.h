#pragma once
class matrix{
    public:
        const int nrows, ncols, size;
        double* const begin;
        matrix(int nrows, int ncols);
        matrix(int nrows, int ncols, double val);
        matrix(const matrix& other);
        ~matrix();
        // display
        void display() const;
        void _display(int len) const;
        //
        void fill(double val);
        void fill_iota();
        void copy(const double* vals);
        // scaler operations
        void add(double val);
        void add(const double* vals, double alpha);
        void scal(double val);
        void scal(const double* vals);
        void divide(const double* vals);
        // property
        void sort();
        void reverse();
        double min() const;
        double max() const;
        double sum() const;
};