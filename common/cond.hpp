#ifndef _COND_HPP_
#define _COND_HPP_

#include "marray.hpp"

class triangular_condition_number_estimator
{
    protected:
        MArray::marray<double, 1> x_, y_;
        double smax_ = 1, smin_ = 1; 

    public:
        triangular_condition_number_estimator(int n)                                                                       
        : x_{n}, y_{n} {}

        double update(const cview<1>& row);
};




#endif

