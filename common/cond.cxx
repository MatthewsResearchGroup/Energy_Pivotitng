#include "cond.hpp"


extern "C" void FC_FUNC(dlaic1,DLAIC1)(const integer*, const integer*, const double*, const double*, const double*, const double*, double*, double*, double*);

double triangular_condition_number_estimator::update(const cview<1>& row_)
{
    //auto row = row_.rebased({0});
    auto row = row_;
    auto n = row.length();
    MARRAY_ASSERT(n > 0);

    if (n == 1)
    {
        x_[0] = y_[0] = 1;
        smax_ = smin_ = std::abs(row[0]);
        return 1;
    }

    integer max_val = 1;
    integer min_val = 2;
    integer one = 1;
    auto done = 1.0;
    auto alphax = dot(row[range(n-1)], x_[range(n-1)]);
    auto alphay = dot(row[range(n-1)], y_[range(n-1)]);
    double c, s, smax_new, smin_new;

    FC_FUNC(dlaic1,DLAIC1)(&max_val, &one, &alphax, &smax_, &done, &row[n-1], &smax_new, &c, &s);
    x_[range(n-1)] *= s;
    x_[n-1] = c;

    FC_FUNC(dlaic1,DLAIC1)(&min_val, &one, &alphay, &smin_, &done, &row[n-1], &smin_new, &c, &s);
    y_[range(n-1)] *= s;
    y_[n-1] = c;

    smax_ = smax_new;
    smin_ = smin_new;

    return smax_/smin_;
}
