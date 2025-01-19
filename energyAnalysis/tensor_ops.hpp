/*****************************************************
 * tensor_ops.hpp
 *
 * Header file that supports tensor operations in 
 * energyAnalsyis.cxx
 *
 * Functions in this file:
 *     L2_normalize
 *     sum_col_squares 
 *     qrdecomp
 *     to_diagonal
 *     jht_fnorm
 *     jht_print
 *     pivot_ene
*****************************************************/
#ifndef _ENERGYANALYSIS_TENSOR_OPS_H_ 
#define _ENERGYANALYSIS_TENSOR_OPS_H_ 

#include "energyAnalysis.hpp"
#include "marray.hpp"

#include <cmath>

/*****************************************************
 * L2_normalize
 *
 * L2 normalizes the columns of a matrix
*****************************************************/
void L2_normalize(tensor<2>& a) {

    const auto nrow = a.length(0);
    const auto ncol = a.length(1);

    #pragma omp for
    for (auto j=0; j < ncol; j++) {

        double l2 = 0;
        double* aptr = &a[0][j];

        #pragma omp simd
        for (auto i=0; i < nrow; i++) {
            l2 += aptr[i]*aptr[i];
        }

        l2 = 1.e0/sqrt(l2);

        #pragma omp simd
        for (auto i=0; i < nrow; i++) {
           aptr[i] *= l2;
        }

    }

}

/*****************************************************
 * sum_col_squares
 *
 * Sums the squared elements of the columns of a 
 * 2d tensor as the elements of a 1D tensor 
*****************************************************/
tensor<1> sum_col_squares(const view<2>& a) {
    const int nrow = a.length(0);
    const int ncol = a.length(1);
    tensor<1> res{ncol};

    #pragma omp parallel for
    for (auto j = 0; j < ncol; j++) {

        const double* aptr = &a[0][j];
        double* rptr = &res[j];

        #pragma omp simd
        for (auto i = 0; i < nrow; i++) {
            *rptr += aptr[i]*aptr[i];
        }
    }

    return res;
}

/*****************************************************
 * qrdecomp
 *
 * performs QR decomposition of a matrix. Returns the
 * Q, orthogonal vectors of said matrix
*****************************************************/
tensor<2> qrdecomp(const view<2>& a) {
    const auto m = a.length(0);
    const auto n = a.length(1);

    tensor<2> q{{m, n},COLUMN_MAJOR};
    tensor<2> tau{{m, m},COLUMN_MAJOR};

    q = a;

    int stat;
    stat = geqrf(m, n, q.data(), q.stride(1), tau.data());
    if (stat != 0) {
        printf("geqrf in qrdecomp exited with bad status %d\n",stat);
        exit(1);
    }

    stat = ungqr(m, n, n, q.data(), q.stride(1), tau.data());
    if (stat != 0) {
        printf("ungqr in qrdecomp exited with bad status %d\n",stat);
        exit(1);
    }

    return q;
}

/*****************************************************
 * to_diagonal
 * 
 * Takes a 2D matrix and returns a vector of the 
 * diagonal elements
*****************************************************/
tensor<1> to_diagonal(const view<2>& a) {
    const auto ni = a.length(0);
    const auto nj = a.length(1);
    const auto nn = std::min(ni,nj);

    tensor<1> data{{nn},COLUMN_MAJOR};

    for (auto j : range(nn)) {
        data[j] = a[j][j];
    }

    return data;
}

/*****************************************************
 * jht_fnorm
 *
 * Returns the Frobenius norm of a vector/matrix
*****************************************************/
double jht_fnorm(const view<2>& a) {
    double Fnorm = 0;
    for (auto j = 0; j < a.length(1); j++) {
        for (auto i = 0; i < a.length(0); i++) {
            Fnorm += a[i][j]*a[i][j];
        }
    }
    return sqrt(Fnorm);
}

double jht_fnorm(const view<2,int>& a) {
    double Fnorm = 0;
    for (auto j = 0; j < a.length(1); j++) {
        for (auto i = 0; i < a.length(0); i++) {
            Fnorm += a[i][j]*a[i][j];
        }
    }
    return sqrt(Fnorm);
}


/*****************************************************
 * jht_print
 *
 * Prints the elements of a vector/matrix
*****************************************************/
void jht_print(const view<2>& data) {
    for (auto row = 0; row < data.length(0); row++) {
        for (auto col = 0; col < data.length(1); col++) {
            printf("%0.15lf  ",data[row][col]);
        }
        printf("\n");
    }
    printf("Its F-norm is %lf \n\n",jht_fnorm(data));
}       

void jht_print(const view<2,int>& data) {
    for (auto row = 0; row < data.length(0); row++) {
        for (auto col = 0; col < data.length(1); col++) {
            printf("%.15d  ",data[row][col]);
        }
        printf("\n");
    }
    printf("Its F-norm is %lf \n\n",jht_fnorm(data));
}       

void jht_print(const view<1>& data) {
    for (auto row = 0; row < data.length(0); row++) {
        printf("%0.15lf  ",data[row]);
        printf("\n");
    }
}

/******************************************************
 * pivot_ene
 *
 * pivot the point with respect to the energy, notably,
 * we need to exclude the selected points 
 *
 ******************************************************/
int pivot_ene(tensor<1>& Eo, std::set<int>& sel_ps, std::vector<int>& pvt)
{
    double maxabsene = 0.0;
    int idx = -1;
    for(int i = 0; i < Eo.length(0); i++)
    {
        if (sel_ps.count(i) == 0 && maxabsene < abs(Eo[i]))
        {
            maxabsene = abs(Eo[i]);
            idx = i;
        }
    }
    sel_ps.insert(idx);
    pvt.push_back(idx);
    return idx;
}

#endif
