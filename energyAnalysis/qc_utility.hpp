/********************************************
 * qc_utility.hpp
 *
 * Header file for quantum chemistry utility
 * functions used in energyAnalysis.cxx
 *
 * Functions in this file:
 *     weighted_evd
 *     c_to_x
 *     make_c
 *     make_E2c
 *     make_E2x
 *     make_E4c
 *     make_E4x
 *     make_E8c
 *     make_E8x
********************************************/
#ifndef _ENERGYANALYSIS_QC_UTILITY_HPP_
#define _ENERGYANALYSIS_QC_UTILITY_HPP_

#include "energyAnalysis.hpp"
#include "marray.hpp"

#include <tuple>

/********************************************
 * weighted_evd 
 *
 * This function generates the transpose of 
 *   the eigenvectors of the input matrix,
 *   which then in turn are weighted by the
 *   absolute values of the eigenvalues. 
 *
 *   The eigenvalues and the transpose of the 
 *   weighted eigenvectors are returned as a
 *   tuple 
 *
 ********************************************/
auto weighted_evd(const view<2>& target,
                  double& Snorm){

    const auto n = std::min(target.length(0), target.length(1));
    tensor<2> u{{target.length(0), target.length(1)}, COLUMN_MAJOR};
    tensor<1> s{n};

    decompose(target, u, s, 0);
    // for (auto i : range(5))
    //     printf("U[%d][%d] = %e\n", i, i, u[i][i]);

    /*
    // The following is a remenent from some testing that I will
    //    leave in place just in case we need it
    printf("reading eigenvectors from file\n");
    FILE* fptr = fopen("T2vecs.dat","r+b");
    if (fptr == NULL) {
        printf("WRITING OVER T2vec.dat\n");
//        fclose(fptr);
        fptr = fopen("T2vecs.dat","wb");
        fwrite(U.data(),sizeof(double),U.size(),fptr);
    } else {
        printf("reading from T2vec.dat\n");
        fread(&U[0][0],sizeof(double),U.size(),fptr);
    }
    fclose(fptr);

//    printf("After reading from file...\n"); jht_print(U);
    */

    Snorm = 0;
    #pragma omp simd
    for (auto i = 0; i < s.length(0); i++) {
        Snorm += s[i]*s[i];
    }

    #pragma omp parallel for 
    for (auto j = 0; j < u.length(1); j++) {
        const double tmp = fabs(s[j]);
        #pragma omp simd
        for (auto i = 0; i < u.length(0); i++) {
            u[i][j] *= tmp;
        }
    }

//    u = u.T(); 
    tensor<2> uT = u.T();
/*
 * IMPORTANT NOTE. CANNOT pass u.T() here, as that is a view, 
 *   one must make a "proper" temporary or transpose u itself, first...
 */
    return std::make_tuple(uT, s);
}

/********************************************
 * aibj_to_ajbi 
 *
 * Given a "colomb"-like ordering (a,i,b,j), 
 * copy it over to the exchange like order
 * (a,j,b,i)
********************************************/
tensor<4> aibj_to_ajbi(const view<4>& A)
{
    auto& sizes = A.lengths();
    const auto nv = sizes[0];
    const auto no = sizes[1];

    tensor<4> X{nv, no, nv, no};

    #pragma omp parallel for collapse(3)
    for (auto j : range(no))
    for (auto b : range(nv))
    for (auto i : range(no))
    {   
        auto Xptr = &X[0][j][b][i];
        auto Aptr = &A[0][i][b][j];
        #pragma omp simd
        for (auto a = 0; a < nv; a++) 
        {
            Xptr[a] = Aptr[a]; 
        }
    } 
    
    return X;
}

/********************************************
 * make_c
 *
 * Creates the T2 matrix in the form aibj
********************************************/
void make_c(const view<4>& v4,
            const view<1>& fa, 
            const view<1>& fi, 
            const view<4>& c2)
{
    auto& sizes = v4.lengths();
    const auto nv = sizes[0];
    const auto no = sizes[1];

    #pragma omp parallel for collapse(3)
    for (auto j = 0; j < no; j++)
    for (auto b = 0; b < nv; b++)
    for (auto i = 0; i < no; i++)
    {
        auto fijb = fi[i] + fi[j] - fa[b];
        auto c    = &c2[0][i][b][j];
        auto v    = &v4[0][i][b][j];
        auto f    = &fa[0];

        #pragma omp simd
        for (int a = 0; a < nv; a++)
            c[a] = v[a] / (fijb - f[a]);
    }
}

void mat_check(const view<2>& mat1,
               const view<2>& mat2,
               double& threshold)
{
    for (auto i : range(mat1.length(0)))
    for (auto j : range(mat1.length(1)))
    {
        if (mat1[i][j] - mat2[i][j] > threshold || mat1[i][j] - mat2[i][j] < -1.0 * threshold)
            printf("i, j, mat1, mat2 = %ld, %ld, %e, %e\n", i, j, mat1[i][j], mat2[i][j]);
    }
}

void sym_check(const view<2>& mat,
               double& threshold)
{
    for (auto i : range(mat.length(0)))
    for (auto j : range(i))
    {
        if (mat[i][j] - mat[j][i] > threshold || mat[i][j] - mat[j][i] < -1.0 *  threshold)
        {
            printf("This matrix is not symmetric!\n");
            printf("i, j = %ld, %ld, %e, %e\n", i, j, mat[i][j], mat[j][i]);
            goto stop;
        }
    }
    stop:
    printf("Symmetry cheking is done!\n");
}

#endif
