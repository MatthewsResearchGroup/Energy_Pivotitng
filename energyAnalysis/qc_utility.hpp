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

double E1(const view<4>& V1, const view<4>& V2, const view<1>& FA, const view<1>& FI) //Vc,Vc gives half coulomb. Vc and Vx gives -exchange
{
    auto sum = 0.0;
    auto& sizes = V1.lengths();
    auto nv = sizes[0];
    auto no = sizes[1];

    #pragma omp parallel for collapse(3), reduction(+:sum)
    for (auto j = 0;j < no;j++)
    for (auto b = 0;b < nv;b++)
    for (auto i = 0;i < no;i++)
    {
        auto fijb = FI[i] + FI[j] - FA[b];
        auto v1 = &V1[0][i][b][j];
        auto v2 = &V2[0][i][b][j];
        auto fa = &FA[0];

        #pragma omp simd
        for (int a = 0;a < nv;a++)
            sum += v1[a] * (v2[a]) / (fijb - fa[a]);
    }

    return sum;
}

double E1(const view<4>& V1, const view<4>& V2, const view<1>& FA, const view<1>& FI, const view<4>& C) // E calculation for MP3, read the second order amplitude from file and calculate the first order using vaibj/ (fi + fj - fa -fb)
{
    auto sum = 0.0;
    auto& sizes = V1.lengths();
    auto nv = sizes[0];
    auto no = sizes[1];

    #pragma omp parallel for collapse(3), reduction(+:sum)
    for (auto j = 0;j < no;j++)
    for (auto b = 0;b < nv;b++)
    for (auto i = 0;i < no;i++)
    {
        auto fijb = FI[i] + FI[j] - FA[b];
        auto v1 = &V1[0][i][b][j];
        auto v2 = &V2[0][i][b][j];
        auto fa = &FA[0];
        auto c = &C[0][i][b][j];

        #pragma omp simd
        for (int a = 0;a < nv;a++)
            sum += v2[a] * ((v1[a]) / (fijb - fa[a]) + c[a]) ;
    }

    return sum;
}


double E1(const view<4>& V, const view<4>& C)
{
    auto sum = 0.0;
    auto& sizes = V.lengths();
    auto nv = sizes[0];
    auto no = sizes[1];

    #pragma omp parallel for collapse(3), reduction(+:sum)
    for (auto j = 0;j < no;j++)
    for (auto b = 0;b < nv;b++)
    for (auto i = 0;i < no;i++)
    {
        auto c = &C[0][i][b][j];
        auto v = &V[0][i][b][j];

        #pragma omp simd
        for (int a = 0;a < nv;a++)
            sum += c[a] * v[a];
    }

    return sum;
}


std::vector<double> E1_no_grid(const view<4>& V, const view<1>& FA, const view<1>& FI)
{
    auto sumC = 0.0;
    auto sumE = 0.0;
    auto& sizes = V.lengths();
    auto nv = sizes[0];
    auto no = sizes[1];

    //#pragma omp parallel for collapse(3), reduction(+:sum)
    for (auto j = 0;j < no;j++)
    for (auto b = 0;b < nv;b++)
    for (auto i = 0;i < no;i++)
    {
        auto fijb = FI[i] + FI[j] - FA[b];
        auto v1 = &V[0][i][b][j];
        auto v2 = &V[0][j][b][i];
        auto fa = &FA[0];

       // #pragma omp simd
        for (int a = 0;a < nv;a++){
            sumC += v1[a];
            sumE += (2*v1[a] - v2[a]);
        }
    }
    std::vector<double> energies;//{sumC,sumE};
    energies.push_back(sumC);
    energies.push_back(sumE);
    return energies;
}

double E2(const view<4>& V, const view<4>& C)
{
    auto sum = 0.0;
    auto& sizes = V.lengths();
    auto nv = sizes[0];
    auto no = sizes[1];

    #pragma omp parallel for collapse(3), reduction(+:sum)
    for (auto j = 0;j < no;j++)
    for (auto b = 0;b < nv;b++)
    for (auto i = 0;i < no;i++)
    {
        auto c = &C[0][i][b][j];
        auto v1 = &V[0][i][b][j];
        auto v2 = &V[0][j][b][i];

        #pragma omp simd
        for (int a = 0;a < nv;a++)
            sum += c[a] * (2*v1[a] - v2[a]);
    }

    return sum;
}

#endif
