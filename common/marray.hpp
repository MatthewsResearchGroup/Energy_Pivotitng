#ifndef _MARRAY_HPP_
#define _MARRAY_HPP_

/**
 * @file
 */

#include "aligned_allocator.hpp"
#include "lapack.h"
#include "timer.h"
#include <iostream>

#include <algorithm>
#include <cmath>

#if defined(DEBUG) && !defined(MARRAY_ENABLE_ASSERTS)
#define MARRAY_ENABLE_ASSERTS
#endif

#define MARRAY_DEFAULT_LAYOUT COLUMN_MAJOR
#include "../marray/include/marray.hpp"
#include "../marray/include/varray.hpp"

#define ERROR(...) do { fprintf(stderr, __VA_ARGS__); abort(); } while (0)

using MArray::slice::all;
using MArray::slice::bcast;
using MArray::range;
using MArray::rangeN;
using MArray::reversed_range;
using MArray::reversed_rangeN;
using MArray::uninitialized;
using MArray::exp;
using MArray::sqrt;
using MArray::ROW_MAJOR;
using MArray::COLUMN_MAJOR;
using MArray::fix;
using MArray::vary;

/**
 * Template alias for a [marray](@ref MArray::marray) of type `double`.
 */
template <int N, typename T=double>
using tensor = MArray::marray<T, N, aligned_allocator<T, 64>>;

/**
 * Template alias for a [marray_view](@ref MArray::marray_view) of type `double`.
 */
template <int N, typename T=double>
using view = MArray::marray_view<T, N>;

/**
 * Template alias for a [marray_view](@ref MArray::marray_view) of type `const double`.
 */
template <int N, typename T=double>
using cview = MArray::marray_view<const T, N>;

/**
 * Return a matrified view of a tensor.
 *
 * Returns a non-modifiable matrix view of the provided
 * tensor view, that groups the first `coldims` dimensions together as the columns,
 * and the remaining dimensions together as the rows.
 *
 * @tparam Tensor   The type of the tensor, should be a tensor, view, or partially-indexed tensor.
 *
 * @param T         The tensor view to matrify. The tensor elements must be contiguous
 *                  along the new matrix row and column dimensions.
 *
 * @param coldims   The number of dimensions to group as the matrix columns.
 *
 * @return          A possibly-immutable matrix view of the matrified tensor.
 */
template <typename Tensor>
auto matrify(Tensor&& T, int coldims)
{
    return T.view().template lowered<2>({coldims});
}

/**
 * Return a row-weighted matrix.
 *
 * Returns a copy of the given matrix `A`, with each row multiplied by the
 * corresponding element of the vector `D`.
 *
 * @param D     The weight vector. Must have the same number of elements as
 *              `A` has rows.
 *
 * @param A     The matrix to weight.
 *
 * @return      A row-weighted copy of `A`.
 */
inline tensor<2> weight(const cview<1>& D, const cview<2>& A)
{
    assert(D.length() == A.length(0));

    tensor<2> DA = A;

    for (auto j : range(DA.length(1)))
    for (auto i : range(DA.length(0)))
        DA[i][j] *= D[i];

    do_flops(DA.length(0)*DA.length(1));

    return DA;
}

/**
 * Return a column-weighted matrix.
 *
 * Returns a copy of the given matrix `A`, with each column multiplied by the
 * corresponding element of the vector `D`.
 *
 * @param A     The matrix to weight.
 *
 * @param D     The weight vector. Must have the same number of elements as
 *              `A` has columns.
 *
 * @return      A column-weighted copy of `A`.
 */
inline tensor<2> weight(const cview<2>& A, const cview<1>& D)
{
    assert(D.length() == A.length(1));

    tensor<2> AD = A;

    for (auto j : range(AD.length(1)))
    for (auto i : range(AD.length(0)))
        AD[i][j] *= D[j];

    do_flops(AD.length(0)*AD.length(1));

    return AD;
}

/**
 * Return a row- and column-weighted matrix.
 *
 * Returns a copy of the given matrix `A`, with each row multiplied by the
 * corresponding element of the vector `D` and each column multiplied by the
 * corresponding element of the vector `E`.
 *
 * @param D     The row weight vector. Must have the same number of elements as
 *              `A` has rows.
 *
 * @param A     The matrix to weight.
 *
 * @param E     The column weight vector. Must have the same number of elements as
 *              `A` has columns.
 *
 * @return      A row- and column-weighted copy of `A`.
 */
inline tensor<2> weight(const cview<1>& D, const cview<2>& A, const cview<1>& E)
{
    assert(D.length() == A.length(0));
    assert(E.length() == A.length(1));

    tensor<2> DAE = A;

    for (auto j : range(DAE.length(1)))
    for (auto i : range(DAE.length(0)))
        DAE[i][j] *= D[i]*E[j];

    do_flops(2*DAE.length(0)*DAE.length(1));

    return DAE;
}

/**
 * Perform the matrix multiplication \f$ C = \alpha A B + \beta C \f$.
 *
 * @param alpha Scalar factor for the product `AB`.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param beta  Scalar factor for the original matrix `C`.
 *
 * @param C     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 */
inline void gemm(double alpha, cview<2> A, cview<2> B,
                  double beta, view<2> C)
{
    auto m = C.length(0);
    auto n = C.length(1);
    auto k = A.length(1);

    assert(A.length(0) == m);
    assert(B.length(1) == n);
    assert(B.length(0) == k);

    if (m == 0 || n == 0) return;

    if (k == 0)
    {
        if (beta != 1.0) C *= beta;
        return;
    }

    if (C.stride(0) > 1)
    {
        std::swap(m,n);
        A.transpose();
        B.transpose();
        C.transpose();
        A.swap(B);
    }

    char transa = A.stride(0) > 1 || (A.stride(1) == 1 && m > 1) ? 'T' : 'N';
    char transb = B.stride(0) > 1 || (B.stride(1) == 1 && k > 1) ? 'T' : 'N';

    if (transa == 'T') A.transpose();
    if (transb == 'T') B.transpose();

    assert(A.stride(0) == 1);
    assert(B.stride(0) == 1);
    assert(C.stride(0) == 1);

    gemm(transa, transb, m, n, k,
         alpha, A.data(), A.stride(1),
                B.data(), B.stride(1),
          beta, C.data(), C.stride(1));

    do_flops(2*m*n*k);
}

/**
 * Perform the matrix multiplication \f$ C = A B \f$.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 */
inline void gemm(const cview<2>& A, const cview<2>& B, const view<2>& C)
{
    gemm(1, A, B, 0, C);
}

/**
 * Return the result of the matrix multiplication \f$ C = \alpha A B \f$.
 *
 * @param alpha Scalar factor for the product `AB`.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A `m`x`n` matrix holding the scaled product.
 */
inline tensor<2> gemm(double alpha, const cview<2>& A, const cview<2>& B)
{
    tensor<2> C({A.length(0), B.length(1)}, uninitialized);
    gemm(alpha, A, B, 0, C);
    return C;
}

/**
 * Return the result of the matrix multiplication \f$ C = A B \f$.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A `m`x`n` matrix holding the product.
 */
inline tensor<2> gemm(const cview<2>& A, const cview<2>& B)
{
    return gemm(1, A, B);
}

/**
 * Perform the triple matrix product \f$ D = \alpha ABC + \beta D \f$.
 *
 * @param alpha Scalar factor for the product `ABC`.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param beta  Scalar factor for the original matrix `D`.
 *
 * @param D     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 */
inline void gemm3(double alpha, cview<2> A, cview<2> B, cview<2> C,
                  double beta, view<2> D)
{
    auto m = A.length(0);
    auto k = B.length(0);
    auto l = C.length(0);
    auto n = C.length(1);

    if (m*k*l + m*l*n < k*l*n + m*k*n)
        gemm(alpha, gemm(A, B), C, beta, D);
    else
        gemm(alpha, A, gemm(B, C), beta, D);
}

/**
 * Perform the triple matrix product \f$ D = ABC \f$.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param D     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 */
inline void gemm3(cview<2> A, cview<2> B, cview<2> C, view<2> D)
{
    gemm3(1, A, B, C, 0, D);
}

/**
 * Return the result of the triple matrix product \f$ D = \alpha ABC \f$.
 *
 * @param alpha Scalar factor for the product `ABC`.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A `m`x`n` matrix holding the scaled product.
 */
inline tensor<2> gemm3(double alpha, cview<2> A, cview<2> B, cview<2> C)
{
    tensor<2> D({A.length(0), C.length(1)}, uninitialized);
    gemm3(alpha, A, B, C, 0, D);
    return D;
}

/**
 * Return the result of the triple matrix product \f$ D = ABC \f$.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A `m`x`n` matrix holding the product.
 */
inline tensor<2> gemm3(cview<2> A, cview<2> B, cview<2> C)
{
    return gemm3(1, A, B, C);
}

/**
 * Perform the matrix-vector multiplication \f$ y = \alpha Ax + \beta y \f$.
 *
 * The "left-side" multiplication \f$ y = \alpha xA + \beta y \f$ can be performed using
 * `gemv(alpha, A.T(), x, beta, y)`.
 *
 * @param alpha Scalar factor for the product `Ax`.
 *
 * @param A     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param x     A vector or vector view of length `n`.
 *
 * @param beta  Scalar factor for the original vector `y`.
 *
 * @param y     A vector or vector view of length `m`.
 */
inline void gemv(double alpha, cview<2> A, const cview<1>& x,
                  double beta, const view<1>& y)
{
    auto m = A.length(0);
    auto n = A.length(1);

    assert(y.length() == m);
    assert(x.length() == n);

    char transa = A.stride(0) > 1 ? 'T' : 'N';

    if (transa == 'T') A.transpose();

    assert(A.stride(0) == 1);

    gemv(transa, A.length(0), A.length(1),
         alpha, A.data(), A.stride(1),
                x.data(), x.stride(),
          beta, y.data(), y.stride());

    do_flops(2*m*n);
}

/**
 * Perform the matrix-vector multiplication \f$ y = Ax \f$.
 *
 * The "left-side" multiplication \f$ y = xA \f$ can be performed using
 * `gemv(A.T(), x, y)`.
 *
 * @param A     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param x     A vector or vector view of length `n`.
 *
 * @param y     A vector or vector view of length `m`.
 */
inline void gemv(const cview<2>& A, const cview<1>& x, const view<1>& y)
{
    gemv(1, A, x, 0, y);
}

/**
 * Return the result of the matrix-vector multiplication \f$ y = \alpha Ax \f$.
 *
 * The "left-side" multiplication \f$ y = \alpha xA \f$ can be performed using
 * `gemv(alpha, A.T(), x)`.
 *
 * @param alpha Scalar factor for the product `Ax`.
 *
 * @param A     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param x     A vector or vector view of length `n`.
 *
 * @return      A vector of length `m` holding the scaled product.
 */
inline tensor<1> gemv(double alpha, const cview<2>& A, const cview<1>& x)
{
    tensor<1> y({A.length(0)}, uninitialized);
    gemv(alpha, A, x, 0, y);
    return y;
}

/**
 * Return the result of the matrix-vector multiplication \f$ y = Ax \f$.
 *
 * The "left-side" multiplication \f$ y = xA \f$ can be performed using
 * `gemv(A.T(), x)`.
 *
 * @param A     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param x     A vector or vector view of length `n`.
 *
 * @return      A vector of length `m` holding the product.
 */
inline tensor<1> gemv(const cview<2>& A, const cview<1>& x)
{
    return gemv(1, A, x);
}

/**
 * Perform the outer product \f$ A = \alpha xy^\text{T} + \beta A \f$.
 *
 * @param alpha Scalar factor for the product \f$ xy^\text{T} \f$.
 *
 * @param x     A vector or vector view of length `m`.
 *
 * @param y     A vector or vector view of length `n`.
 *
 * @param beta  Scalar factor for the original matrix `A`.
 *
 * @param A     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 */
inline void ger(double alpha, cview<1> x, cview<1> y,
                double beta, view<2> A)
{
    auto m = A.length(0);
    auto n = A.length(1);

    assert(x.length() == m);
    assert(y.length() == n);

    if (A.stride(0) > 1)
    {
        A.transpose();
        x.swap(y);
    }

    assert(A.stride(0) == 1);

    if (beta == 0.0) A = 0;
    else if (beta != 1.0) A *= beta;

    ger(A.length(0), A.length(1),
        alpha, x.data(), x.stride(),
               y.data(), y.stride(),
               A.data(), A.stride(1));

    do_flops(2*m*n);
}

/**
 * Perform the outer product \f$ A = xy^\text{T} \f$.
 *
 * @param x     A vector or vector view of length `m`.
 *
 * @param y     A vector or vector view of length `n`.
 *
 * @param A     A `m`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 */
inline void ger(const cview<1>& x, const cview<1>& y, const view<2>& A)
{
    ger(1, x, y, 0, A);
}

/**
 * Return the result of the outer product \f$ A = \alpha xy^\text{T} \f$.
 *
 * @param alpha Scalar factor for the product \f$ xy^\text{T} \f$.
 *
 * @param x     A vector or vector view of length `m`.
 *
 * @param y     A vector or vector view of length `n`.
 *
 * @return      A `m`x`n` matrix holding the scaled product.
 */
inline tensor<2> ger(double alpha, const cview<1>& x, const cview<1>& y)
{
    tensor<2> A({x.length(), y.length()});
    ger(alpha, x, y, 1, A);
    return A;
}

/**
 * Return the result of the outer product \f$ A = xy^\text{T} \f$.
 *
 * @param x     A vector or vector view of length `m`.
 *
 * @param y     A vector or vector view of length `n`.
 *
 * @return      A `m`x`n` matrix holding the product.
 */
inline tensor<2> ger(const cview<1>& x, const cview<1>& y)
{
    return ger(1, x, y);
}

/**
 * Form the diagonal elements of the matrix multiplication \f$ C = \alpha AB + \beta C \f$.
 *
 * @param alpha Scalar factor for the product `AB`.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`m` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param beta  Scalar factor for the original vector `C`.
 *
 * @param C     A vector or vector view of length `m`.
 */
inline void gemmdiag(double alpha, const cview<2>& A, const cview<2>& B,
                     double beta, const view<1>& C)
{
    auto m = A.length(0);
    auto n = A.length(1);

    assert(B.length(0) == n);
    assert(B.length(1) == m);
    assert(C.length(0) == m);

    if (A.stride(0) == 1 && B.stride(1) == 1 && C.stride(0) == 1)
    {
        double* __restrict c = &C[0];

        if (beta == 0)
        {
            for (int i = 0;i < m;i++)
                c[i] = 0;
        }
        else if (beta != 1.0)
        {
            for (int i = 0;i < m;i++)
                c[i] *= beta;
        }

        for (int j = 0;j < n;j++)
        {
            const double* __restrict a = &A[0][j];
            const double* __restrict b = &B[j][0];

            for (int i = 0;i < m;i++)
                c[i] += alpha*a[i]*b[i];
        }
    }
    else if (A.stride(1) == 1 && B.stride(0) == 1)
    {
        for (int i = 0;i < m;i++)
        {
            const double* __restrict a = &A[i][0];
            const double* __restrict b = &B[0][i];

            double sum = 0;
            for (int j = 0;j < n;j++)
                sum += a[j]*b[j];

            if (beta == 0)
                C[i] = alpha*sum;
            else
                C[i] = alpha*sum + beta*C[i];
        }
    }
    else
    {
        for (int i = 0;i < m;i++)
        {
            double sum = 0;
            for (int j = 0;j < n;j++)
                sum += A[i][j]*B[j][i];

            if (beta == 0)
                C[i] = alpha*sum;
            else
                C[i] = alpha*sum + beta*C[i];
        }
    }

    do_flops(2*m*n);
}

/**
 * Form the diagonal elements of the matrix multiplication \f$ C = AB \f$.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`m` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A vector or vector view of length `m`.
 */
inline void gemmdiag(const cview<2>& A, const cview<2>& B, const view<1>& C)
{
    gemmdiag(1, A, B, 0, C);
}

/**
 * Return the diagonal elements of the matrix multiplication \f$ C = \alpha AB \f$.
 *
 * @param alpha Scalar factor for the product `AB`.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`m` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A vector of length `m` holding the diagonal elements of the scaled product.
 */
inline tensor<1> gemmdiag(double alpha, const cview<2>& A, const cview<2>& B)
{
    tensor<1> C({A.length(0)}, uninitialized);
    gemmdiag(alpha, A, B, 0, C);
    return C;
}

/**
 * Return the diagonal elements of the matrix multiplication \f$ C = AB \f$.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`m` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A vector of length `m` holding the diagonal elements of the product.
 */
inline tensor<1> gemmdiag(const cview<2>& A, const cview<2>& B)
{
    return gemmdiag(1, A, B);
}

/**
 * Form the diagonal elements of the triple matrix product \f$ D = \alpha ABC + \beta D \f$.
 *
 * @param alpha Scalar factor for the product `ABC`.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`m` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param beta  Scalar factor for the original vector `D`.
 *
 * @param D     A vector or vector view of length `m`.
 */
inline void gemm3diag(double alpha, const cview<2>& A, const cview<2>& B, const cview<2>& C,
                      double beta, const view<1>& D)
{
    auto k = B.length(0);
    auto l = B.length(1);

    if (k < l)
        gemmdiag(alpha, gemm(A, B), C, beta, D);
    else
        gemmdiag(alpha, A, gemm(B, C), beta, D);
}

/**
 * Form the diagonal elements of the triple matrix product \f$ D = ABC \f$.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`m` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param D     A vector or vector view of length `m`.
 */
inline void gemm3diag(const cview<2>& A, const cview<2>& B, const cview<2>& C, const view<1>& D)
{
    gemm3diag(1, A, B, C, 0, D);
}

/**
 * Return the diagonal elements of the triple matrix product \f$ D = \alpha ABC \f$.
 *
 * @param alpha Scalar factor for the product `ABC`.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`m` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A vector of length `m` holding the diagonal elements of the scaled product.
 */
inline tensor<1> gemm3diag(double alpha, const cview<2>& A, const cview<2>& B, const cview<2>& C)
{
    tensor<1> D({A.length(0)}, uninitialized);
    gemm3diag(alpha, A, B, C, 0, D);
    return D;
}

/**
 * Return the diagonal elements of the triple matrix product \f$ D = ABC \f$.
 *
 * @param A     A `m`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`m` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A vector of length `m` holding the diagonal elements of the product.
 */
inline tensor<1> gemm3diag(const cview<2>& A, const cview<2>& B, const cview<2>& C)
{
    return gemm3diag(1, A, B, C);
}

/**
 * Perform the multiplication of a Khatri-Rao product of two matrices and a
 * third matrix \f$ D_{pqr} = \sum_s \alpha A_{ps} B_{sq} C_{sr} + \beta D_{pqr} \f$.
 *
 * @param alpha Scalar factor for the KRP-matrix product.
 *
 * @param A     A `m`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `l`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param beta  Scalar factor for the original tensor `D`.
 *
 * @param D     A `m`x`n`x`k` tensor or tensor view. One of the first two
 *              dimensions must have a stride of one.
 */
inline void gemkrp(double alpha, const cview<2>& A, const cview<2>& B, const cview<2>& C,
                   double beta, const view<3>& D)
{
    auto k = D.length(2);
    assert(A.length(0) == D.length(0));
    assert(B.length(1) == D.length(1));
    assert(C.length(1) == D.length(2));
    assert(A.length(1) == B.length(0));
    assert(A.length(1) == C.length(0));

    for (auto p : range(k))
        gemm(alpha, A, weight(C[all][p], B), beta, D[all][all][p]);
}

/**
 * Perform the multiplication of a Khatri-Rao product of two matrices and a
 * third matrix \f$ D_{pqr} = \sum_s A_{ps} B_{sq} C_{sr} \f$.
 *
 * @param A     A `m`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `l`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param D     A `m`x`n`x`k` tensor or tensor view. One of the first two
 *              dimensions must have a stride of one.
 */
inline void gemkrp(const cview<2>& A, const cview<2>& B, const cview<2>& C, const view<3>& D)
{
    gemkrp(1, A, B, C, 0, D);
}

/**
 * Returns the result of multiplying a Khatri-Rao product of two matrices and a
 * third matrix \f$ D_{pqr} = \sum_s \alpha A_{ps} B_{sq} C_{sr} \f$.
 *
 * @param alpha Scalar factor for the KRP-matrix product.
 *
 * @param A     A `m`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `l`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A `m`x`n`x`k` tensor holding the scaled KRP-matrix product.
 */
inline tensor<3> gemkrp(double alpha, const cview<2>& A, const cview<2>& B, const cview<2>& C)
{
    tensor<3> D({A.length(0), B.length(1), C.length(1)}, uninitialized);
    gemkrp(alpha, A, B, C, 0, D);
    return D;
}

/**
 * Returns the result of multiplying a Khatri-Rao product of two matrices and a
 * third matrix \f$ D_{pqr} = \sum_s A_{ps} B_{sq} C_{sr} \f$.
 *
 * @param A     A `m`x`l` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `l`x`n` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param C     A `l`x`k` matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @return      A `m`x`n`x`k` tensor holding the KRP-matrix product.
 */
inline tensor<3> gemkrp(const cview<2>& A, const cview<2>& B, const cview<2>& C)
{
    return gemkrp(1, A, B, C);
}

/**
 * Perform a Khatri-Rao product of two matrices \f$ C_{pqr} = \alpha A_{pq} B_{pr} + \beta C_{pqr} \f$.
 *
 * @param alpha Scalar factor for the Khatri-Rao product.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`k` matrix or matrix view.
 *
 * @param beta  Scalar factor for the original tensor `C`.
 *
 * @param C     A `m`x`n`x`k` tensor or tensor view.
 */
inline void krp(double alpha, const cview<2>& A, const cview<2>& B,
                       double beta, const view<3>& C)
{
    auto m = C.length(0);
    auto n = C.length(1);
    auto k = C.length(2);
    assert(A.length(0) == C.length(0));
    assert(A.length(1) == C.length(1));
    assert(B.length(0) == C.length(0));
    assert(B.length(1) == C.length(2));

    if (beta == 0)
    {
        if (A.stride(0) == 1 && B.stride(0) == 1 && C.stride(0) == 1)
        {
            for (int p = 0;p < k;p++)
            for (int j = 0;j < n;j++)
            {
                const double* __restrict a = &A[0][j];
                const double* __restrict b = &B[0][p];
                      double* __restrict c = &C[0][j][p];

                for (int i = 0;i < m;i++)
                    c[i] = alpha*a[i]*b[i];
            }
        }
        else
        {
            for (int p = 0;p < k;p++)
            for (int j = 0;j < n;j++)
            for (int i = 0;i < m;i++)
                C[i][j][p] = alpha*A[i][j]*B[i][p];
        }
    }
    else
    {
        if (A.stride(0) == 1 && B.stride(0) == 1 && C.stride(0) == 1)
        {
            for (int p = 0;p < k;p++)
            for (int j = 0;j < n;j++)
            {
                const double* __restrict a = &A[0][j];
                const double* __restrict b = &B[0][p];
                      double* __restrict c = &C[0][j][p];

                for (int i = 0;i < m;i++)
                    c[i] = alpha*a[i]*b[i] + beta*c[i];
            }
        }
        else
        {
            for (int p = 0;p < k;p++)
            for (int j = 0;j < n;j++)
            for (int i = 0;i < m;i++)
                C[i][j][p] = alpha*A[i][j]*B[i][p] + beta*C[i][j][p];
        }
    }

    do_flops(4*m*n*k);
}

/**
 * Perform a Khatri-Rao product of two matrices \f$ C_{pqr} = A_{pq} B_{pr} \f$.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`k` matrix or matrix view.
 *
 * @param C     A `m`x`n`x`k` tensor or tensor view.
 */
inline void krp(const cview<2>& A, const cview<2>& B, const view<3>& C)
{
    krp(1, A, B, 0, C);
}

/**
 * Return the Khatri-Rao product of two matrices \f$ C_{pqr} = \alpha A_{pq} B_{pr} \f$.
 *
 * @param alpha Scalar factor for the Khatri-Rao product.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`k` matrix or matrix view.
 *
 * @return      A `m`x`n`x`k` tensor holding the scaled product.
 */
inline tensor<3> krp(double alpha, const cview<2>& A, const cview<2>& B)
{
    tensor<3> C({A.length(0), A.length(1), B.length(1)}, uninitialized);
    krp(alpha, A, B, 0, C);
    return C;
}

/**
 * Return the Khatri-Rao product of two matrices \f$ C_{pqr} = A_{pq} B_{pr} \f$.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`k` matrix or matrix view.
 *
 * @return      A `m`x`n`x`k` tensor holding the product.
 */
inline tensor<3> krp(const cview<2>& A, const cview<2>& B)
{
    return krp(1, A, B);
}

/**
 * Perform the element-wise multiplication \f$ C = \alpha A \ast B + \beta C \f$.
 *
 * @param alpha Scalar factor for the product `A*B`.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @param beta  Scalar factor for the original matrix `C`.
 *
 * @param C     A `m`x`n` matrix or matrix view.
 */
inline void cwise_prod(double alpha, const cview<2>& A, const cview<2>& B,
                       double beta, const view<2>& C)
{
    auto m = A.length(0);
    auto n = A.length(1);
    assert(B.length(0) == m);
    assert(B.length(1) == n);
    assert(C.length(0) == m);
    assert(C.length(1) == n);

    if (A.stride(0) == 1 && B.stride(0) == 1 && C.stride(0) == 1)
    {
        #pragma omp parallel for schedule(static)
        for (int q = 0;q < n;q++)
        {
            const double* __restrict a = &A[0][q];
            const double* __restrict b = &B[0][q];
                  double* __restrict c = &C[0][q];

            if (beta == 0.0)
            {
                for (int p = 0;p < m;p++)
                    c[p] = alpha*a[p]*b[p];
            }
            else
            {
                for (int p = 0;p < m;p++)
                    c[p] = alpha*a[p]*b[p] + beta*c[p];
            }
        }
    }
    else
    {
        if (beta == 0.0) C = alpha*A*B;
        else C = alpha*A*B + beta*C;
    }

    do_flops(4*m*n);
}

/**
 * Perform the element-wise multiplication \f$ C = A \ast B \f$.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @param C     A `m`x`n` matrix or matrix view.
 */
inline void cwise_prod(const cview<2>& A, const cview<2>& B, const view<2>& C)
{
    cwise_prod(1, A, B, 0, C);
}

/**
 * Return the result of the element-wise multiplication \f$ C = \alpha A \ast B \f$.
 *
 * @param alpha Scalar factor for the product `A*B`.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @return      A `m`x`n` matrix holding the scaled product.
 */
inline tensor<2> cwise_prod(double alpha, const cview<2>& A, const cview<2>& B)
{
    tensor<2> C(A.lengths(), uninitialized);
    cwise_prod(alpha, A, B, 0, C);
    return C;
}

/**
 * Return the result of the element-wise multiplication \f$ C = A \ast B \f$.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @return      A `m`x`n` matrix holding the product.
 */
inline tensor<2> cwise_prod(const cview<2>& A, const cview<2>& B)
{
    return cwise_prod(1, A, B);
}

/**
 * Perform the element-wise multiplication \f$ D = \alpha A \ast B \ast C + \beta D \f$.
 *
 * @param alpha Scalar factor for the product `A*B*C`.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @param C     A `m`x`n` matrix or matrix view.
 *
 * @param beta  Scalar factor for the original matrix `D`.
 *
 * @param D     A `m`x`n` matrix or matrix view.
 */
inline void cwise_prod3(double alpha, const cview<2>& A, const cview<2>& B, const cview<2>& C,
                        double beta, const view<2>& D)
{
    auto m = A.length(0);
    auto n = A.length(1);
    assert(B.length(0) == m);
    assert(B.length(1) == n);
    assert(C.length(0) == m);
    assert(C.length(1) == n);
    assert(D.length(0) == m);
    assert(D.length(1) == n);

    if (A.stride(0) == 1 && B.stride(0) == 1 && C.stride(0) == 1 && D.stride(0) == 1)
    {
        #pragma omp parallel for schedule(static)
        for (int q = 0;q < n;q++)
        {
            const double* __restrict a = &A[0][q];
            const double* __restrict b = &B[0][q];
            const double* __restrict c = &C[0][q];
                  double* __restrict d = &D[0][q];

            if (beta == 0.0)
            {
                for (int p = 0;p < m;p++)
                    d[p] = alpha*a[p]*b[p]*c[p];
            }
            else
            {
                for (int p = 0;p < m;p++)
                    d[p] = alpha*a[p]*b[p]*c[p] + beta*d[p];
            }
        }
    }
    else
    {
        if (beta == 0.0) D = alpha*A*B*C;
        else D = alpha*A*B*C + beta*D;
    }

    do_flops(5*m*n);
}

/**
 * Perform the element-wise multiplication \f$ D = A \ast B \ast C \f$.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @param C     A `m`x`n` matrix or matrix view.
 *
 * @param D     A `m`x`n` matrix or matrix view.
 */
inline void cwise_prod3(const cview<2>& A, const cview<2>& B, const cview<2>& C, const view<2>& D)
{
    cwise_prod3(1, A, B, C, 0, D);
}

/**
 * Return the result of the element-wise multiplication \f$ D = \alpha A \ast B \ast C \f$.
 *
 * @param alpha Scalar factor for the product `A*B*C`.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @param C     A `m`x`n` matrix or matrix view.
 *
 * @return      A `m`x`n` matrix holding the scaled product.
 */
inline tensor<2> cwise_prod3(double alpha, const cview<2>& A, const cview<2>& B, const cview<2>& C)
{
    tensor<2> D(A.lengths(), uninitialized);
    cwise_prod3(alpha, A, B, C, 0, D);
    return D;
}

/**
 * Return the result of the element-wise multiplication \f$ D = A \ast B \ast C \f$.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @param C     A `m`x`n` matrix or matrix view.
 *
 * @return      A `m`x`n` matrix holding the product.
 */
inline tensor<2> cwise_prod3(const cview<2>& A, const cview<2>& B, const cview<2> C)
{
    return cwise_prod3(1, A, B, C);
}

/**
 * Return the matrix dot product (inner product) of three matrices \f$ (A \ast B) \cdot C \f$ =
 * \f$ \sum_{pq} A_{pq} B_{pq} C_{pq} \f$.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @param C     A `m`x`n` matrix or matrix view.
 *
 * @return      The dot product `A.B.C`.
 */
inline double cwise_dot(const cview<2>& A, const cview<2>& B, const cview<2>& C)
{
    auto prod = 0.0;
    auto m = A.length(0);
    auto n = A.length(1);
    assert(B.length(0) == m);
    assert(B.length(1) == n);
    assert(C.length(0) == m);
    assert(C.length(1) == n);

    if (A.stride(0) == 1 && B.stride(0) == 1 && C.stride(0) == 1)
    {
        #pragma omp parallel for schedule(static) reduction(+:prod)
        for (int q = 0;q < n;q++)
        {
            const double* __restrict a = &A[0][q];
            const double* __restrict b = &B[0][q];
            const double* __restrict c = &C[0][q];

            for (int p = 0;p < m;p++)
                prod += a[p]*b[p]*c[p];
        }
    }
    else
    {
        #pragma omp parallel for schedule(static) reduction(+:prod)
        for (int q = 0;q < n;q++)
        for (int p = 0;p < m;p++)
            prod += A[p][q]*B[p][q]*C[p][q];
    }

    do_flops(3*m*n);

    return prod;
}

/**
 * Return the matrix dot product (inner product) of two matrices \f$ A \cdot B \f$
 * = \f$ \sum_{pq} A_{pq} B_{pq} \f$.
 *
 * @param A     A `m`x`n` matrix or matrix view.
 *
 * @param B     A `m`x`n` matrix or matrix view.
 *
 * @return      The dot product `A.B`.
 */
inline double cwise_dot(const cview<2>& A, const cview<2>& B)
{
    auto prod = 0.0;
    auto m = A.length(0);
    auto n = A.length(1);
    assert(B.length(0) == m);
    assert(B.length(1) == n);

    if (A.stride(0) == 1 && B.stride(0) == 1)
    {
        #pragma omp parallel for schedule(static) reduction(+:prod)
        for (int q = 0;q < n;q++)
        {
            const double* __restrict a = &A[0][q];
            const double* __restrict b = &B[0][q];

            for (int p = 0;p < m;p++)
                prod += a[p]*b[p];
        }
    }
    else
    {
        #pragma omp parallel for schedule(static) reduction(+:prod)
        for (int q = 0;q < n;q++)
        for (int p = 0;p < m;p++)
            prod += A[p][q]*B[p][q];
    }

    do_flops(2*m*n);

    return prod;
}

inline void fix_pivots(std::vector<integer>& pivot)
{
    int n = pivot.size();

    if (pivot[0] > n)
    {
        int32_t* pivot32 = (int32_t*)pivot.data();

        for (int i = n;i --> 0;)
            pivot[i] = pivot32[i];
    }

    for (int i = 0;i < n;i++)
        pivot[i]--;
}

struct LU : MArray::marray<double,2>
{
    std::vector<integer> ipiv;
    using MArray::marray<double,2>::marray;
};

/**
 * Perform the LU decomposition of a matrix and return a decomposition object that
 * can be used to [solve](@ref solve) a linear system of equations.
 *
 * @param A     A matrix or matrix view, which may be rectangular. The data will
 *              not be modified.
 *
 * @return      A LU decomposition object that can be used to solve systems of equations.
 */
inline LU lu(const cview<2>& A)
{
    auto n = A.length(0);
    assert(A.length(1) == n);

    LU result(A);
    result.ipiv.resize(n);

    if (n == 0) return result;

    auto info = getrf(n, n, result.data(), n, result.ipiv.data());
    //fix_pivots(result.ipiv);

    if (info != 0) ERROR("getrf: info = %d", info);

    return result;
}

/**
 * Solve the system of equations \f$ AX = B \f$ using a precomputed LU decomposition.
 *
 * @param A     A [LU](@ref lu) decomposition of the `m`x`m` matrix `A`.
 *
 * @param B     The `m`x`n` right-hand-side matrix. This matrix overwritten by the solution matrix `X`.
 */
inline void solve(const LU& A, const view<2>& B)
{
    auto n = A.length(0);
    auto nrhs = B.length(1);
    assert(B.length(0) == n);
    assert(B.stride(0) == 1);

    if (n == 0 || nrhs == 0) return;
    getrs('N', n, nrhs,
          A.data(), n, A.ipiv.data(),
          B.data(), B.stride(1));
}

/**
 * Solve the system of equations \f$ XA = B \f$ using a precomputed LU decomposition.
 *
 * @param B     The `m`x`n` right-hand-side matrix. This matrix overwritten by the solution matrix `X`.
 *
 * @param A     A [LU](@ref lu) decomposition of the `n`x`n` matrix `A`.
 */
inline void solve(const view<2>& B, const LU& A)
{
    auto n = A.length(0);
    auto nrhs = B.length(0);
    assert(B.length(1) == n);
    assert(B.stride(0) == 1);

    if (n == 0 || nrhs == 0) return;

#if 0

    trsm('R', 'U', 'N', 'N',
         nrhs, n, 1.0,
         A.data(), n,
         B.data(), B.stride(1));

    trsm('R', 'L', 'N', 'U',
         nrhs, n, 1.0,
         A.data(), n,
         B.data(), B.stride(1));

    for (auto k : range(n))
    {
        auto l = A.ipiv[k];
        if (k == l) continue;

        swap(nrhs, &B[0][k], 1,
                   &B[0][l], 1);
    }

#else

    tensor<2> Bt = B.T();

    getrs('T', n, nrhs,
          A.data(), n, A.ipiv.data(),
          Bt.data(), Bt.stride(1));

    B = Bt.T();

#endif
}

typedef enum {LEFT, RIGHT} side;
typedef enum {UPPER, LOWER} uplo;

/**
 * Solve the system of equations \f$ AX = B \f$ (`side == LEFT`) or \f$ XA = B \f$ (`side == RIGHT`).
 *
 * This function performs the LU decomposition of `A` internally.
 *
 * @param side  Either `LEFT` or `RIGHT`.
 *
 * @param A     A `m`x`m` (`side == LEFT`) or `n`x`n` (`side == RIGHT`) matrix or matrix view.
 *              The data is not modified.
 *
 * @param B     The `m`x`n` right-hand-side matrix. This matrix overwritten by the solution matrix `X`.
 */
inline void solve(side side, const cview<2>& A, const view<2>& B)
{
    //if (side == LEFT) solve(lu(A), B);
    //else              solve(B, lu(A));

    auto m = B.length(0);
    auto n = B.length(1);

    tensor<2> Atmp(A);

    if (side == LEFT)
    {
        assert(A.length(0) == m);
        assert(A.length(1) == m);
        assert(B.stride(0) == 1);

        std::vector<integer> jpvt(m);
        integer rank;
        auto info = gelsy(m, m, n, Atmp.data(), m, B.data(), B.stride(1), jpvt.data(), 1e-10, rank);

        if (info != 0) ERROR("gelsy: info = %d", info);
    }
    else
    {
        tensor<2> Btmp{n,m};

        assert(A.length(0) == n);
        assert(A.length(1) == n);

        Btmp = B.T();

        std::vector<integer> jpvt(n);
        integer rank;
        auto info = gelsy(n, n, m, Atmp.data(), n, Btmp.data(), n, jpvt.data(), 1e-10, rank);

        if (info != 0) ERROR("gelsy: info = %d", info);

        B = Btmp.T();
    }
}

/**
 * Solve the system of equations \f$ Ax = b \f$.
 *
 * This function performs the LU decomposition of `A` internally.
 *
 * @param A     A `n`x`n` matrix or matrix view.
 *              The data is not modified.
 *
 * @param b     The right-hand-side vector of length `n`. This vector overwritten by the solution vector `x`.
 */
inline void solve(const cview<2>& A, const view<1>& b)
{
    solve(LEFT, A, view<2>{{b.length(), 1}, b.data()});
}

/**
 * Solve the system of equations \f$ xA = b \f$.
 *
 * This function performs the LU decomposition of `A` internally.
 *
 * @param b     The right-hand-side vector of length `n`. This vector overwritten by the solution vector `x`.
 *
 * @param A     A `n`x`n` matrix or matrix view.
 *              The data is not modified.
 */
inline void solve(const view<1>& b, const cview<2>& A)
{
    solve(RIGHT, A, view<2>{{1, b.length()}, b.data()});
}

/**
 * Solve the system of equations \f$ AX = B \f$ (`side == LEFT`) or \f$ XA = B \f$ (`side == RIGHT`).
 *
 * This function works with triangular A matrices.
 *
 * @param side  Either `LEFT` or `RIGHT`.
 *
 * @param A     A `m`x`m` (`side == LEFT`) or `n`x`n` (`side == RIGHT`) matrix or matrix view.
 *              The data is not modified.
 *
 * @param B     The `m`x`n` right-hand-side matrix. This matrix overwritten by the solution matrix `X`.
 */
inline void solve_tri(side side, uplo uplo, const cview<2>& A, const view<2>& B)
{
    auto m = B.length(0);
    auto n = B.length(1);

    assert(B.stride(0) == 1);
    assert(A.length(0) == side == LEFT ? m : n);
    assert(A.length(1) == side == LEFT ? m : n);

    auto trans = 'N';
    auto lda = A.stride(1);

    if (A.stride(0) != 1)
    {
        uplo = uplo == UPPER ? LOWER : UPPER;
        trans = 'T';
        lda = A.stride(0);
        assert(A.stride(1) == 1);
    }

    trsm(side == LEFT ? 'L' : 'R', uplo == UPPER ? 'U' : 'L', trans, 'N',
         m, n, 1.0, A.data(), lda, B.data(), B.stride(1));
}

/**
 * Solve the system of equations \f$ APX = PB \f$ (`side == LEFT`) or \f$ XPA = BP \f$ (`side == RIGHT`).
 *
 * This function works with triangular permuted matrices P^-1 A P.
 *
 * @param side  Either `LEFT` or `RIGHT`.
 *
 * @param A     A `m`x`m` (`side == LEFT`) or `n`x`n` (`side == RIGHT`) matrix or matrix view.
 *              The data is not modified.
 *
 * @param B     The `m`x`n` right-hand-side matrix. This matrix overwritten by the solution matrix `X`.
 */
template <typename T>
void solve_tri(side side, uplo uplo, const cview<2>& A, const view<2>& B, const std::vector<T>& P)
{
    auto m = B.length(0);
    auto n = B.length(1);

    assert(B.stride(0) == 1);
    assert(A.length(0) == side == LEFT ? m : n);
    assert(A.length(1) == side == LEFT ? m : n);
    assert(P.size() == side == LEFT ? m : n);

    auto trans = 'N';
    auto lda = A.stride(1);

    if (A.stride(0) != 1)
    {
        uplo = uplo == UPPER ? LOWER : UPPER;
        trans = 'T';
        lda = A.stride(0);
        assert(A.stride(1) == 1);
    }

    tensor<2> BP{m, n};

    if (side == LEFT)
    {
        for (auto j : range(n))
        for (auto i : range(m))
            BP[i][j] = B[P[i]][j];

        trsm('L', uplo == UPPER ? 'U' : 'L', trans, 'N',
             m, n, 1.0, A.data(), lda, BP.data(), BP.stride(1));

        for (auto j : range(n))
        for (auto i : range(m))
            B[P[i]][j] = BP[i][j];
    }
    else
    {
        for (auto j : range(n))
        for (auto i : range(m))
            BP[i][j] = B[i][P[j]];

        trsm('R', uplo == UPPER ? 'U' : 'L', trans, 'N',
             m, n, 1.0, A.data(), lda, BP.data(), BP.stride(1));

        for (auto j : range(n))
        for (auto i : range(m))
            B[i][P[j]] = BP[i][j];
    }
}

/**
 * Return the squared 2-norm of the given tensor.
 *
 * @tparam Tensor   The type of the tensor, should be a tensor, view, or partially-indexed tensor.
 *
 * @param x     A tensor view.
 *
 * @return      The squared 2-norm, which is equal to the sum of squares of the elements.
 */
template <typename Tensor>
double norm2(const Tensor& x)
{
    auto nrm = 0.0;
    x.view().for_each_element([&](double e) { nrm += e*e; });
    return nrm;
}

/**
 * Return the 2-norm of the given tensor.
 *
 * @tparam Tensor   The type of the tensor, should be a tensor, view, or partially-indexed tensor.
 *
 * @param x     A vector or tensor view.
 *
 * @return      The 2-norm, which is equal to the square root of the sum of squares of the elements.
 */
template <typename Tensor>
double norm(const Tensor& x)
{
    // Could suffer from overflow issues...
    return sqrt(norm2(x));
}

/**
 * Return the infinity-norm of the given tensor.
 *
 * @tparam Tensor   The type of the tensor, should be a tensor, view, or partially-indexed tensor.
 *
 * @param x     A vector or tensor view.
 *
 * @return      The infinity-norm, which is equal to the maximum absolute value of the elements.
 */
template <typename Tensor>
double amaxv(const Tensor& x)
{
    auto nrm = 0.0;
    x.view().for_each_element([&](double e) { if (std::abs(e) > nrm) nrm = std::abs(e); });
    return nrm;
}

/**
 * Return the dot product of two vectors.
 *
 * @param x     A vector view.
 *
 * @param y     Another vector view.
 *
 * @return      The dot product, which is equal to the sum of the products of the elements of x and y.
 */
inline double dot(const cview<1>& x, const cview<1>& y)
{
    auto n = x.length();
    assert(y.length() == n);

    auto res = 0.0;
    for (auto i : range(n)) res += x[i]*y[i];
    return res;
}

/**
 * Perform an eigenvalue decomposition (EVD) on the matrix A, and retain only significant
 * eigenvalue/eigenvector pairs.
 *
 * @param A     The matrix or matrix view to decompose. Not modified.
 *
 * @param U     The matrix of eigenvectors, must be a matrix and not a view.
 *
 * @param sigma The vector of eigenvalues, must be a vector and not a view.
 *
 * @param tol   Retain only eigenvalues with absolute value greater or equal to `tol`.
 */
inline void decompose(const cview<2>& A, tensor<2>& U, tensor<1>& sigma, double tol = -1.0)
{
    auto n = A.length(0);
    assert(A.length(1) == n);

    U.reset(A);
    sigma.reset({n}, uninitialized);
    heev('V', 'U', n, U.data(), n, sigma.data());

    auto m = 0;
    for (auto i : range(n))
    {
        if (std::abs(sigma[i]) < tol) continue;
        sigma[m] = sigma[i];
        std::copy_n(&U[0][i], n, &U[0][m]);
        m++;
    }

    sigma.resize({m});
    U.resize({n,m});
}

/**
 * Perform an singular value decomposition (SVD) on the matrix A, and retain only significant
 * singular values/vectors.
 *
 * @param A     The matrix or matrix view to decompose. Not modified.
 *
 * @param U     The matrix of left singular vectors, must be a matrix and not a view.
 *
 * @param V     The matrix of right singular vectors, must be a matrix and not a view.
 *
 * @param sigma The vector of eigenvalues, must be a vector and not a view.
 *
 * @param tol   Retain only eigenvalues with absolute value greater or equal to `tol`.
 */
inline void decompose(const cview<2>& A, tensor<2>& U, tensor<2>& V, tensor<1>& sigma, double tol = -1.0)
{
    auto m = A.length(0);
    auto n = A.length(1);
    auto mn = std::min(m, n);

    tensor<2> tmp(A);
    U.reset({m,mn});
    V.reset({mn,n});

    sigma.reset({mn}, uninitialized);
    gesdd('S', m, n, tmp.data(), tmp.stride(1), sigma.data(), U.data(), U.stride(1), V.data(), V.stride(1));
    tmp[range(mn)] = V;
    V.reset({n,mn});

    auto k = 0;
    for (auto i : range(mn))
    {
        if (std::abs(sigma[i]) < tol) continue;
        sigma[k] = sigma[i];
        U[all][i] = U[all][k];
        V[all][i] = tmp[k][all];
        k++;
    }

    sigma.resize({k});
    U.resize({m,k});
    V.resize({n,k});
}

#endif
