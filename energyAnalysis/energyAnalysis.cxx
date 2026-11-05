/*
 *energyAnalysis.cxx
 *
 * Energy-pivoted selection of THC grid points (singles and pairs).
 *
 *Stage 1 : Energy pivot overall single grid points (chi1 * norb kept)
 *
 *Stage 3 : draw n_pair random Uniques pairs of stage 1 points
 *
 *Branch a : 5a rank pairs by stand-alone energy, keep n_top
 *           6a energy pivot over stage 1 singles + top pairs (chi2a * norb)
 *
 *Branch b : 5b re-pivot stage 1 singles (chi2b * norb)
 *           6b force 5b singles, rank pairs by conditional energy, keep n_top
 *           7b force 5b singles, energy pivot over top pairs (chi3b * norb)
 *
 *Energy of candidate o: dE = E4 + E8
 *Coulomb weights 4 and 2, exchange weights -2 and -1
 *
 */

#include "energyAnalysis.hpp"
#include "cond.hpp"
#include "input.hpp"
#include "io.hpp"
#include "jobinfo.hpp"
#include "make_distance_pairs.hpp"
#include "marray.hpp"
#include "pair_points.hpp"
#include "qc_utility.hpp"
#include "tensor_ops.hpp"
#include <array>

#include "docopt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tuple>
#include <utility>
#include <vector>

inline void swap_ranges(const view<1>& x, const view<1>& y)
{
    assert(x.length() == y.length());
    swap(x.length(), x.data(), x.stride(), y.data(), y.stride());
}

inline cview<1> diag(const cview<2>& A)
{
    return {{std::min(A.length(0), A.length(1))},
            A.data(),
            {A.stride(0) + A.stride(1)}};
}

inline view<1> diag(const view<2>& A)
{
    return {{std::min(A.length(0), A.length(1))},
            A.data(),
            {A.stride(0) + A.stride(1)}};
}

class pointwise_energy
{
  protected:
    tensor<2> Y;
    tensor<2> S;
    tensor<2> L;
    tensor<2> W;
    tensor<2> CY;
    tensor<2> XY;
    tensor<2> TY;
    tensor<2> CW;
    tensor<2> XW;
    tensor<2> TW;
    tensor<2> D;
    tensor<2> CD;
    tensor<2> XD;
    tensor<2> TD;
    tensor<2> WD;
    std::vector<int> pvt;

  public:
    pointwise_energy(const cview<2>& VC,
                     const cview<2>& VX,
                     const cview<2>& T,
                     const cview<2>& Y_,
                     int nps = 0)
    : Y(Y_)
    {
        auto nvo = Y.length(0);
        auto ntot = Y.length(1);

        S.reset(gemm(Y.T(), Y));
        CY.reset(gemm(VC, Y));
        XY.reset(gemm(VX, Y));
        TY.reset(gemm(T, Y));

        /****************************************
         * Here, we pre-define some intermedia matrices
         *
         *  Yp, the pivoted Y matrix after we select the next grid point
         *
         *  Sp, the pivoted S matrix
         *
         *  Lp, the new cholsky decompostion pivoted L
         *
         *  Wp, WpLp^T = Yp, we could solve the euqation with TRSM.
         *
         *  d^T = Y^T(WpWp^T - I), d^T should be initalized as -YT.
         *
         *  gWp, gWp = (gY_s + gWp l_{10}) / \lamda_{11}, therefore, we
         * could pre-calculate gY(not gYp).
         *
         *  t^TWp, t^TWp = (t^TY_s + t^TWp l_10) / \lamda_{11}, we need
         * to pre-calculate t^TY.
         *
         *  gd, gd' = gd + (g\Delat Wp)(Y^T \Delta Wp)^T, we need get
         * gWp and Y^T
         * \Delta Wp.
         *
         *  td, td' = td + (t \Delta Wp)(Y^T \Delta Wp)^T
         *
         *
         ***************************************/
        L.reset({ntot, ntot});
        W.reset({nvo, ntot});
        CW.reset({nvo, ntot});
        XW.reset({nvo, ntot});
        TW.reset({nvo, ntot});
        D.reset(-Y);
        CD.reset(-CY);
        XD.reset(-XY);
        TD.reset(-TY);
        WD.reset({ntot, ntot});

        if (nps)
        {
            for (auto i : range(nps)) pvt.push_back(i);

            L[range(nps)][range(nps)] = S[range(nps)][range(nps)];
            potrf('L', nps, L.data(), L.stride(1));

            W[all][range(nps)] = Y[all][range(nps)];
            trsm('R',
                 'L',
                 'T',
                 'N',
                 nps,
                 nvo,
                 1.0,
                 L.data(),
                 L.stride(1),
                 W.data(),
                 W.stride(1));

            gemm(VC, W[all][range(nps)], CW[all][range(nps)]);
            gemm(VX, W[all][range(nps)], XW[all][range(nps)]);
            gemm(T, W[all][range(nps)], TW[all][range(nps)]);

            gemm3(W[all][range(nps)],
                  W[all][range(nps)].T(),
                  Y,
                  D[all][range(nps)]);

            gemm(VC, D[all][range(nps)], CD[all][range(nps)]);
            gemm(VX, D[all][range(nps)], XD[all][range(nps)]);
            gemm(T, D[all][range(nps)], TD[all][range(nps)]);
        }
    }

    std::pair<tensor<1>, tensor<1>> energy() const
    {
        auto nvo = Y.length(0);
        auto i = pvt.size();
        auto mu = -gemmdiag(Y.T(), D);

        /*************************
         *
         * The calculation of E4
         *
         *           T           T     T  T
         *  E = Tr( d    gW) (  d     t W)
         *         O*M   M*P   O*M    M*P
         * ***********************/

        tensor<1> E4Co =
            4
            * (i > nvo ? gemmdiag(gemm3(CW[all][range(i)],
                                        TW[all][range(i)].T(),
                                        D[all][range(i)]),
                                  D[all][range(i)].T())
                       : gemmdiag(
                             gemm(D[all][range(i)].T(), CW[all][range(i)]),
                             gemm(D[all][range(i)].T(), TW[all][range(i)]).T()))
            / mu;
        tensor<1> E4Xo =
            -2
            * (i > nvo ? gemmdiag(gemm3(XW[all][range(i)],
                                        TW[all][range(i)].T(),
                                        D[all][range(i)]),
                                  D[all][range(i)].T())
                       : gemmdiag(
                             gemm(D[all][range(i)].T(), XW[all][range(i)]),
                             gemm(D[all][range(i)].T(), TW[all][range(i)]).T()))
            / mu;

        /**************************
         *
         *  We only need to calculate the last term, \Delta E = 4\mu
         * Tr[(d^TgW)(d^Tt^TW)^T]
         *                                                    + 2
         * Tr[(\mud^Tgd)(\mud^Ttd)]
         *
         *
         *  In the begining, W was initlized as 0, we, therefore, only
         * need to calcute the last term, where d was initlized as -Y.
         *
         *
         *  E = 2 (  d^T   gd)  (   d^T   td )
         *          O*M    M*O      O*M   M*O
         *
         * ************************/

        tensor<1> E8Co = 2
                       * gemmdiag(D[all][range(i)].T(), CD[all][range(i)])
                       * gemmdiag(D[all][range(i)].T(), TD[all][range(i)])
                       / (mu * mu);
        tensor<1> E8Xo = -1
                       * gemmdiag(D[all][range(i)].T(), XD[all][range(i)])
                       * gemmdiag(D[all][range(i)].T(), TD[all][range(i)])
                       / (mu * mu);

        return {E4Co + E8Co, E4Xo + E8Xo};
    }

    void select_point(int idx)
    {
        auto ntot = Y.length(1);
        auto i = pvt.size();
        assert(idx > i && idx < ntot);
        pvt.push_back(idx);

        /***********************************************
         *
         *
         * Update YP matrix, here we need to update the i
         *
         * column with the idx column in Y matrix,
         *
         *
         *
         * *********************************************/
        PROFILE_SECTION("Y update")
        swap_ranges(Y[all][i].view(), Y[all][idx].view());
        swap_ranges(CY[all][i].view(), CY[all][idx].view());
        swap_ranges(XY[all][i].view(), XY[all][idx].view());
        swap_ranges(TY[all][i].view(), TY[all][idx].view());
        swap_ranges(S[all][i].view(), S[all][idx].view());
        swap_ranges(S[i][all].view(), S[idx][all].view());
        PROFILE_STOP

        /************************************************
         *
         *  L update.
         *
         *  L' = [  L00      *    ]
         *       [  l10    lambda ]
         *
         *      T
         *  YPmp Y[idx] = L l10,
         *
         *  solve by trsm, solve_tri
         *                      T              T
         *  lambda = sqrt(Y[idx]   Y[idx] - l10  l10);
         *                                 T
         *         = sqrt(S[idx][idx] - l10 l10)
         *
         *  for the first step, L00 is empty, l10 is therefore
         *  0, and lambda is sqrt(S[idx][idx]);
         *
         *
         ***********************************************/
        PROFILE_SECTION("L update")
        gemv(Y[all][range(i)].T(), Y[all][i], L[i][range(i)]);
        trsv('L',
             'N',
             'N',
             i,
             L.data(),
             L.stride(1),
             L[i][range(i)].data(),
             L.stride(1));
        L[i][i] = sqrt(S[i][i] - dot(L[i][range(i)], L[i][range(i)]));
        PROFILE_STOP

        /***********************************************
         *
         *
         *  W matrix update
         *
         *  \Delta W = (\Delta Y - WP l_10) / lambda
         *        M*1         M*1  M*P P*1
         *
         *  Fot the first step, WP is empty.
         *
         *  \Delta W = (\Delta Y) / lambda
         *        M*1         M*1
         *
         **********************************************/

        PROFILE_SECTION("W update")
        W[all][i] =
            (Y[all][i] - gemv(W[all][range(i)], L[i][range(i)])) / L[i][i];
        PROFILE_STOP

        /******************************************
         *
         *  g\DeltaW = (gY[idx] - gWP l_10) / lambda
         * M*1         M*1       M*P P*1
         *
         *  For the first step, l_10 is empty.
         *  g\Deltaw = gY[idx]
         *****************************************/

        PROFILE_SECTION("gCWp update")
        CW[all][i] =
            (CY[all][i] - gemv(CW[all][range(i)], L[i][range(i)])) / L[i][i];
        PROFILE_STOP

        PROFILE_SECTION("gXWp update")
        XW[all][i] =
            (XY[all][i] - gemv(XW[all][range(i)], L[i][range(i)])) / L[i][i];
        PROFILE_STOP

        /******************************************
         *   T             T          T
         *  t \Delta W = (t Y[idx] - t WP l_10) / lambda
         * M*1         M*1       M*P P*1
         *
         *  For the first step, l_10 is empty.
         *  t\Deltaw = tY[idx]
         *
         *   T           T
         *  t \Deltaw = t Y[idx]
         *****************************************/

        PROFILE_SECTION("tWp update")
        TW[all][i] =
            (TY[all][i] - gemv(TW[all][range(i)], L[i][range(i)])) / (L[i][i]);
        PROFILE_STOP

        /************************************************
         *          T
         *  Update d part
         *   T    T                 T
         *  d += Y \Delta W \Delta W
         * O*M  O*M      M*1      1*M
         *
         *************************************************/
        tensor<1> WY = gemv(Y.T(), W[all][i]);
        PROFILE_SECTION("dT update")
        ger(1.0, W[all][i], WY, 1.0, D);
        PROFILE_STOP

        /******************************************
         *
         * gd update
         *                        T         T
         *  gd + =  g \Delta W  (Y \Delta W)
         *  M*O    M*M      M*1 O*M      M*1
         *****************************************/
        PROFILE_SECTION("gCd update")
        ger(1.0, CW[all][i], WY, 1.0, CD);
        PROFILE_STOP

        PROFILE_SECTION("gXd update")
        ger(1.0, XW[all][i], WY, 1.0, XD);
        PROFILE_STOP

        /******************************************
         *
         * td update
         *                        T         T
         *  td + =  t \Delta W  (Y \Delta W)
         *  M*O    M*M      M*1 O*M      M*1
         *****************************************/
        PROFILE_SECTION("td update")
        ger(1.0, TW[all][i], WY, 1.0, TD);
        PROFILE_STOP
    }
};

/********************************************
 * main function
 ********************************************/

int main(int argc, char** argv)
{
    std::map<std::string, docopt::value> args =
        docopt::docopt(USAGE,
                       {argv + 1, argv + argc},
                       true,
                       "Energy Analysis 1.0");

    // Parse the input and setup the jobinfo struct
    Jobinfo jobinfo;
    if (parse_input(args, jobinfo) != 0)
    {
        printf("Bad input to energyAnalysis\n");
        exit(1);
    }

    // read the dimensions we need
    //  no        -> number of occupied orbitals
    //  nv        -> number of virtual orbitals
    //  nps       -> number of TOTAL grid points
    int nv, no, nps;
    read_dimensions(nv,
                    jobinfo.path_to_fa,
                    no,
                    jobinfo.path_to_fi,
                    nps,
                    jobinfo.path_to_xa);
    int nvo = nv * no;
    int norb = no + nv;
    printf("nvrt : %d \nnocc : %d \nngrd : %d \n", nv, no, nps);

    double threshold = 1.0e-10;

    // read Fock matrix elements
    // Note that the fa, fi, Xa, and Xi files all have offsets of Int*1,
    // as the first entry in the file is the relevant dimension
    auto faa = tensor_from_file(jobinfo.path_to_fa, sizeof(Int), nv, nv);
    auto fii = tensor_from_file(jobinfo.path_to_fi, sizeof(Int), no, no);
    auto FA = to_diagonal(faa);
    auto FI = to_diagonal(fii);

    // Read in vc,vx
    auto Vaibj = aibj_from_file(jobinfo.path_to_v, 0, nv, no);
    auto Vajbi = aibj_to_ajbi(Vaibj);
    auto VC = Vaibj.lowered(2);
    auto VX = Vajbi.lowered(2);
    sym_check(VC, threshold);
    sym_check(VX, threshold);

    // Generate T
    //  if method is MP2, we will generate amplitude from V.
    //  Otherwise, we will read amplitude from files.
    tensor<4> Taibj{nv, no, nv, no};
    if (jobinfo.method == "MP2")
    {
        make_c(Vaibj, FA, FI, Taibj);
    }
    else if (jobinfo.method == "MP3")
    {
        // first order of amplitude from MP2, second order of amplitude
        // from the file
        make_c(Vaibj, FA, FI, Taibj);
        auto Taibj_2 = aibj_from_file(jobinfo.path_to_t, 0, nv, no);
        Taibj += Taibj_2;
    }
    else if (jobinfo.method == "CCSD")
    {
        Taibj = aibj_from_file(jobinfo.path_to_t, 0, nv, no);
    }
    auto T = Taibj.lowered(2);
    sym_check(T, threshold);

    // read the THC matrix elements (note offset from start of file,
    // which skips the number of gridpoints
    auto xpa = tensor_from_file(jobinfo.path_to_xa, sizeof(Int), nps, nv);
    auto xpi = tensor_from_file(jobinfo.path_to_xi, sizeof(Int), nps, no);

    /***************************************
     * Form Y   , krp of occ and virt gridpoints
     *       MP'
     *
     *  M  = a x i (outer product), MO basis
     *  P  = THC gridpoints
     *  P' = (potentially) reduced set of THC gridpoints
     *
     *
     *  Y    =  X    *  X
     *   p'ai    p'a     p'i
     *
     *
     *  Y    =   Y     -> lower(ai) -> transpose
     *   MP'      P'ai
     *
     ***************************************/

    tensor<2> Y1 =
        krp(xpa[all][all], xpi[all][all]).lowered(1).T(); // nvo * nps

    auto nps1 = int(jobinfo.chi1 * norb);
    std::mt19937 rng(jobinfo.seed);
    std::uniform_int_distribution<int> rand(0, nps1 - 1);
    pair_list pairs;

    // Draw blindly up to n_pair, sort, remove duplicates,
    // draw more if necessary, repeat.
    while (pairs.size() < jobinfo.n_pair)
    {
        for (auto d : range(jobinfo.n_pair - pairs.size()))
        {
            auto i = rand(rng);
            auto j = rand(rng);

            if (j > i)
                std::swap(i, j);
            else if (i == j)
                continue;

            pairs.push_back({i, j});
        }

        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    }
    auto npair = pairs.size();

    tensor<2> Y11{nvo, npair};
    tensor<2> Y12{nvo, npair};
    tensor<2> Y21{nvo, npair};
    tensor<2> Y22{nvo, npair};

#pragma omp parallel
    for (int r = 0; auto [p, q] : pairs)
    {
#pragma omp for collapse(2)
        for (auto i : range(no))
        {
            for (auto a : range(nv))
            {
                auto ai = a + i * nv;
                Y11[ai][r] = xpa[p][a] * xpi[p][i];
                Y12[ai][r] = xpa[p][a] * xpi[q][i];
                Y21[ai][r] = xpa[q][a] * xpi[p][i];
                Y22[ai][r] = xpa[q][a] * xpi[q][i];
            }
        }

        r++;
    }

    tensor<2> Y2 = Y11 + Y12 + Y21 + Y22;
    tensor<2> Y{nvo, nps1 + npair};
    Y[all][range(nps1)] = Y1[all][range(nps1)];
    Y[all][range(nps1, nps1 + npair)] = Y2;

    pointwise_energy energy(VC, VX, T, Y, nps1);
    auto [EX, EC] = energy.energy();

    timer::print_timers();
}
