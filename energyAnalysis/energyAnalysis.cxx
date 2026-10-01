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
#include "marray.hpp"
#include "input.hpp"
#include "jobinfo.hpp"
#include "io.hpp"
#include "tensor_ops.hpp"
#include "pair_points.hpp"
#include "make_distance_pairs.hpp"
#include <array>
#include "qc_utility.hpp"
#include "cond.hpp"

#include "docopt.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <stdio.h>
#include <iomanip>
#include <set>
#include <chrono>
#include <random>
#include <sstream>
#include <tuple>
#include <map>
#include <utility>

/********************************************
 * main function
 ********************************************/

int main(int argc, char **argv) {

    std::map<std::string, docopt::value> args = docopt::docopt(USAGE,
            { argv+1, argv+argc },
            true,
            "Energy Analysis 1.0");

    //Parse the input and setup the jobinfo struct
    Jobinfo jobinfo;
    if (parse_input(args,jobinfo) != 0)
    {
        printf("Bad input to energyAnalysis\n");
        exit(1);
    }

    //read the dimensions we need
    // no        -> number of occupied orbitals
    // nv        -> number of virtual orbitals
    // nps       -> number of TOTAL grid points
    int nv, no, nps;
    read_dimensions(nv,  jobinfo.path_to_fa,
            no,  jobinfo.path_to_fi,
            nps, jobinfo.path_to_xa);
    int nvo = nv * no;
    int norb = no + nv ;
    printf("nvrt : %d \nnocc : %d \nngrd : %d \n", nv, no, nps);

    double threshold = 1.0e-10;

    //read Fock matrix elements
    //Note that the fa, fi, Xa, and Xi files all have offsets of Int*1,
    //as the first entry in the file is the relevant dimension
    auto faa = tensor_from_file(jobinfo.path_to_fa, sizeof(Int), nv, nv);
    auto fii = tensor_from_file(jobinfo.path_to_fi, sizeof(Int), no, no);
    auto FA = to_diagonal(faa);
    auto FI = to_diagonal(fii);

    //Read in vc,vx
    auto Vaibj = aibj_from_file(jobinfo.path_to_v, 0, nv, no);
    auto Vajbi = aibj_to_ajbi(Vaibj);
    auto VCmm = Vaibj.lowered(2);
    auto VXmm = Vajbi.lowered(2);
    sym_check(VCmm, threshold);
    sym_check(VXmm, threshold);

    //Generate T
    // if method is MP2, we will generate amplitude from V.
    // Otherwise, we will read amplitude from files.
    tensor<4> Taibj{nv, no, nv, no};
    if (jobinfo.method == "MP2")
    {
        make_c(Vaibj, FA, FI, Taibj);
    }
    else if (jobinfo.method == "MP3")
    {
        // first order of amplitude from MP2, second order of amplitude from the file
        make_c(Vaibj, FA, FI, Taibj);
        auto Taibj_2 = aibj_from_file(jobinfo.path_to_t, 0, nv, no);
        Taibj += Taibj_2;
    }
    else if (jobinfo.method == "CCSD")
    {
        Taibj = aibj_from_file(jobinfo.path_to_t, 0, nv, no);
    }
    auto Tmm = Taibj.lowered(2);

    double E_exact_c, E_exact_x;
    E_exact_c = 2 * E1(Vaibj, Taibj);
    E_exact_x = -E1(Vajbi, Taibj);

    //read the THC matrix elements (note offset from start of file,
    //which skips the number of gridpoints
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

    tensor<2> YT_sp = krp(xpa[all][all], xpi[all][all]).lowered(1); // nps * nvo
    tensor<2> Y_sp  = YT_sp.T();

    //stage 1: energy pivot single points only
    int ntot_stage1 = nps;

    tensor<2> Y = Y_sp;

    tensor<2> YT =Y.T(); // nvo * nps
    tensor<2> S = gemm(YT, Y); // ntot, ntot

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
     *  gWp, gWp = (gY_s + gWp l_{10}) / \lamda_{11}, therefore, we could pre-calculate gY(not gYp).
     *
     *  t^TWp, t^TWp = (t^TY_s + t^TWp l_10) / \lamda_{11}, we need to pre-calculate t^TY.
     *
     *  gd, gd' = gd + (g\Delat Wp)(Y^T \Delta Wp)^T, we need get gWp and Y^T \Delta Wp.
     *
     *  td, td' = td + (t \Delta Wp)(Y^T \Delta Wp)^T
     *
     *
     ***************************************/
    tensor<2> YPmp{nvo, ntot_stage1};
    tensor<2> LPpp{ntot_stage1, ntot_stage1};
    tensor<2> WPmp{nvo, ntot_stage1};
    tensor<2> gCWPmp{nvo, ntot_stage1};
    tensor<2> gXWPmp{nvo, ntot_stage1};
    tensor<2> tTWPmp{nvo, ntot_stage1};
    tensor<2> tWPmp{nvo, ntot_stage1};  // In the update of td, we also need tW.

    auto gCY = gemm(VCmm, Y);
    auto gXY = gemm(VXmm, Y);
    auto tY = gemm(Tmm, Y);
    auto tTY = gemm(Tmm.T(), Y);

    tensor<2> dTPom = -YT;
    tensor<2> gCdPmo = -gCY;
    tensor<2> gXdPmo = -gXY;
    tensor<2> tdPmo = -tY;

    std::set<int> selected_points {};
    std::vector<int> pvt;
    double total_EC = 0.0;
    double total_EX = 0.0;
    double total_E = 0.0;

    int num_grid_keep = std::min(static_cast<int>(jobinfo.chi1 * norb), nps);

    double cond = 1.0;

    // constract a cond number estimator class
    triangular_condition_number_estimator cond_num_esti(ntot_stage1);

    for(int i = 0; i < num_grid_keep; i++)
    {
        int idx;
        printf("\n\nThe %d step\n", i);
        int ngrid = i + 1;
        if (i == 0)
        {
            /*******************************
             *
             * We also need to calculate \mu
             *            T         T        T
             * mu = diag(Y   Y -   Y    W   W   Y)
             *          om   mo   om    mp pm  mo
             *******************************/
            tensor<1> MUo = gemmdiag(YT, Y);

            /*************************
             *
             * The calculation of E4 for the firsr step is skipped
             * due to W is empty
             *
             * ***********************/

            /**************************
             *
             *  We only need to calculate the last term, \Delta E = 4\mu Tr[(d^TgW)(d^Tt^TW)^T]
             *                                                    + 2    Tr[(\mud^Tgd)(\mud^Ttd)]
             *
             *
             *  In the begining, W was initlized as 0, we, therefore, only need to calcute the last term,
             *  where d was initlized as -Y.
             *
             *
             *  E = 2 (  d^T   g    d)  (   d^T   t    d )
             *          O*M   M*M  M*O     O*M   M*M  M*O
             *
             * ************************/

            tensor<1> E8Co = 2 * gemmdiag(dTPom, gCdPmo) * gemmdiag(dTPom, tdPmo) / (MUo * MUo);
            tensor<1> E8Xo = -1.0 * gemmdiag(dTPom, gXdPmo) * gemmdiag(dTPom, tdPmo) / (MUo * MUo);

            tensor<1> Eo = E8Co + E8Xo;

            std::cout << "Ene Piv: EC, EX = " << E8Co[i] << ", " << E8Xo[i]  << std::endl;
            if (getenv("CHOL"))
                idx = i;
            else
                idx = pivot_ene(Eo, selected_points, pvt);

            double ECs = E8Co[idx];
            double EXs = E8Xo[idx];
            total_EC += E8Co[idx];
            total_EX += E8Xo[idx];
            total_E += Eo[idx];

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
            YPmp[all][i] = Y[all][idx];

            /************************************************
             *
             *  L update.
             *
             *  L' = [  L00      *    ]
             *       [  l10    lambda ]
             *
             *      T
             *  YPmp Y[idx] = LPpp l10,
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

            LPpp[i][i] = sqrt(S[idx][idx]);

            // Note: this local 'cond' shadows the outer one, the CSV shows 1.0 untill step 20

            double cond = cond_num_esti.update(LPpp[i][range(i+1)]);

            output_to_csv(jobinfo, idx, ngrid, norb, E_exact_c, E_exact_x, ECs, EXs, total_EC, total_EX, LPpp[i][i], cond);

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
            WPmp[all][i] = YPmp[all][i] * (1 / LPpp[i][i]);

            /************************************************
             *          T
             *  Update d part
             *   T    T                 T
             *  d += Y \Delta W \Delta W
             * O*M  O*M      M*1      1*M
             *
             *************************************************/

            tensor<1> YTW = gemv(YT, WPmp[all][i]);
            ger(1.0, YTW, WPmp[all][i], 1.0, dTPom);

            /******************************************
             *
             *  g\DeltaW = (gY[idx] - gWP l_10) / lambda
             * M*1         M*1       M*P P*1
             *
             *  For the first step, l_10 is empty.
             *  g\Deltaw = gY[idx]
             *****************************************/
            gCWPmp[all][i] = gCY[all][idx] / (LPpp[i][i]);
            gXWPmp[all][i] = gXY[all][idx] / (LPpp[i][i]);

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
            tWPmp[all][i] = tY[all][idx] / (LPpp[i][i]);
            tTWPmp[all][i] = tTY[all][idx] / (LPpp[i][i]);

            /******************************************
             *
             * gd update
             *                        T         T
             *  gd + =  g \Delta W  (Y \Delta W)
             *  M*O    M*M      M*1 O*M      M*1
             *****************************************/
            ger(1.0, gCWPmp[all][i] ,YTW, 1.0, gCdPmo);
            ger(1.0, gXWPmp[all][i] ,YTW, 1.0, gXdPmo);

            /******************************************
             *
             * td update
             *                        T         T
             *  td + =  t \Delta W  (Y \Delta W)
             *  M*O    M*M      M*1 O*M      M*1
             *****************************************/
            ger(1.0, tWPmp[all][i] ,YTW, 1.0, tdPmo);

        }
        else
        {

            tensor<2> WP = WPmp[all][range(i)];
            tensor<2> Dmo = Y;
            gemm3(-1.0, WP, WP.T(), Y, 1.0, Dmo);
            tensor<1> MUo = gemmdiag(YT, Dmo);

            /*************************
             *
             * The calculation of E4
             *
             *           T           T     T  T
             *  E = Tr( d    gW) (  d     t W)
             *         O*M   M*P   O*M    M*P
             * ***********************/

            tensor<1> E4Co = 4.0 * gemmdiag(gemm(dTPom, gCWPmp[all][range(i)]), gemm(dTPom, tTWPmp[all][range(i)]).T()) / MUo;
            tensor<1> E4Xo = -2.0 * gemmdiag(gemm(dTPom, gXWPmp[all][range(i)]), gemm(dTPom, tTWPmp[all][range(i)]).T()) / MUo;

            /**************************
             *
             *  We only need to calculate the last term, \Delta E = 4\mu Tr[(d^TgW)(d^Tt^TW)^T]
             *                                                    + 2    Tr[(\mud^Tgd)(\mud^Ttd)]
             *
             *
             *  In the begining, W was initlized as 0, we, therefore, only need to calcute the last term,
             *  where d was initlized as -Y.
             *
             *
             *  E = 2 (  d^T   gd)  (   d^T   td )
             *          O*M    M*O      O*M   M*O
             *
             * ************************/

            tensor<1> E8Co = 2.0 * gemmdiag(dTPom, gCdPmo) * gemmdiag(dTPom, tdPmo) / (MUo * MUo);
            tensor<1> E8Xo = -1.0 * gemmdiag(dTPom, gXdPmo) * gemmdiag(dTPom, tdPmo) / (MUo * MUo);

            tensor<1> EC = E4Co + E8Co;
            tensor<1> EX = E4Xo + E8Xo;

            tensor<1> Eo = E4Co + E8Co + E4Xo + E8Xo;

            if (getenv("CHOL"))
                idx = i;
            else
                idx = pivot_ene(Eo, selected_points, pvt);

            double ECs = EC[idx];
            double EXs = EX[idx];

            std::cout << "Selected energy: EC = " << ECs
                << " EX = " << EXs
                << " E = " << Eo[idx]
                << std::endl;

            total_EC += EC[idx];
            total_EX += EX[idx];
            total_E += Eo[idx];

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
                YPmp[all][i] = Y[all][idx];
            PROFILE_STOP

                /************************************************
                 *
                 *  L update.
                 *
                 *  L' = [  L00      *    ]
                 *       [  l10    lambda ]
                 *
                 *      T
                 *  YPmp Y[idx] = LPpp l10,
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
                auto YPTYS = gemv(YPmp[all][range(i)].T(), YPmp[all][i]);

            chaotrsv('L', LPpp[range(i)][range(i)], YPTYS);

            LPpp[i][range(i)] = YPTYS;
            LPpp[i][i] = sqrt(S[idx][idx] - dot(LPpp[i][range(i)], LPpp[i][range(i)]));

            if (i % 20 == 0)
            {
                char jobu = 'N';
                char jobvt = 'N';
                int m = i + 1;
                int n = i + 1;
                int lda = i + 1;
                tensor<1> s{n};
                tensor<2> u{m, n};
                int ldu = i + 1;
                tensor<2> vt{n, n};
                int ldvt = i + 1;
                tensor<2> A = LPpp[range(i+1)][range(i+1)];

                c_dgesvd( jobu, jobvt, m, n, A.data(), lda, s.data(), u.data(),
                        ldu, vt.data(), ldvt);

                cond = max<double>(s) / min<double>(s);
            }

            PROFILE_STOP

                output_to_csv(jobinfo, idx, ngrid, norb, E_exact_c, E_exact_x, ECs, EXs, total_EC, total_EX, LPpp[i][i], cond);
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
                WPmp[all][i] = (YPmp[all][i] - gemv(WPmp[all][range(i)], LPpp[i][range(i)])) * (1 / LPpp[i][i]);
            PROFILE_STOP

                /************************************************
                 *          T
                 *  Update d part
                 *   T    T                 T
                 *  d += Y \Delta W \Delta W
                 * O*M  O*M      M*1      1*M
                 *
                 *************************************************/
                tensor<1> YTW = gemv(YT, WPmp[all][i]);
            PROFILE_SECTION("dT update")
                ger(1.0, YTW, WPmp[all][i], 1.0, dTPom);
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
                gCWPmp[all][i] = (gCY[all][idx] - gemv(gCWPmp[all][range(i)], LPpp[i][range(i)])) / (LPpp[i][i]);
            PROFILE_STOP

                PROFILE_SECTION("gXWp update")
                gXWPmp[all][i] = (gXY[all][idx] - gemv(gXWPmp[all][range(i)], LPpp[i][range(i)])) / (LPpp[i][i]);
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
                tWPmp[all][i] = (tY[all][idx] - gemv(tWPmp[all][range(i)], LPpp[i][range(i)])) / (LPpp[i][i]);
            PROFILE_STOP

                PROFILE_SECTION("tTWp update")
                tTWPmp[all][i] = (tTY[all][idx] - gemv(tTWPmp[all][range(i)], LPpp[i][range(i)])) / (LPpp[i][i]);
            PROFILE_STOP

                /******************************************
                 *
                 * gd update
                 *                        T         T
                 *  gd + =  g \Delta W  (Y \Delta W)
                 *  M*O    M*M      M*1 O*M      M*1
                 *****************************************/
                PROFILE_SECTION("gCd update")
                ger(1.0, gCWPmp[all][i], YTW, 1.0, gCdPmo);
            PROFILE_STOP

                PROFILE_SECTION("gXd update")
                ger(1.0, gXWPmp[all][i], YTW, 1.0, gXdPmo);
            PROFILE_STOP

                /******************************************
                 *
                 * td update
                 *                        T         T
                 *  td + =  t \Delta W  (Y \Delta W)
                 *  M*O    M*M      M*1 O*M      M*1
                 *****************************************/
                PROFILE_SECTION("td update")
                ger(1.0, tWPmp[all][i] ,YTW, 1.0, tdPmo);
            PROFILE_STOP

        }

        printf("Stage 1 selected point : %d\n", idx);

    }
    //verify stage 1 selections
    printf("stage 1 selected count = %zu\n", pvt.size());

    //===== Step 3: random unique pairs of Stage-1 points =====

    const int n_stage1 = static_cast<int>(pvt.size());

    long long max_possible_pairs =
        static_cast<long long>(n_stage1) * (n_stage1 - 1) / 2;

    printf("Maximum possible Stage-1 pairs = %lld\n",
            max_possible_pairs);

    if (jobinfo.n_pair > max_possible_pairs)
    {
        printf("ERROR: requested n_pair = %d, but only %lld uniques pairs are possible.\n",
                jobinfo.n_pair, max_possible_pairs);
        return 1;
    }

    std::mt19937 rng(jobinfo.seed);
    std::uniform_int_distribution<int> random_single(0, n_stage1 - 1);
    std::uniform_int_distribution<int> random_other(0, n_stage1 - 2);

    pair_list pairs;

    // Draw blindly up to n_pair, sort, remove duplicates,
    // draw more if necessary, repeat.
    while (static_cast<int>(pairs.size()) < jobinfo.n_pair)
    {
        int n_draw = jobinfo.n_pair - static_cast<int>(pairs.size());

        for (int d = 0; d < n_draw; d++)
        {
            int i = random_single(rng);
            int j = random_other(rng);
            if (j >= i) j++;   // guarantees j != i, still uniform

            int p = pvt[i];
            int q = pvt[j];
            if (p > q) std::swap(p, q);
            pairs.push_back({p, q});
        }

        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    }

    printf("requested random pairs = %d\n", jobinfo.n_pair);
    printf("generated random pairs = %zu\n", pairs.size());

    //===== Branch a, Step 5a: rank pairs by stand alone energy =====

    if (jobinfo.branch == "a")
    {
        int npairs_stage5a = static_cast<int>(pairs.size());

        printf("\n--- Pair Energy Ranking ---\n");
        printf("Number of pair candidates = %d\n", npairs_stage5a);

        // Construct Y columns for the random pair points
        tensor<2> Y2{nvo, npairs_stage5a};
        Y2 = make_Y2(xpa, xpi, pairs, jobinfo);

        tensor<2> Y2T = Y2.T();

        //same intermediates as the energy-pivot step,
        //but now Y contains Only pair-point columns.
        auto gCY_pair = gemm(VCmm, Y2);
        auto gXY_pair = gemm(VXmm, Y2);
        auto tY_pair  = gemm(Tmm, Y2);

        tensor<2> dT_pair = -Y2T;
        tensor<2> gCd_pair = -gCY_pair;
        tensor<2> gXd_pair = -gXY_pair;
        tensor<2> td_pair = -tY_pair;

        //mu for every pair candidate
        tensor<1> MU_pair = gemmdiag(Y2T, Y2);

        //Coulomb energy contribution

        tensor<1> EC_pair =
            2.0 * gemmdiag(dT_pair, gCd_pair)
            * gemmdiag(dT_pair, td_pair)
            / (MU_pair * MU_pair);

        //Exchange energy contribution

        tensor<1> EX_pair =
            -1.0 * gemmdiag(dT_pair, gXd_pair)
            * gemmdiag(dT_pair, td_pair)
            / (MU_pair * MU_pair);

        //total energy contribution
        tensor<1> E_pair = EC_pair + EX_pair;

        printf("Pair energy contributions calculated = %d\n",
                npairs_stage5a);

        //print only first 5 pairs for checking
        int nprint = std::min(5, npairs_stage5a);

        for (int k = 0; k < nprint; k++)
        {
            printf("Pair %d : (%d,%d) EC = %.10e EX = %.10e  E = %.10e\n",
                    k,
                    pairs[k].first,
                    pairs[k].second,
                    EC_pair[k],
                    EX_pair[k],
                    E_pair[k]);
        }
        //Rank pair candidtaes by energy contribution

        // check ranking options
        if (jobinfo.rank_energy != "total" &&
                jobinfo.rank_energy != "exchange")
        {
            printf("Error: --rank-energy must be 'total' or 'exchange'\n");
            return 1;
        }

        //store pair indices: 0, 1, 2, ..., npairs_stage5a-1
        std::vector<int> pair_order(npairs_stage5a);

        for (int k = 0; k < npairs_stage5a; k++)
        {
            pair_order[k] = k;
        }

        //rank by magnitude of requested energy contribution
        std::sort(pair_order.begin(), pair_order.end(),
                [&](int a, int b)
                {
                double score_a;
                double score_b;

                if (jobinfo.rank_energy == "exchange")
                {
                score_a = std::abs(EX_pair[a]);
                score_b = std::abs(EX_pair[b]);
                }
                else
                {
                score_a = std::abs(E_pair[a]);
                score_b = std::abs(E_pair[b]);
                }

                return score_a > score_b;
                });

        //verify ranking is desscending order
        for (int r = 1; r < npairs_stage5a; r++)
        {
            int prev = pair_order[r - 1];
            int curr = pair_order[r];

            double score_prev;
            double score_curr;

            if (jobinfo.rank_energy == "exchange")
            {
                score_prev = std::abs(EX_pair[prev]);
                score_curr = std::abs(EX_pair[curr]);
            }
            else
            {
                score_prev = std::abs(E_pair[prev]);
                score_curr = std::abs(E_pair[curr]);
            }

            if (score_curr > score_prev)
            {
                printf("Error: pair ranking is not in descending order\n");
                return 1;
            }
        }

        printf("pair ranking order verified\n");

        //number of top-ranked pairs to retain
        int n_top_keep = std::min(jobinfo.n_top, npairs_stage5a);

        //store actual top ranked pairs
        pair_list top_pairs_a;

        for (int r = 0; r < n_top_keep; r++)
        {
            int k = pair_order[r];
            top_pairs_a.push_back(pairs[k]);
        }

        printf("\n -- Top pair selection --\n");
        printf(" ranking criterion = %s\n", jobinfo.rank_energy.c_str());
        printf("pair candidates = %d\n", npairs_stage5a);
        printf("top ranked pairs selected = %zu\n", top_pairs_a.size());

        //print first five ranked pairs
        int nprint_top = std::min(5, n_top_keep);

        for (int r = 0; r < nprint_top; r++)
        {
            int k = pair_order[r];

            printf("Rank %d : Pair (%d,%d) EC = %.10e EX = %.10e E = %.10e\n",
                    r + 1,
                    pairs[k].first,
                    pairs[k].second,
                    EC_pair[k],
                    EX_pair[k],
                    E_pair[k]);
        }

        // ===== Branch a. Stepp 6a: pivot over singles + top pairs =====

        printf("\n ---Combined Energy Pivot --\n");

        int nsingles_6a = static_cast<int>(pvt.size());
        int npairs_6a = static_cast<int>(top_pairs_a.size());
        int ntot_6a = nsingles_6a + npairs_6a;

        printf("Stage1 singles  = %d\n", nsingles_6a);
        printf("Top ranked pairs  = %d\n", npairs_6a);
        printf("Combined candidates  = %d\n", ntot_6a);

        //Stage1 selected single-point columns
        tensor<2> Ysingle_6a{nvo, nsingles_6a};

        for (int k = 0; k < nsingles_6a; k++)
        {
            Ysingle_6a[all][k] = Y_sp[all][pvt[k]];
        }

        //top ranked pair point columns

        tensor<2> Ypair_6a = make_Y2(xpa, xpi, top_pairs_a, jobinfo);

        //Combine selected singles and top ranked pairs
        tensor<2> Y_6a{nvo, ntot_6a};

        Y_6a[all][range(0, nsingles_6a)] = Ysingle_6a;
        Y_6a[all][range(nsingles_6a, ntot_6a)] = Ypair_6a;

        printf("Y_6a dimensions = %ld x %ld\n",
                Y_6a.length(0), Y_6a.length(1));

        int num_grid_keep_6a =
            std::min(static_cast<int>(jobinfo.chi2a * norb), ntot_6a);

        printf("chi2a                  = %.2f\n", jobinfo.chi2a);
        printf("norb                   = %d\n", norb);
        printf("Combined pivot points to keep = %d\n", num_grid_keep_6a);

        //Initialize Combined energy pivot

        tensor<2> YT_6a = Y_6a.T();
        tensor<2> S_6a  = gemm(YT_6a, Y_6a);

        tensor<2> YPmp_6a{nvo, ntot_6a};
        tensor<2> LPpp_6a{ntot_6a, ntot_6a};
        tensor<2> WPmp_6a{nvo, ntot_6a};

        tensor<2> gCWPmp_6a{nvo, ntot_6a};
        tensor<2> gXWPmp_6a{nvo, ntot_6a};
        tensor<2> tTWPmp_6a{nvo, ntot_6a};
        tensor<2> tWPmp_6a{nvo, ntot_6a};

        auto gCY_6a = gemm(VCmm, Y_6a);
        auto gXY_6a = gemm(VXmm, Y_6a);
        auto tY_6a  = gemm(Tmm, Y_6a);
        auto tTY_6a = gemm(Tmm.T(), Y_6a);

        tensor<2> dTPom_6a = -YT_6a;
        tensor<2> gCdPmo_6a = -gCY_6a;
        tensor<2> gXdPmo_6a = -gXY_6a;
        tensor<2> tdPmo_6a = -tY_6a;

        std::set<int> selected_points_6a {};
        std::vector<int> pvt_6a;
        double total_EC_6a = 0.0;
        double total_EX_6a = 0.0;
        double total_E_6a = 0.0;

        double cond_6a = 1.0;

        triangular_condition_number_estimator cond_num_esti_6a(ntot_6a);

        for (int i = 0; i < num_grid_keep_6a; i++)
        {
            int idx_6a;

            printf("\n6a pivot step %d\n", i);

            if (i == 0)
            {
                tensor<1> MUo_6a = gemmdiag(YT_6a, Y_6a);
                tensor<1> E8Co_6a =
                    2.0 * gemmdiag(dTPom_6a, gCdPmo_6a)
                    * gemmdiag(dTPom_6a, tdPmo_6a)
                    / (MUo_6a * MUo_6a);

                tensor<1> E8Xo_6a =
                    -1.0 * gemmdiag(dTPom_6a, gXdPmo_6a)
                    * gemmdiag(dTPom_6a, tdPmo_6a)
                    / (MUo_6a * MUo_6a);
                tensor<1> Eo_6a = E8Co_6a + E8Xo_6a;

                idx_6a = pivot_ene(Eo_6a,
                        selected_points_6a,
                        pvt_6a);

                double ECs_6a = E8Co_6a[idx_6a];
                double EXs_6a = E8Xo_6a[idx_6a];
                double E_6a = Eo_6a[idx_6a];

                total_EC_6a += ECs_6a;
                total_EX_6a += EXs_6a;
                total_E_6a += E_6a;

                printf("selected point = %d\n", idx_6a);
                printf("selected energy: EC = %.10e EX = %.10e E = %.10e \n",
                        ECs_6a,
                        EXs_6a,
                        Eo_6a[idx_6a]);

                YPmp_6a[all][i] = Y_6a[all][idx_6a];
                LPpp_6a[i][i] = sqrt(S_6a[idx_6a][idx_6a]);

                cond_6a =
                    cond_num_esti_6a.update(LPpp_6a[i][range(i + 1)]);

                WPmp_6a[all][i] =
                    YPmp_6a[all][i] * (1.0 / LPpp_6a[i][i]);

                tensor<1> YTW_6a =
                    gemv(YT_6a, WPmp_6a[all][i]);

                ger(1.0,
                        YTW_6a,
                        WPmp_6a[all][i],
                        1.0,
                        dTPom_6a);

                gCWPmp_6a[all][i] =
                    gCY_6a[all][idx_6a] / LPpp_6a[i][i];

                gXWPmp_6a[all][i] =
                    gXY_6a[all][idx_6a] / LPpp_6a[i][i];

                tWPmp_6a[all][i] =
                    tY_6a[all][idx_6a] / LPpp_6a[i][i];

                tTWPmp_6a[all][i] =
                    tTY_6a[all][idx_6a] / LPpp_6a[i][i];

                ger(1.0,
                        gCWPmp_6a[all][i],
                        YTW_6a,
                        1.0,
                        gCdPmo_6a);

                ger(1.0,
                        gXWPmp_6a[all][i],
                        YTW_6a,
                        1.0,
                        gXdPmo_6a);

                ger(1.0,
                        tWPmp_6a[all][i],
                        YTW_6a,
                        1.0,
                        tdPmo_6a);
            }
            else
            {
                tensor<2> WP_6a = WPmp_6a[all][range(i)];
                tensor<2> Dmo_6a = Y_6a;

                gemm3(-1.0,
                        WP_6a,
                        WP_6a.T(),
                        Y_6a,
                        1.0,
                        Dmo_6a);

                tensor<1> MUo_6a = gemmdiag(YT_6a, Dmo_6a);

                tensor<1> E4Co_6a =
                    4.0 * gemmdiag(
                            gemm(dTPom_6a, gCWPmp_6a[all][range(i)]),
                            gemm(dTPom_6a, tTWPmp_6a[all][range(i)]).T()
                            )    / MUo_6a;

                tensor<1> E4Xo_6a =
                    -2.0 * gemmdiag(
                            gemm(dTPom_6a, gXWPmp_6a[all][range(i)]),
                            gemm(dTPom_6a, tTWPmp_6a[all][range(i)]).T()
                            )    / MUo_6a;

                tensor<1> E8Co_6a =
                    2.0 * gemmdiag(dTPom_6a, gCdPmo_6a)
                    * gemmdiag(dTPom_6a, tdPmo_6a)
                    / (MUo_6a * MUo_6a);

                tensor<1> E8Xo_6a =
                    -1.0 * gemmdiag(dTPom_6a, gXdPmo_6a)
                    * gemmdiag(dTPom_6a, tdPmo_6a)
                    / (MUo_6a * MUo_6a);

                tensor<1> EC_6a = E4Co_6a + E8Co_6a;
                tensor<1> EX_6a = E4Xo_6a + E8Xo_6a;

                tensor<1> Eo_6a = EC_6a + EX_6a;

                idx_6a =
                    pivot_ene(Eo_6a,
                            selected_points_6a,
                            pvt_6a);

                double ECs_6a = EC_6a[idx_6a];
                double EXs_6a = EX_6a[idx_6a];

                total_EC_6a += ECs_6a;
                total_EX_6a += EXs_6a;
                total_E_6a += Eo_6a[idx_6a];

                printf("selected point = %d\n", idx_6a);

                printf("selected energy: EC = %.10e EX = %.10e E = %.10e \n",
                        ECs_6a,
                        EXs_6a,
                        Eo_6a[idx_6a]);

                YPmp_6a[all][i] = Y_6a[all][idx_6a];

                auto YPTYS_6a =
                    gemv(YPmp_6a[all][range(i)].T(),
                            YPmp_6a[all][i]);

                chaotrsv('L',
                        LPpp_6a[range(i)][range(i)],
                        YPTYS_6a);

                LPpp_6a[i][range(i)] = YPTYS_6a;

                LPpp_6a[i][i] =
                    sqrt(
                            S_6a[idx_6a][idx_6a] -
                            dot(LPpp_6a[i][range(i)],
                                LPpp_6a[i][range(i)])
                        );

                WPmp_6a[all][i] =
                    (YPmp_6a[all][i] -
                     gemv(WPmp_6a[all][range(i)],
                         LPpp_6a[i][range(i)]))
                    / LPpp_6a[i][i];

                tensor<1> YTW_6a =
                    gemv(YT_6a, WPmp_6a[all][i]);

                ger(1.0,
                        YTW_6a,
                        WPmp_6a[all][i],
                        1.0,
                        dTPom_6a);

                gCWPmp_6a[all][i] =
                    (gCY_6a[all][idx_6a] -
                     gemv(gCWPmp_6a[all][range(i)],
                         LPpp_6a[i][range(i)]))
                    / LPpp_6a[i][i];

                gXWPmp_6a[all][i] =
                    (gXY_6a[all][idx_6a] -
                     gemv(gXWPmp_6a[all][range(i)],
                         LPpp_6a[i][range(i)]))
                    / LPpp_6a[i][i];

                tWPmp_6a[all][i] =
                    (tY_6a[all][idx_6a] -
                     gemv(tWPmp_6a[all][range(i)],
                         LPpp_6a[i][range(i)]))
                    / LPpp_6a[i][i];

                tTWPmp_6a[all][i] =
                    (tTY_6a[all][idx_6a] -
                     gemv(tTWPmp_6a[all][range(i)],
                         LPpp_6a[i][range(i)]))
                    / LPpp_6a[i][i];

                ger(1.0,
                        gCWPmp_6a[all][i],
                        YTW_6a,
                        1.0,
                        gCdPmo_6a);

                ger(1.0,
                        gXWPmp_6a[all][i],
                        YTW_6a,
                        1.0,
                        gXdPmo_6a);

                ger(1.0,
                        tWPmp_6a[all][i],
                        YTW_6a,
                        1.0,
                        tdPmo_6a);

            }
        }

        printf("\nCombined Energy pivot complete\n");
        printf("selected combined point = %zu\n", pvt_6a.size());
        printf("Combined EC = %.10f\n", total_EC_6a);
        printf("Combined EX = %.10f\n", total_EX_6a);
        printf("Combined E = %.10f\n", total_E_6a);
    }

    else if (jobinfo.branch == "b")
    {
        int npairs = static_cast<int>(pairs.size());
        int n_5b = std::min(static_cast<int>(jobinfo.chi2b * norb), n_stage1);
        if (n_5b < 1) { printf("ERROR: chi2b gives zero singles\n"); return 1; }
        // ===== Branch b, Step 5b: re-pivot stage-1 singles =====

        int ntot_5r = n_stage1;
        tensor<2> Y_5r{nvo, ntot_5r};
        for (int k = 0; k < ntot_5r; k++)
            Y_5r[all][k] = Y_sp[all][pvt[k]];
        int num_grid_keep_5r = n_5b;
        tensor<2> YT_5r = Y_5r.T();
        tensor<2> S_5r  = gemm(YT_5r, Y_5r);

        tensor<2> YPmp_5r{nvo, ntot_5r};
        tensor<2> LPpp_5r{ntot_5r, ntot_5r};
        tensor<2> WPmp_5r{nvo, ntot_5r};

        tensor<2> gCWPmp_5r{nvo, ntot_5r};
        tensor<2> gXWPmp_5r{nvo, ntot_5r};
        tensor<2> tTWPmp_5r{nvo, ntot_5r};
        tensor<2> tWPmp_5r{nvo, ntot_5r};

        auto gCY_5r = gemm(VCmm, Y_5r);
        auto gXY_5r = gemm(VXmm, Y_5r);
        auto tY_5r  = gemm(Tmm, Y_5r);
        auto tTY_5r = gemm(Tmm.T(), Y_5r);

        tensor<2> dTPom_5r = -YT_5r;
        tensor<2> gCdPmo_5r = -gCY_5r;
        tensor<2> gXdPmo_5r = -gXY_5r;
        tensor<2> tdPmo_5r = -tY_5r;

        std::set<int> selected_points_5r {};
        std::vector<int> pvt_5r;
        double total_EC_5r = 0.0;
        double total_EX_5r = 0.0;
        double total_E_5r = 0.0;

        double cond_5r = 1.0;

        triangular_condition_number_estimator cond_num_esti_5r(ntot_5r);

        for (int i = 0; i < num_grid_keep_5r; i++)
        {
            int idx_5r;

            printf("\n5b pivot step %d\n", i);

            if (i == 0)
            {
                tensor<1> MUo_5r = gemmdiag(YT_5r, Y_5r);
                tensor<1> E8Co_5r =
                    2.0 * gemmdiag(dTPom_5r, gCdPmo_5r)
                    * gemmdiag(dTPom_5r, tdPmo_5r)
                    / (MUo_5r * MUo_5r);

                tensor<1> E8Xo_5r =
                    -1.0 * gemmdiag(dTPom_5r, gXdPmo_5r)
                    * gemmdiag(dTPom_5r, tdPmo_5r)
                    / (MUo_5r * MUo_5r);
                tensor<1> Eo_5r = E8Co_5r + E8Xo_5r;

                idx_5r = pivot_ene(Eo_5r,
                        selected_points_5r,
                        pvt_5r);

                double ECs_5r = E8Co_5r[idx_5r];
                double EXs_5r = E8Xo_5r[idx_5r];
                double E_5r = Eo_5r[idx_5r];

                total_EC_5r += ECs_5r;
                total_EX_5r += EXs_5r;
                total_E_5r += E_5r;

                printf("selected point = %d\n", idx_5r);
                printf("selected energy: EC = %.10e EX = %.10e E = %.10e \n",
                        ECs_5r,
                        EXs_5r,
                        Eo_5r[idx_5r]);

                YPmp_5r[all][i] = Y_5r[all][idx_5r];
                LPpp_5r[i][i] = sqrt(S_5r[idx_5r][idx_5r]);

                cond_5r =
                    cond_num_esti_5r.update(LPpp_5r[i][range(i + 1)]);

                WPmp_5r[all][i] =
                    YPmp_5r[all][i] * (1.0 / LPpp_5r[i][i]);

                tensor<1> YTW_5r =
                    gemv(YT_5r, WPmp_5r[all][i]);

                ger(1.0,
                        YTW_5r,
                        WPmp_5r[all][i],
                        1.0,
                        dTPom_5r);

                gCWPmp_5r[all][i] =
                    gCY_5r[all][idx_5r] / LPpp_5r[i][i];

                gXWPmp_5r[all][i] =
                    gXY_5r[all][idx_5r] / LPpp_5r[i][i];

                tWPmp_5r[all][i] =
                    tY_5r[all][idx_5r] / LPpp_5r[i][i];

                tTWPmp_5r[all][i] =
                    tTY_5r[all][idx_5r] / LPpp_5r[i][i];

                ger(1.0,
                        gCWPmp_5r[all][i],
                        YTW_5r,
                        1.0,
                        gCdPmo_5r);

                ger(1.0,
                        gXWPmp_5r[all][i],
                        YTW_5r,
                        1.0,
                        gXdPmo_5r);

                ger(1.0,
                        tWPmp_5r[all][i],
                        YTW_5r,
                        1.0,
                        tdPmo_5r);
            }
            else
            {
                tensor<2> WP_5r = WPmp_5r[all][range(i)];
                tensor<2> Dmo_5r = Y_5r;

                gemm3(-1.0,
                        WP_5r,
                        WP_5r.T(),
                        Y_5r,
                        1.0,
                        Dmo_5r);

                tensor<1> MUo_5r = gemmdiag(YT_5r, Dmo_5r);

                tensor<1> E4Co_5r =
                    4.0 * gemmdiag(
                            gemm(dTPom_5r, gCWPmp_5r[all][range(i)]),
                            gemm(dTPom_5r, tTWPmp_5r[all][range(i)]).T()
                            )    / MUo_5r;

                tensor<1> E4Xo_5r =
                    -2.0 * gemmdiag(
                            gemm(dTPom_5r, gXWPmp_5r[all][range(i)]),
                            gemm(dTPom_5r, tTWPmp_5r[all][range(i)]).T()
                            )    / MUo_5r;

                tensor<1> E8Co_5r =
                    2.0 * gemmdiag(dTPom_5r, gCdPmo_5r)
                    * gemmdiag(dTPom_5r, tdPmo_5r)
                    / (MUo_5r * MUo_5r);

                tensor<1> E8Xo_5r =
                    -1.0 * gemmdiag(dTPom_5r, gXdPmo_5r)
                    * gemmdiag(dTPom_5r, tdPmo_5r)
                    / (MUo_5r * MUo_5r);

                tensor<1> EC_5r = E4Co_5r + E8Co_5r;
                tensor<1> EX_5r = E4Xo_5r + E8Xo_5r;

                tensor<1> Eo_5r = EC_5r + EX_5r;

                idx_5r =
                    pivot_ene(Eo_5r,
                            selected_points_5r,
                            pvt_5r);

                double ECs_5r = EC_5r[idx_5r];
                double EXs_5r = EX_5r[idx_5r];

                total_EC_5r += ECs_5r;
                total_EX_5r += EXs_5r;
                total_E_5r += Eo_5r[idx_5r];

                printf("selected point = %d\n", idx_5r);

                printf("selected energy: EC = %.10e EX = %.10e E = %.10e \n",
                        ECs_5r,
                        EXs_5r,
                        Eo_5r[idx_5r]);

                YPmp_5r[all][i] = Y_5r[all][idx_5r];

                auto YPTYS_5r =
                    gemv(YPmp_5r[all][range(i)].T(),
                            YPmp_5r[all][i]);

                chaotrsv('L',
                        LPpp_5r[range(i)][range(i)],
                        YPTYS_5r);

                LPpp_5r[i][range(i)] = YPTYS_5r;

                LPpp_5r[i][i] =
                    sqrt(
                            S_5r[idx_5r][idx_5r] -
                            dot(LPpp_5r[i][range(i)],
                                LPpp_5r[i][range(i)])
                        );

                WPmp_5r[all][i] =
                    (YPmp_5r[all][i] -
                     gemv(WPmp_5r[all][range(i)],
                         LPpp_5r[i][range(i)]))
                    / LPpp_5r[i][i];

                tensor<1> YTW_5r =
                    gemv(YT_5r, WPmp_5r[all][i]);

                ger(1.0,
                        YTW_5r,
                        WPmp_5r[all][i],
                        1.0,
                        dTPom_5r);

                gCWPmp_5r[all][i] =
                    (gCY_5r[all][idx_5r] -
                     gemv(gCWPmp_5r[all][range(i)],
                         LPpp_5r[i][range(i)]))
                    / LPpp_5r[i][i];

                gXWPmp_5r[all][i] =
                    (gXY_5r[all][idx_5r] -
                     gemv(gXWPmp_5r[all][range(i)],
                         LPpp_5r[i][range(i)]))
                    / LPpp_5r[i][i];

                tWPmp_5r[all][i] =
                    (tY_5r[all][idx_5r] -
                     gemv(tWPmp_5r[all][range(i)],
                         LPpp_5r[i][range(i)]))
                    / LPpp_5r[i][i];

                tTWPmp_5r[all][i] =
                    (tTY_5r[all][idx_5r] -
                     gemv(tTWPmp_5r[all][range(i)],
                         LPpp_5r[i][range(i)]))
                    / LPpp_5r[i][i];

                ger(1.0,
                        gCWPmp_5r[all][i],
                        YTW_5r,
                        1.0,
                        gCdPmo_5r);

                ger(1.0,
                        gXWPmp_5r[all][i],
                        YTW_5r,
                        1.0,
                        gXdPmo_5r);

                ger(1.0,
                        tWPmp_5r[all][i],
                        YTW_5r,
                        1.0,
                        tdPmo_5r);

            }
        }

        std::vector<int> pvt_5b;
        for (int idx : pvt_5r)
            pvt_5b.push_back(pvt[idx]);

        bool same_5b = true;
        for (int k = 0; k < n_5b; k++)
            if (pvt_5b[k] != pvt[k]) same_5b = false;
        printf("5b re-pivot: %d singles selected, same order as stage 1: %s\n",
                n_5b, same_5b ? "yes" : "no");

        // ===== Branch b, Step 6b: pair energies given 5b singles =====

        int ntot_6b = n_5b + npairs;
        tensor<2> Y_6b{nvo, ntot_6b};
        int num_grid_keep_6b = n_5b + 1;
        for (int k = 0; k < n_5b; k++)
            Y_6b[all][k] = Y_sp[all][pvt_5b[k]];
        Y_6b[all][range(n_5b, ntot_6b)] = make_Y2(xpa, xpi, pairs, jobinfo);

        tensor<1> EC_pair_6b{npairs};
        tensor<1> EX_pair_6b{npairs};

        tensor<2> YT_6b = Y_6b.T();
        tensor<2> S_6b  = gemm(YT_6b, Y_6b);

        tensor<2> YPmp_6b{nvo, ntot_6b};
        tensor<2> LPpp_6b{ntot_6b, ntot_6b};
        tensor<2> WPmp_6b{nvo, ntot_6b};

        tensor<2> gCWPmp_6b{nvo, ntot_6b};
        tensor<2> gXWPmp_6b{nvo, ntot_6b};
        tensor<2> tTWPmp_6b{nvo, ntot_6b};
        tensor<2> tWPmp_6b{nvo, ntot_6b};

        auto gCY_6b = gemm(VCmm, Y_6b);
        auto gXY_6b = gemm(VXmm, Y_6b);
        auto tY_6b  = gemm(Tmm, Y_6b);
        auto tTY_6b = gemm(Tmm.T(), Y_6b);

        tensor<2> dTPom_6b = -YT_6b;
        tensor<2> gCdPmo_6b = -gCY_6b;
        tensor<2> gXdPmo_6b = -gXY_6b;
        tensor<2> tdPmo_6b = -tY_6b;

        std::set<int> selected_points_6b {};
        std::vector<int> pvt_6b;
        double total_EC_6b = 0.0;
        double total_EX_6b = 0.0;
        double total_E_6b = 0.0;

        double cond_6b = 1.0;

        triangular_condition_number_estimator cond_num_esti_6b(ntot_6b);

        for (int i = 0; i < num_grid_keep_6b; i++)
        {
            int idx_6b;

            printf("\n6b pivot step %d\n", i);

            if (i == 0)
            {
                tensor<1> MUo_6b = gemmdiag(YT_6b, Y_6b);
                tensor<1> E8Co_6b =
                    2.0 * gemmdiag(dTPom_6b, gCdPmo_6b)
                    * gemmdiag(dTPom_6b, tdPmo_6b)
                    / (MUo_6b * MUo_6b);

                tensor<1> E8Xo_6b =
                    -1.0 * gemmdiag(dTPom_6b, gXdPmo_6b)
                    * gemmdiag(dTPom_6b, tdPmo_6b)
                    / (MUo_6b * MUo_6b);
                tensor<1> Eo_6b = E8Co_6b + E8Xo_6b;

                if (i < n_5b)
                {
                    idx_6b = i;
                    selected_points_6b.insert(i);
                    pvt_6b.push_back(i);
                }
                else
                    idx_6b = pivot_ene(Eo_6b, selected_points_6b, pvt_6b);

                double ECs_6b = E8Co_6b[idx_6b];
                double EXs_6b = E8Xo_6b[idx_6b];
                double E_6b = Eo_6b[idx_6b];

                total_EC_6b += ECs_6b;
                total_EX_6b += EXs_6b;
                total_E_6b += E_6b;

                printf("selected point = %d\n", idx_6b);
                printf("selected energy: EC = %.10e EX = %.10e E = %.10e \n",
                        ECs_6b,
                        EXs_6b,
                        Eo_6b[idx_6b]);

                YPmp_6b[all][i] = Y_6b[all][idx_6b];
                LPpp_6b[i][i] = sqrt(S_6b[idx_6b][idx_6b]);

                cond_6b =
                    cond_num_esti_6b.update(LPpp_6b[i][range(i + 1)]);

                WPmp_6b[all][i] =
                    YPmp_6b[all][i] * (1.0 / LPpp_6b[i][i]);

                tensor<1> YTW_6b =
                    gemv(YT_6b, WPmp_6b[all][i]);

                ger(1.0,
                        YTW_6b,
                        WPmp_6b[all][i],
                        1.0,
                        dTPom_6b);

                gCWPmp_6b[all][i] =
                    gCY_6b[all][idx_6b] / LPpp_6b[i][i];

                gXWPmp_6b[all][i] =
                    gXY_6b[all][idx_6b] / LPpp_6b[i][i];

                tWPmp_6b[all][i] =
                    tY_6b[all][idx_6b] / LPpp_6b[i][i];

                tTWPmp_6b[all][i] =
                    tTY_6b[all][idx_6b] / LPpp_6b[i][i];

                ger(1.0,
                        gCWPmp_6b[all][i],
                        YTW_6b,
                        1.0,
                        gCdPmo_6b);

                ger(1.0,
                        gXWPmp_6b[all][i],
                        YTW_6b,
                        1.0,
                        gXdPmo_6b);

                ger(1.0,
                        tWPmp_6b[all][i],
                        YTW_6b,
                        1.0,
                        tdPmo_6b);
            }
            else
            {
                tensor<2> WP_6b = WPmp_6b[all][range(i)];
                tensor<2> Dmo_6b = Y_6b;

                gemm3(-1.0,
                        WP_6b,
                        WP_6b.T(),
                        Y_6b,
                        1.0,
                        Dmo_6b);

                tensor<1> MUo_6b = gemmdiag(YT_6b, Dmo_6b);

                tensor<1> E4Co_6b =
                    4.0 * gemmdiag(
                            gemm(dTPom_6b, gCWPmp_6b[all][range(i)]),
                            gemm(dTPom_6b, tTWPmp_6b[all][range(i)]).T()
                            )    / MUo_6b;

                tensor<1> E4Xo_6b =
                    -2.0 * gemmdiag(
                            gemm(dTPom_6b, gXWPmp_6b[all][range(i)]),
                            gemm(dTPom_6b, tTWPmp_6b[all][range(i)]).T()
                            )    / MUo_6b;

                tensor<1> E8Co_6b =
                    2.0 * gemmdiag(dTPom_6b, gCdPmo_6b)
                    * gemmdiag(dTPom_6b, tdPmo_6b)
                    / (MUo_6b * MUo_6b);

                tensor<1> E8Xo_6b =
                    -1.0 * gemmdiag(dTPom_6b, gXdPmo_6b)
                    * gemmdiag(dTPom_6b, tdPmo_6b)
                    / (MUo_6b * MUo_6b);

                tensor<1> EC_6b = E4Co_6b + E8Co_6b;
                tensor<1> EX_6b = E4Xo_6b + E8Xo_6b;

                tensor<1> Eo_6b = EC_6b + EX_6b;

                // All 5b singles are selected: record each pair's energy given them, then stop
                //Force select the 5b singles in order; pivot on energy only after them

                if (i == n_5b)
                {
                    int n_skipped = 0;
                    for (int k = 0; k < npairs; k++)
                    {
                        EC_pair_6b[k] = EC_6b[n_5b + k];
                        EX_pair_6b[k] = EX_6b[n_5b + k];

                        //guard: pair column almos fully spanned by forced singles
                        if (MUo_6b[n_5b + k] < 1e-10)
                        {
                            EC_pair_6b[k] = 0.0;
                            EX_pair_6b[k] = 0.0;
                            n_skipped++;
                        }
                    }
                    printf("pairs skipped (MU < 1e-10) = %d of %d\n", n_skipped, npairs);
                    break;
                }

                // All 5b singles are selected: record each pair's energy given them, then stop
                //Force select the 5b singles in order; pivot on energy only after them

                if (i < n_5b)
                {
                    idx_6b = i;
                    selected_points_6b.insert(i);
                    pvt_6b.push_back(i);
                }
                else
                    idx_6b = pivot_ene(Eo_6b, selected_points_6b, pvt_6b);

                double ECs_6b = EC_6b[idx_6b];
                double EXs_6b = EX_6b[idx_6b];

                total_EC_6b += ECs_6b;
                total_EX_6b += EXs_6b;
                total_E_6b += Eo_6b[idx_6b];

                printf("selected point = %d\n", idx_6b);

                printf("selected energy: EC = %.10e EX = %.10e E = %.10e \n",
                        ECs_6b,
                        EXs_6b,
                        Eo_6b[idx_6b]);

                YPmp_6b[all][i] = Y_6b[all][idx_6b];

                auto YPTYS_6b =
                    gemv(YPmp_6b[all][range(i)].T(),
                            YPmp_6b[all][i]);

                chaotrsv('L',
                        LPpp_6b[range(i)][range(i)],
                        YPTYS_6b);

                LPpp_6b[i][range(i)] = YPTYS_6b;

                LPpp_6b[i][i] =
                    sqrt(
                            S_6b[idx_6b][idx_6b] -
                            dot(LPpp_6b[i][range(i)],
                                LPpp_6b[i][range(i)])
                        );

                WPmp_6b[all][i] =
                    (YPmp_6b[all][i] -
                     gemv(WPmp_6b[all][range(i)],
                         LPpp_6b[i][range(i)]))
                    / LPpp_6b[i][i];

                tensor<1> YTW_6b =
                    gemv(YT_6b, WPmp_6b[all][i]);

                ger(1.0,
                        YTW_6b,
                        WPmp_6b[all][i],
                        1.0,
                        dTPom_6b);

                gCWPmp_6b[all][i] =
                    (gCY_6b[all][idx_6b] -
                     gemv(gCWPmp_6b[all][range(i)],
                         LPpp_6b[i][range(i)]))
                    / LPpp_6b[i][i];

                gXWPmp_6b[all][i] =
                    (gXY_6b[all][idx_6b] -
                     gemv(gXWPmp_6b[all][range(i)],
                         LPpp_6b[i][range(i)]))
                    / LPpp_6b[i][i];

                tWPmp_6b[all][i] =
                    (tY_6b[all][idx_6b] -
                     gemv(tWPmp_6b[all][range(i)],
                         LPpp_6b[i][range(i)]))
                    / LPpp_6b[i][i];

                tTWPmp_6b[all][i] =
                    (tTY_6b[all][idx_6b] -
                     gemv(tTWPmp_6b[all][range(i)],
                         LPpp_6b[i][range(i)]))
                    / LPpp_6b[i][i];

                ger(1.0,
                        gCWPmp_6b[all][i],
                        YTW_6b,
                        1.0,
                        gCdPmo_6b);

                ger(1.0,
                        gXWPmp_6b[all][i],
                        YTW_6b,
                        1.0,
                        gXdPmo_6b);

                ger(1.0,
                        tWPmp_6b[all][i],
                        YTW_6b,
                        1.0,
                        tdPmo_6b);

            }
        }
        //Step 6b: rank pairs by energy with 5b singles as starting points
        tensor<1> E_pair_6b = EC_pair_6b + EX_pair_6b;

        std::vector<int> pair_order_6b(npairs);
        for (int k = 0; k < npairs; k++) pair_order_6b[k] = k;

        std::sort(pair_order_6b.begin(), pair_order_6b.end(),
                [&](int a, int b)
                {
                if (jobinfo.rank_energy == "exchange")
                return std::abs(EX_pair_6b[a]) > std::abs(EX_pair_6b[b]);
                return std::abs(E_pair_6b[a]) > std::abs(E_pair_6b[b]);
                });

        int n_top_keep_6b = std::min(jobinfo.n_top, npairs);
        pair_list top_pairs_6b;
        for (int r = 0; r < n_top_keep_6b; r++)
            top_pairs_6b.push_back(pairs[pair_order_6b[r]]);

        printf("\n -- 6b top pair selection --\n");
        printf("ranking criterion = %s\n", jobinfo.rank_energy.c_str());
        printf("top ranked pairs selected = %zu\n", top_pairs_6b.size());

        for (int r = 0; r < std::min(5, n_top_keep_6b); r++)
        {
            int k = pair_order_6b[r];
            printf("Rank %d : Pair (%d, %d) EC = %.10e EX = %.10e E = %.10e \n",
                    r + 1, pairs[k].first, pairs[k].second,
                    EC_pair_6b[k], EX_pair_6b[k], E_pair_6b[k]);
        }

        printf("6b Total Energy (%d forced singles) = %.10f\n", n_5b, total_E_6b);

        // ===== Branch b, Step 7b: pivot top pairs after 5b singles =====

        int npairs_7b = static_cast<int>(top_pairs_6b.size());
        int ntot_7b = n_5b + npairs_7b;
        tensor<2> Y_7b{nvo, ntot_7b};
        for (int k = 0; k < n_5b; k++)
            Y_7b[all][k] = Y_sp[all][pvt_5b[k]];
        Y_7b[all][range(n_5b, ntot_7b)] = make_Y2(xpa, xpi, top_pairs_6b, jobinfo);

        int num_grid_keep_7b =
            std::min(static_cast<int>(jobinfo.chi3b * norb), ntot_7b);
        tensor<2> YT_7b = Y_7b.T();
        tensor<2> S_7b  = gemm(YT_7b, Y_7b);

        tensor<2> YPmp_7b{nvo, ntot_7b};
        tensor<2> LPpp_7b{ntot_7b, ntot_7b};
        tensor<2> WPmp_7b{nvo, ntot_7b};

        tensor<2> gCWPmp_7b{nvo, ntot_7b};
        tensor<2> gXWPmp_7b{nvo, ntot_7b};
        tensor<2> tTWPmp_7b{nvo, ntot_7b};
        tensor<2> tWPmp_7b{nvo, ntot_7b};

        auto gCY_7b = gemm(VCmm, Y_7b);
        auto gXY_7b = gemm(VXmm, Y_7b);
        auto tY_7b  = gemm(Tmm, Y_7b);
        auto tTY_7b = gemm(Tmm.T(), Y_7b);

        tensor<2> dTPom_7b = -YT_7b;
        tensor<2> gCdPmo_7b = -gCY_7b;
        tensor<2> gXdPmo_7b = -gXY_7b;
        tensor<2> tdPmo_7b = -tY_7b;

        std::set<int> selected_points_7b {};
        std::vector<int> pvt_7b;
        double total_EC_7b = 0.0;
        double total_EX_7b = 0.0;
        double total_E_7b = 0.0;

        double cond_7b = 1.0;

        triangular_condition_number_estimator cond_num_esti_7b(ntot_7b);

        for (int i = 0; i < num_grid_keep_7b; i++)
        {
            int idx_7b;

            printf("\n7b pivot step %d\n", i);

            if (i == 0)
            {
                tensor<1> MUo_7b = gemmdiag(YT_7b, Y_7b);
                tensor<1> E8Co_7b =
                    2.0 * gemmdiag(dTPom_7b, gCdPmo_7b)
                    * gemmdiag(dTPom_7b, tdPmo_7b)
                    / (MUo_7b * MUo_7b);

                tensor<1> E8Xo_7b =
                    -1.0 * gemmdiag(dTPom_7b, gXdPmo_7b)
                    * gemmdiag(dTPom_7b, tdPmo_7b)
                    / (MUo_7b * MUo_7b);
                tensor<1> Eo_7b = E8Co_7b + E8Xo_7b;

                if (i < n_5b)
                {
                    idx_7b = i;
                    selected_points_7b.insert(i);
                    pvt_7b.push_back(i);
                }
                else
                    idx_7b = pivot_ene(Eo_7b, selected_points_7b, pvt_7b);

                double ECs_7b = E8Co_7b[idx_7b];
                double EXs_7b = E8Xo_7b[idx_7b];
                double E_7b = Eo_7b[idx_7b];

                total_EC_7b += ECs_7b;
                total_EX_7b += EXs_7b;
                total_E_7b += E_7b;

                printf("selected point = %d\n", idx_7b);
                printf("selected energy: EC = %.10e EX = %.10e E = %.10e \n",
                        ECs_7b,
                        EXs_7b,
                        Eo_7b[idx_7b]);

                YPmp_7b[all][i] = Y_7b[all][idx_7b];
                LPpp_7b[i][i] = sqrt(S_7b[idx_7b][idx_7b]);

                cond_7b =
                    cond_num_esti_7b.update(LPpp_7b[i][range(i + 1)]);

                WPmp_7b[all][i] =
                    YPmp_7b[all][i] * (1.0 / LPpp_7b[i][i]);

                tensor<1> YTW_7b =
                    gemv(YT_7b, WPmp_7b[all][i]);

                ger(1.0,
                        YTW_7b,
                        WPmp_7b[all][i],
                        1.0,
                        dTPom_7b);

                gCWPmp_7b[all][i] =
                    gCY_7b[all][idx_7b] / LPpp_7b[i][i];

                gXWPmp_7b[all][i] =
                    gXY_7b[all][idx_7b] / LPpp_7b[i][i];

                tWPmp_7b[all][i] =
                    tY_7b[all][idx_7b] / LPpp_7b[i][i];

                tTWPmp_7b[all][i] =
                    tTY_7b[all][idx_7b] / LPpp_7b[i][i];

                ger(1.0,
                        gCWPmp_7b[all][i],
                        YTW_7b,
                        1.0,
                        gCdPmo_7b);

                ger(1.0,
                        gXWPmp_7b[all][i],
                        YTW_7b,
                        1.0,
                        gXdPmo_7b);

                ger(1.0,
                        tWPmp_7b[all][i],
                        YTW_7b,
                        1.0,
                        tdPmo_7b);
                { int ngrid_7b = i + 1; output_to_csv(jobinfo, idx_7b, ngrid_7b, norb, E_exact_c, E_exact_x, ECs_7b, EXs_7b, total_EC_7b, total_EX_7b, LPpp_7b[i][i], cond_7b, "_7b"); }
            }
            else
            {
                tensor<2> WP_7b = WPmp_7b[all][range(i)];
                tensor<2> Dmo_7b = Y_7b;

                gemm3(-1.0,
                        WP_7b,
                        WP_7b.T(),
                        Y_7b,
                        1.0,
                        Dmo_7b);

                tensor<1> MUo_7b = gemmdiag(YT_7b, Dmo_7b);

                tensor<1> E4Co_7b =
                    4.0 * gemmdiag(
                            gemm(dTPom_7b, gCWPmp_7b[all][range(i)]),
                            gemm(dTPom_7b, tTWPmp_7b[all][range(i)]).T()
                            )    / MUo_7b;

                tensor<1> E4Xo_7b =
                    -2.0 * gemmdiag(
                            gemm(dTPom_7b, gXWPmp_7b[all][range(i)]),
                            gemm(dTPom_7b, tTWPmp_7b[all][range(i)]).T()
                            )    / MUo_7b;

                tensor<1> E8Co_7b =
                    2.0 * gemmdiag(dTPom_7b, gCdPmo_7b)
                    * gemmdiag(dTPom_7b, tdPmo_7b)
                    / (MUo_7b * MUo_7b);

                tensor<1> E8Xo_7b =
                    -1.0 * gemmdiag(dTPom_7b, gXdPmo_7b)
                    * gemmdiag(dTPom_7b, tdPmo_7b)
                    / (MUo_7b * MUo_7b);

                tensor<1> EC_7b = E4Co_7b + E8Co_7b;
                tensor<1> EX_7b = E4Xo_7b + E8Xo_7b;

                tensor<1> Eo_7b = EC_7b + EX_7b;

                // All 5b singles are selected: record each pair's energy given them, then stop
                //Force select the 5b singles in order; pivot on energy only after them

                if (i < n_5b)
                {
                    idx_7b = i;
                    selected_points_7b.insert(i);
                    pvt_7b.push_back(i);
                }
                else
                    idx_7b = pivot_ene(Eo_7b, selected_points_7b, pvt_7b);

                double ECs_7b = EC_7b[idx_7b];
                double EXs_7b = EX_7b[idx_7b];

                total_EC_7b += ECs_7b;
                total_EX_7b += EXs_7b;
                total_E_7b += Eo_7b[idx_7b];

                printf("selected point = %d\n", idx_7b);

                printf("selected energy: EC = %.10e EX = %.10e E = %.10e \n",
                        ECs_7b,
                        EXs_7b,
                        Eo_7b[idx_7b]);

                YPmp_7b[all][i] = Y_7b[all][idx_7b];

                auto YPTYS_7b =
                    gemv(YPmp_7b[all][range(i)].T(),
                            YPmp_7b[all][i]);

                chaotrsv('L',
                        LPpp_7b[range(i)][range(i)],
                        YPTYS_7b);

                LPpp_7b[i][range(i)] = YPTYS_7b;

                LPpp_7b[i][i] =
                    sqrt(
                            S_7b[idx_7b][idx_7b] -
                            dot(LPpp_7b[i][range(i)],
                                LPpp_7b[i][range(i)])
                        );

                WPmp_7b[all][i] =
                    (YPmp_7b[all][i] -
                     gemv(WPmp_7b[all][range(i)],
                         LPpp_7b[i][range(i)]))
                    / LPpp_7b[i][i];

                tensor<1> YTW_7b =
                    gemv(YT_7b, WPmp_7b[all][i]);

                ger(1.0,
                        YTW_7b,
                        WPmp_7b[all][i],
                        1.0,
                        dTPom_7b);

                gCWPmp_7b[all][i] =
                    (gCY_7b[all][idx_7b] -
                     gemv(gCWPmp_7b[all][range(i)],
                         LPpp_7b[i][range(i)]))
                    / LPpp_7b[i][i];

                gXWPmp_7b[all][i] =
                    (gXY_7b[all][idx_7b] -
                     gemv(gXWPmp_7b[all][range(i)],
                         LPpp_7b[i][range(i)]))
                    / LPpp_7b[i][i];

                tWPmp_7b[all][i] =
                    (tY_7b[all][idx_7b] -
                     gemv(tWPmp_7b[all][range(i)],
                         LPpp_7b[i][range(i)]))
                    / LPpp_7b[i][i];

                tTWPmp_7b[all][i] =
                    (tTY_7b[all][idx_7b] -
                     gemv(tTWPmp_7b[all][range(i)],
                         LPpp_7b[i][range(i)]))
                    / LPpp_7b[i][i];

                ger(1.0,
                        gCWPmp_7b[all][i],
                        YTW_7b,
                        1.0,
                        gCdPmo_7b);

                ger(1.0,
                        gXWPmp_7b[all][i],
                        YTW_7b,
                        1.0,
                        gXdPmo_7b);

                ger(1.0,
                        tWPmp_7b[all][i],
                        YTW_7b,
                        1.0,
                        tdPmo_7b);
                { int ngrid_7b = i + 1; output_to_csv(jobinfo, idx_7b, ngrid_7b, norb, E_exact_c, E_exact_x, ECs_7b, EXs_7b, total_EC_7b, total_EX_7b, LPpp_7b[i][i], cond_7b, "_7b"); }

            }
        }

        printf("\n7b energy pivot complete\n");
        printf("\n7b selected points = %zu (target %d)\n", pvt_7b.size(), num_grid_keep_7b);

        int npicked_pairs_7b = 0;
        for (int idx : pvt_7b)
            if (idx >= n_5b) npicked_pairs_7b++;
        printf("7b singles = %d, pairs = %d\n", n_5b, npicked_pairs_7b);

        printf("7b EC = %.10f EX = %.10f E = %.10f\n",
                total_EC_7b, total_EX_7b, total_E_7b);

        printf("7b Ec error = %.4f%% Ex error = %.4f%%\n",
                100.0 * std::abs((total_EC_7b - E_exact_c) / E_exact_c),
                100.0 * std::abs((total_EX_7b - E_exact_x) / E_exact_x));
    }

    printf("Stage-1 Total Energy (%zu singles) = %.10f\n", pvt.size(), total_E);

    timer::print_timers();
}
