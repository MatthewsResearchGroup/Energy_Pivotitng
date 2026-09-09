#include "energyAnalysis.hpp"
#include "marray.hpp"
#include "input.hpp"
#include "jobinfo.hpp"
#include "io.hpp"
#include "tensor_ops.hpp"
#include "pair_points.hpp"
// Change #1 : add one include
#include "make_distance_pairs.hpp"

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
#include <vector>
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
                                                 "Energy Analsyis 1.0");

    //Parse the input and setup the jobinfo struct
    Jobinfo jobinfo;
    if (parse_input(args,jobinfo) == 0) 
    {
        printf("Bad input to energyAnalysis\n");
        exit(1);
    }

    //read the dimensions we need
    // no 	     -> number of occupied orbitals
    // nv	     -> number of virtual orbitals
    // nps	     -> number of TOTAL grid points
    // num_pairs     -> number of pairs of subset of gridpoints to keep
    int nv, no, nps;
    read_dimensions(nv,  jobinfo.path_to_fa,
                    no,  jobinfo.path_to_fi,
                    nps, jobinfo.path_to_xa);
                    // psvec, jobinfo.path_to_points_set);
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
    // printf("checking the symmetry of VCmm...\n");
    sym_check(VCmm, threshold);
    // printf("checking the symmetry of VXmm...\n");
    sym_check(VXmm, threshold);



    //Generate T
    // if method is MP2, we will generate amplitude from V.
    // Otherwise, we will read amplitude from files.
    //
    tensor<4> Taibj{nv, no, nv, no};
    if (jobinfo.method == "MP2")
    {
        // printf("Calculation method is MP2, making T from V...\n");
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
        // printf("Calculation method is MP3 or CCSD. Reading t from file: %s\n", jobinfo.path_to_t.c_str());
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

// Change # 2: after reading xpa/xpi, add pair-candidate setup
    
    // PAIR-POINT EXTENSION: distance filtered candidate pairs + Y2
    // NEW file/fields needed: jobinfo.path_to_coords, pair_dist_min, pair_dist_max
    auto coords = tensor_from_file(jobinfo.path_to_coords, sizeof(Int), nps, 3);
    pair_list pairs = make_distance_pairs(coords, jobinfo.pair_dist_min, jobinfo.pair_dist_max):
    int npairs = (int) pairs.size();
    printf("npairs (after distance filter [%g, %g]) : %d \n",
        jobinfo.pair_dist_min, jobinfo.pair_dist_max, npairs);

    tensor<2> Y2 = make_Y2(xpa, xpi, pairs, jobinfo); // nvo *  npairs
    int ntot = nps + npairs;                            
    

/*
    printf("FAA tensor is...\n"); jht_print(faa); 
    printf("FII tensor is...\n"); jht_print(fii); 
    printf("FA vector is...\n"); jht_print(FA);
    printf("FI vector is...\n"); jht_print(FI);
    printf("XA tensor is...\n"); jht_print(xpa); 
    printf("XI tensor is...\n"); jht_print(xpi); 
    printf("Vaibj matirx in C++ is...\n");jht_print(v.lowered(2));
    printf("T2 matrix is...\n"); jht_print(Tmm);
*/

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
// Change #3: merge single + pair points into Y/YT/S
    
    tensor<2> YT_sp = krp(xpa[all][all], xpi[all][all]).lowered(1); // nps * nvo
    tensor<2> Y_sp  = YT_sp.T();  
    tensor<2> Y{nvo, ntot};
    Y[all][range(nps)] = Y_single;

    if (npairs > 0) Y[all][range(nps, ntot)] = Y2;

    tensor<2> YT =Y.T(); // nvo * nps
    tensor<2> S = gemm(YT, Y); // nps, nps

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
// Change #4: resize the per-iteration arrays (nps → ntot)
    tensor<2> YPmp{nvo, ntot};   
    tensor<2> SPpp{ntot, ntot};
    tensor<2> LPpp{ntot, ntot};
    tensor<2> WPmp{nvo, ntot};
    tensor<2> gCWPmp{nvo, ntot};
    tensor<2> gXWPmp{nvo, ntot};
    tensor<2> tTWPmp{nvo, ntot};  
    tensor<2> tWPmp{nvo, ntot};  // In the update of td, we also need tW.

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

// Change #5: num_grid_keep and the condition estimator (nps → ntot)
    
    int num_grid_keep = (norb * 8 < ntot)? norb * 8 : ntot;

    double cond = 1.0;


    // constract a cond number estimator class
    triangular_condition_number_estimator cond_num_esti(ntot);

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

            std::cout << "Ene Piv: EX, EC = " << E8Co[i] << ", " << E8Xo[i]  << std::endl;  
            if (getenv("CHOL"))
                idx = i;
            else
                idx = pivot_ene(Eo, selected_points, pvt);
            
            double ECs = E8Co[idx];
            double EXs = E8Xo[idx];
            total_EC += E8Co[idx];
            total_EX += E8Xo[idx];
            total_E += Eo[idx];

            // output_to_csv(jobinfo, idx, ngrid, norb, E_exact_c, E_exact_x, ECs, EXs, total_EC, total_EX);

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
            double cond = cond_num_esti.update(LPpp[i][range(i+1)]);
            // double cond = 1.0;

            //output_to_csv(jobinfo, idx, ngrid, norb, E_exact_c, E_exact_x, ECs, EXs, total_EC, total_EX, LPpp[i][i]);
            output_to_csv(jobinfo, idx, ngrid, norb, E_exact_c, E_exact_x, ECs, EXs, total_EC, total_EX, LPpp[i][i], cond);
            // for(auto k = 0; k <= i; k++)
            // {
            //     for(auto l = 0; l <= i; l++)
            //     {
            //         printf("%f, ", LPpp[k][l]);
            //     }
            //     printf("\n");
            // }

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
            // printf("********** WPmp **********\n");
            // for(auto k = 0; k < nvo; k++)
            // {
            //     for(auto l = 0; l <= i; l++)
            //     {
            //         printf("%f, ", WPmp[k][l]);
            //     }
            //     printf("\n");
            // }
           

            /************************************************
             *          T    
             *  Update d part
             *   T    T                 T
             *  d += Y \Delta W \Delta W
             * O*M  O*M      M*1      1*M
             *  
             *************************************************/
            // printf("********** after  dT **********\n");
            // for(auto k = 0; k < 5; k++)
            // {
            //     for(auto l = 0; l < 5; l++)
            //     {
            //         printf("%.9f, ", dTPom[k][l]);
            //     }
            //     printf("\n");
            // }

           
            tensor<1> YTW = gemv(YT, WPmp[all][i]);
            ger(1.0, YTW, WPmp[all][i], 1.0, dTPom);            

            // printf("********** after  dT **********\n");
            // for(auto k = 0; k < 5; k++)
            // {
            //     for(auto l = 0; l < 5; l++)
            //     {
            //         printf("%.9f, ", dTPom[k][l]);
            //     }
            //     printf("\n");
            // }


            /******************************************
             * 
             * Verify the correctness of the above code
             *
             * ***************************************/


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

            std::cout << "Ene Piv: EX, EC = " << EC[i] << ", " << EX[i]  << std::endl;  


            tensor<1> Eo = E4Co + E8Co + E4Xo + E8Xo; 

            if (getenv("CHOL"))
                idx = i;
            else
                idx = pivot_ene(Eo, selected_points, pvt);

            double ECs = EC[idx];
            double EXs = EX[idx];
            total_EC += EC[idx];
            total_EX += EX[idx];
            total_E += Eo[idx];

            // output_to_csv(jobinfo, idx, ngrid, norb, E_exact_c, E_exact_x, ECs, EXs, total_EC, total_EX, );

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
            // double cond;
            PROFILE_SECTION("L update")
            auto YPTYS = gemv(YPmp[all][range(i)].T(), YPmp[all][i]); 

           

            chaotrsv('L', LPpp[range(i)][range(i)], YPTYS);

            // printf("LPpp = %d, %d, %d\n",  LPpp.length(0), LPpp.stride(1),  YPTYS.stride());
            // trsv('L', 'N', 'N', i, LPpp.data(), LPpp.stride(1), YPTYS.data(), YPTYS.stride());

            LPpp[i][range(i)] = YPTYS;
            LPpp[i][i] = sqrt(S[idx][idx] - dot(LPpp[i][range(i)], LPpp[i][range(i)]));


            // cond = cond_num_esti.update(LPpp[i][range(i+1)]);
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

                int info = c_dgesvd( jobu, jobvt, m, n, A.data(), lda, s.data(), u.data(),
                                     ldu, vt.data(), ldvt);

                cond = max<double>(s) / min<double>(s);
            }
            // printf("info, max(s), min(s) = %d, %f, %f, \n", info, max<double>(s), min<double>(s));

            PROFILE_STOP

            //output_to_csv(jobinfo, idx, ngrid, norb, E_exact_c, E_exact_x, ECs, EXs, total_EC, total_EX, LPpp[i][i]);
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

            // printf("********** W update ****************\n\n");
            
            tensor<1> WPL_10 = gemv(WPmp[all][range(i)], LPpp[i][range(i)]);
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
             * Verify the correctness of the above code
             *
             * ***************************************/


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
    // Change #6 : clarify which kind of point was picked
        
        if (idx < nps)
            printf("Selected Point : %d (single) \n", idx);
        else
            printf("Selected Point : %d (pair: %d, %d) \n", idx, 
                pairs[idx - nps].first, pairs[idx -nps].second);

    }
    // pvt_to_file(pvt);
    //
    printf("Total Energy = %.10f\n", total_E);

    timer::print_timers();

}
