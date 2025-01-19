/*
 * pair_points.hpp
 *
 * Header file which contains functions related
 * to pairs of gridpoints used in 
 * energyAnslysis.cxx
 *
 * Functions in this file:
 *    make_random_pairs
 *    make_Y2mo
 *    make_Y2_all
 *    make_Y2_cross
 *    make_Y2_nocross
************************************************/
#ifndef _ENERGYANALYSIS_PAIR_POINTS_HPP_
#define _ENERGYANALYSIS_PAIR_POINTS_HPP_

#include "energyAnalysis.hpp"
#include "jobinfo.hpp"
#include "marray.hpp"

#include "randutils.hpp"

#include <vector>
#include <utility>
#include <stdio.h>

/********************************************
 * make_random_pairs 
 *
 * returns a pair_list (defined in energyAnalysis.hpp),
 * of random pairs of points. We do NOT test for 
 * redundencies of any kind: the pair point (1,1) is 
 * included just as reapeats (1,2), (1,2), (2,1) are 
 * included 
********************************************/
pair_list make_random_pairs(const int num_pairs, const int num_points, const int in_seed)
{
    randutils::seed_seq_fe256 seed{in_seed};
    randutils::mt19937_rng rng{seed};

    printf("\nInitializing random pair list with seed : %d\n",in_seed);

    pair_list pairs;
    pairs.reserve(num_pairs);

    // pairs.emplace_back(2,11);
    // pairs.emplace_back(5,9);
    // pairs.emplace_back(14,3);
    for (auto pair : range(num_pairs)) 
    {
        auto i = rng.uniform(0,num_points-1), j = rng.uniform(0,num_points-1);
        // auto i = 2, j = rng.uniform(0,num_points-1);
        while (i == j)
        {
            i = rng.uniform(0,num_points-1);
            j = rng.uniform(0,num_points-1);   
        }
        pairs.emplace_back(i, j);
        // pairs.emplace_back(rng.uniform(0,num_points-1),rng.uniform(0,num_points-1));   
    }

    return pairs;
}

/********************************************
 *  make_Y2_all
 *
 *  Generate the Y2 matrix of pair-points from 
 *  the xpi and xpa matrices. This includes both 
 *  cross and diagonal terms.
********************************************/
tensor<2> make_Y2_all(const tensor<2>& xpa,
                      const tensor<2>& xpi,
                      const pair_list& pairs)
{                     
    const auto num_pairs = pairs.size();
    const auto nv = xpa.length(1);
    const auto no = xpi.length(1);
    
    tensor<2> xxa{{num_pairs, nv},COLUMN_MAJOR};
    tensor<2> xxi{{num_pairs, no},COLUMN_MAJOR};
    
    for (auto pair : range(num_pairs)) {
        auto [p, q] = pairs[pair];
        xxa[pair][all] = xpa[p][all]
                       + xpa[q][all];
        xxi[pair][all] = xpi[p][all]
                       + xpi[q][all];
    }

    tensor<2> Y2mo = krp(xxa, xxi).lowered(1).T();
    return Y2mo;
}   

/********************************************
 *  make_Y2_cross
 *
 *  Generate the Y2 matrix of pair-points from 
 *  the xpi and xpa matrices. This includes only the 
 *  cross-product terms 
 *
 *  This is not super efficient code, and could be improved if need be
********************************************/
tensor<2> make_Y2_cross(const tensor<2>& xpa,
                        const tensor<2>& xpi,
                        const pair_list& pairs)
{   
    const auto num_pairs = pairs.size();
    const auto nv = xpa.length(1);
    const auto no = xpi.length(1); 
    
    tensor<2> Y2mo{{nv*no, num_pairs}, COLUMN_MAJOR};

    #pragma omp parallel
    for (auto pair : range(num_pairs)) 
    {
        auto ai = 0;
        const auto p1 = pairs[pair].first;
        const auto p2 = pairs[pair].second;

        #pragma omp for
        for (auto i : range(no))
        for (auto a : range(nv))
        {
            Y2mo[ai][pair] = xpa[p1][a]*xpi[p2][i] 
                           + xpa[p2][a]*xpi[p1][i]; 
            ai++;
        }
    } 

    return Y2mo;
}

/********************************************
 *  make_Y2_nocross
 *
 *  Generate the Y2 matrix of pair-points from 
 *  the xpi and xpa matrices. This includes only the 
 *  non-cross-product terms 
 *
 *  This is not super efficient code, and could be improved if need be
********************************************/
tensor<2> make_Y2_nocross(const tensor<2>& xpa,
                          const tensor<2>& xpi,
                          const pair_list& pairs)
{                         
    const auto num_pairs = pairs.size();
    const auto nv = xpa.length(1);
    const auto no = xpi.length(1); 
    
    tensor<2> Y2mo{{nv*no,num_pairs},COLUMN_MAJOR};
    
    #pragma omp parallel
    for (auto pair : range(num_pairs)) 
    {
        auto ai = 0;
        const auto p1 = pairs[pair].first;
        const auto p2 = pairs[pair].second;

        #pragma omp for
        for (auto i : range(no))
        for (auto a : range(nv))
        {
            Y2mo[ai][pair] = xpa[p1][a]*xpi[p1][i] 
                           + xpa[p2][a]*xpi[p2][i]; 
            ai++;
        }
    } 

    return Y2mo;
}   

/********************************************
 * make_Y2mo
 *
 * Construct the Y2 matrix for the pairs of 
 * gridpoints, with the appropriate sub-function
 * controlled by the ftype arguement
********************************************/
tensor<2> make_Y2(const tensor<2>& xpa,
                  const tensor<2>& xpi,
                  const pair_list& pairs,
                  const Jobinfo&   jobinfo)
{                         
    switch(jobinfo.ftype)
    {
        case Jobinfo::Ftypes::all     : return make_Y2_all(xpa, xpi, pairs); 
        case Jobinfo::Ftypes::cross   : return make_Y2_cross(xpa, xpi, pairs); 
        case Jobinfo::Ftypes::nocross : return make_Y2_nocross(xpa, xpi, pairs); 
/*
        if (ftype == 0) {return make_Y2_all(xpa, xpi, pairs);}
        if (ftype == 1) {return make_Y2_cross(xpa, xpi, pairs);}
        if (ftype == 2) {return make_Y2_nocross(xpa, xpi, pairs);}
*/
    }
    
    printf("Bad ftype input into make_Y2 \n"); 
    exit(1);
}   



#endif
