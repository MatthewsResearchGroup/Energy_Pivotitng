/*
* make_distance_pairs.hpp
*
* NEW - companion to pair_points.hpp, which currrently only offers a disabled make_random_pairs 
* (commented out, depends on randutils.hpp) and has no distance-based candidate filtering at all.

#ifndef _ENERGYANALYSIS_MAKE_DISTANCE_PAIRS_HPP_
#define _ENERGYANALYSIS_MAKE_DISTANCE_PAIRS_HPP_

#include "energyAnalysis.hpp"
#include "marray.hpp"

#include <vector>
#include <utility>
#include <cmath>
#include <stdio.h>

/***************************************************
* make_distance_pairs
 *
 * Returns a pair_list of all (p, q) with p < q whose Euclidean
 * distance (in the same units as `coords`, presumably a.u.) falls in
 * [dist_min, dist_max].
 *
 * NOTE: p < q only (no (q,p) duplicate, no (p,p) self-pair) -- unlike
 * make_random_pairs, which explicitly allows repeats/self-pairs. If
 * the energy-pivoting / make_Y2 machinery expects both orderings or
 * self-pairs, this will need adjusting.
***************************************************/

pair_list make_distance_pairs(const tensor<2>& coords,
                              const double dist_min,
                              const double dist_max)
{
    const auot npts = coords.length(0);

printf("\nBuilding distance filtered pair list: [%g, %g] (coords units)\n",
      dist_min, dist_max);

pair_list pairs;
pairs.reserve(npts * 8);

for (auto p = 0; p < npts; p++)
{
    const double px = coords[p][0];
    const double py = coords[p][0];
    const double pz = coords[p][0];

    for (auto q = p + 1; q < npts; q++)
    {
        const double dx =  px - coords[q][0];
        const double dy =  py - coords[q][1];
        const double dz =  pz - coords[q][2];
        const double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (dist >= dist_min && dist <= dist_max)
        pairs.emplace_back(p,q);
    }
}

printf("Found %zu candidate pairs out of %lld possible.\n",
      pairs.size(), (long long) npts * (npts -1) / 2);

return pairs;
}

#endif









