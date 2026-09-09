/************************************************* 
 * input.hpp 
 *
 * Input parser for energyAnalysis.cxx
 *
 *  Uses the docopt library found at:
 *  https://github.com/docopt/docopt.cpp
*************************************************/

#ifndef _ENERGYANALYSIS_INPUT_HPP_
#define _ENERGYANALYSIS_INPUT_HPP_

#include "energyAnalysis.hpp"
#include "jobinfo.hpp"
#include "docopt.h"

#include <string>
#include <map>
#include <iostream>

/************************************************* 
 *  String for docopt
*************************************************/ 
// Change #1: add new command-line options to the USAGE string
static const char USAGE[] =
R"(EnergyAnalysis.

  Usage:
    energyAnalysis --mol=<mol> --method=<method> --basis=<basis>
    energyAnalysis (-h | --help)
    energyAnalysis --version

  Options:
    -h --help    Show this screen.
     --version   Show version.
     --ftype=<ftype> Pair-point construction type: all, cross, or nocross [default: all].
     --dist-min=<dmin> Minimum pair-point distance cutoff, a.u. [default: 1.0].
     --dist-max=<dmax> Maximum pair-point distance cutoff, a.u. [default: 3.0].

)";


/************************************************* 
 *  Parse and check input 
*************************************************/ 
int parse_input(const std::map<std::string,docopt::value> args, 
                Jobinfo& jobinfo)
{
    for (auto const& arg : args) 
    {

//Turn this on if you want debugging
#if 1
       std::cout << arg.first << ": " <<arg.second << std::endl; 
#endif

       // moleucle options
       if (arg.first == "--mol") {jobinfo.set_molecule(arg.second.asString());}

       // Method options
       if (arg.first == "--method") {jobinfo.set_method(arg.second.asString());}

       // basis options
       if (arg.first == "--basis") {jobinfo.set_basis(arg.second.asString());}

       // Path options
       // if (arg.first == "--grid-path") {jobinfo.set_grid_path(arg.second.asString());}
       // if (arg.first == "--orb-path")  {jobinfo.set_orb_path(arg.second.asString());}
       // if (arg.first == "--points-sets-path") {jobinfo.set_points_set_path(arg.second.asString());}
      
       // Change #2: parse --ftype, --dist-min, --dist-max, and call set_coords_path()
      // PAIR-POINT EXTENSION options
        if (arg.first == "--ftype" && arg.second)
        {
            std::string ft = arg.second.asString();
            if  (ft == "all"  jobinfo.ftype = jobinfo::Ftype::all;
            else if (ft == "cross"  jobinfo.ftype = jobinfo::Ftype::cross;
            else if (ft == "nocross"  jobinfo.ftype = jobinfo::Ftype::nocross;
            else { printf("Bad --ftype value: %s\n", ft.c_str()); exit(1); }
       }
       if (arg.first == "--dist-min" && arg.second) {jobinfo.pair_dist_min = std::stod(arg.second.asString());}
       if (arg.first == "--dist-max" && arg.second) {jobinfo.pair_dist_max = std::stod(arg.second.asString());}
  
       
       //Seed options
       // if (arg.first == "--seed" && arg.second) {jobinfo.seed = (int) arg.second.asLong();}
       // set relative path.
       jobinfo.set_work_path();
       jobinfo.set_orb_path();
       jobinfo.set_grid_path();
       jobinfo.set_coords_path(); //change
       jobinfo.set_t_path();

    }//end loop over arguements

    printf("Job Information\n%s\n",jobinfo.print_str().c_str());

    return jobinfo.validate();
}


#endif
