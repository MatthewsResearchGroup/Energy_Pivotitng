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
static const char USAGE[] =
R"(EnergyAnalysis.

  Usage:
    energyAnalysis --method=<method>  --npair=<npair> [--seed=<seed>] 
    energyAnalysis (-h | --help)
    energyAnalysis --version

  Options:
    -h --help    Show this screen.
     --version   Show version.
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

       /* Ftype options */
       // if (arg.first == "--ftype") 
       // {
       //     if (arg.second.asString() == "all") {jobinfo.ftype = Jobinfo::Ftypes::all;}
       //     else if (arg.second.asString() == "cross") {jobinfo.ftype = Jobinfo::Ftypes::cross;}
       //     else if (arg.second.asString() == "nocross") {jobinfo.ftype = Jobinfo::Ftypes::nocross;}
       //     else {printf("Bad ftype option.\n"); return 1;}
       // }
       
       // Method options
       if (arg.first == "--method") {jobinfo.set_method(arg.second.asString());}

       // Path options
       // if (arg.first == "--grid-path") {jobinfo.set_grid_path(arg.second.asString());}
       // if (arg.first == "--orb-path")  {jobinfo.set_orb_path(arg.second.asString());}
       // if (arg.first == "--points-sets-path") {jobinfo.set_points_set_path(arg.second.asString());}


       //Point options
       // if (arg.first == "--ngrid") {jobinfo.num_points = (int) arg.second.asLong();}
       if (arg.first == "--npair") {jobinfo.num_pairs = (int) arg.second.asLong();}

       //Seed options
       if (arg.first == "--seed" && arg.second) {jobinfo.seed = (int) arg.second.asLong();}

       // set relative path.
       jobinfo.set_work_path();
       jobinfo.set_orb_path();
       jobinfo.set_grid_path();
       jobinfo.set_t_path();

    }//end loop over arguements

    printf("Job Information\n%s\n",jobinfo.print_str().c_str());

    return jobinfo.validate();
}


#endif
