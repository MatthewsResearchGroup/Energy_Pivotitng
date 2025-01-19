/***************************************************
 * io.hpp
 *
 * Header file for io operators in energyAnalysis.cxx
 *
 * Current list of functions:
 *     read_dimensions
 *     tensor_from_file
 *     aibj_from_file 
 *     output_to_csv
****************************************************/
#ifndef _ENERGYANALYSIS_IO_HPP_
#define _ENERGYANALYSIS_IO_HPP_

#include "energyAnalysis.hpp"
#include "marray.hpp"

#include <string>
#include <stdio.h>
#include <fstream>
#include <filesystem>


/***************************************************
 * read_dimensions
 *
 * reads the dimensions of the neccesary tensors  
***************************************************/
//Read the relevant dimensions from the data files
void read_dimensions(Int& nv,  const std::string& nv_file_path,
                     Int& no,  const std::string& no_file_path,
                     Int& nps, const std::string& nps_file_path) {
                     // std::vector<int>& psvec, const std::string& ps_file_path) {

    //Read number of virtuals 
    auto nv_file = fopen(nv_file_path.c_str(), "rb");
    if (!nv_file) {
        printf("Could not open %s \n",nv_file_path.c_str());
        exit(1);
    }
    fread(&nv, sizeof(Int), 1, nv_file);
    if (fclose(nv_file) != 0) {
        printf("Error closing %s\n", nv_file_path.c_str());
        exit(1);
    }

    //Read number of occupied
    auto no_file = fopen(no_file_path.c_str(), "rb");
    if (!no_file) {
        printf("Could not open %s \n", no_file_path.c_str());
        exit(1);
    }
    fread(&no, sizeof(Int), 1, no_file);
    if (fclose(no_file) != 0) {
        printf("Error closing %s\n", no_file_path.c_str());
        exit(1);
    }

    //Read number of gridpoInts
    auto nps_file = fopen(nps_file_path.c_str(),"rb");
    if (!nps_file) {
        printf("Could not open %s \n", nps_file_path.c_str());
        exit(1);
    }
    fread(&nps, sizeof(Int), 1, nps_file);
    if (fclose(nps_file) != 0) {
        printf("Error closing %s\n", nps_file_path.c_str());
        exit(1);
    }

    // Read the points sets
    // std::string s;
    // std::fstream ps_file;
    // ps_file.open(ps_file_path.c_str());
    // if (!ps_file.is_open())
    // {
    //     printf("Could not open %s \n", ps_file_path.c_str());
    //     exit(1);
    // }
    // std::getline(ps_file, s); // Get rid of the first row, which may be the name of data
    // while(std::getline(ps_file, s))
    // {
    //     printf("%s\n",s.c_str());
    //     psvec.push_back(stoi(s));
    // }
}

/***************************************************
 * tensor_from_file
 * 
 * Read a (double) tensor from a file, with an offset 
***************************************************/
tensor<2> tensor_from_file(const std::string& file_name,
                           const size_t offset,
                           const Int n0,
                           const Int n1) {
    tensor<2> data{{n0, n1},COLUMN_MAJOR};

    auto data_file = fopen(file_name.c_str(),"rb");
    if (!data_file) {
        printf("Error opening %s \n",file_name.c_str());
    }

    if (fseek(data_file, offset, SEEK_SET) != 0) {
        printf("Error seeking to %zu in %s \n",offset,file_name.c_str());
    } 

    for (auto j : range(data.length(1))) {
        fread(&data[0][j], sizeof(double), data.length(0), data_file);
    }

    if (fclose(data_file) != 0) {
        printf("Error closing %s \n",file_name.c_str());
    }

    return data;
    
}

tensor<4> tensor_from_file(const std::string& file_name,
                           const size_t offset,
                           const Int n0,
                           const Int n1,
                           const Int n2,
                           const Int n3) {

    tensor<4> data{{n0, n1, n2, n3},COLUMN_MAJOR};

    auto data_file = fopen(file_name.c_str(),"rb");
    if (!data_file) {
        printf("Error opening %s \n",file_name.c_str());
    }

    if (fseek(data_file, offset, SEEK_SET) != 0) {
        printf("Error seeking to %zu in %s \n",offset,file_name.c_str());
    } 

    for (auto l : range(data.length(3)))
    for (auto k : range(data.length(2)))
    for (auto j : range(data.length(1)))
    {
        fread(&data[0][j][k][l], sizeof(double), data.length(0), data_file);
    }

    if (fclose(data_file) != 0) {
        printf("Error closing %s \n",file_name.c_str());
    }

    return data;
}

/***************************************************
 * aibj_cx_from_file
 *
 * Reads a tensor stored (vrt, occ, vrt, occ) from a 
 * path. 
****************************************************/
tensor<4> aibj_from_file(const std::string& file_name,
                         const size_t offset,
                         const Int nv,
                         const Int no) {
                         
    tensor<4> data{{nv, no, nv, no},COLUMN_MAJOR};
    
    auto data_file = fopen(file_name.c_str(),"rb");
    if (data_file == NULL) {
        printf("Error opening %s \n",file_name.c_str());
    }   
    
    if (fseek(data_file, offset, SEEK_SET) != 0) {
        printf("Error seeking to %zu in %s \n",offset, file_name.c_str());
    } 

    for (auto j : range(no))
    for (auto i : range(no))
    for (auto b : range(nv)) 
    {
        fread(&data[0][i][b][j], sizeof(double), nv, data_file);
    }

    if (fclose(data_file) != 0) 
    {
        printf("Error closing %s \n",file_name.c_str());
    }

    return data;
    
}

/***************************************************
 * output_to_csv
 *
 * writes the output of the calculation to a files called
 * energyAnalysis.csv 
***************************************************/
void output_to_csv(const Jobinfo& jobinfo,
                   const pair_list& pairs,
                   const view<1>& Fo,  const view<1>& Svals,
                   const view<1>& E4C, const view<1>& E4X,
                   const view<1>& E8C, const view<1>& E8X)
{
    std::string filename = "energyAnalysis.csv";
    auto out_csv = fopen(filename.c_str(),"w+");
    if (!out_csv) 
    {
        printf("\nERROR: could not open %s for output\n",filename.c_str());
        exit(1);
    }

    //Generate formatting for output 
    std::string header = "Point1,Point2";
    std::string values_f = "%d,%d";

    header   += ",Fval,Sval";
    values_f += ",%lf,%lf";

//    header   += ",E2C,E2X";
//    values_f += ",%lf,%lf";

    header   += ",E4C,E4X";
    values_f += ",%lf,%lf";

    header   += ",E8C,E8X";
    values_f += ",%lf,%lf";

    values_f += "\n";

    if (std::filesystem::is_empty(filename))
        fprintf(out_csv,"%s\n",header.c_str());
    for (auto pair : range(jobinfo.num_pairs))
    {
        fprintf(out_csv,values_f.c_str(),
                pairs[pair].first, pairs[pair].second,
                Fo[pair],          Svals[pair],
                E4C[pair],         E4X[pair],
                E8C[pair],         E8X[pair]);
    }
    
    fclose(out_csv);
}


/***************************************************
 * output_to_csv
 *
 * Another version to save the reulst for points set of grid
 * writes the output of the calculation to a files called
 * energyAnalysis.csv 
***************************************************/
template <typename T>
void output_to_csv(const Jobinfo& jobinfo,
                   const pair_list& pairs,
                   int ps,
                   const view<1>& Fo,  const view<1>& Svals,
                   const view<1>& Mu,
                   const T& E2C, const T& E2X,
                   const view<1>& E4C, const view<1>& E4X,
                   const view<1>& E8C, const view<1>& E8X)
{
    // std::string filename = "energyAnalysis.csv"; 
    std::string filename = "energyAnalysis_" + jobinfo.method + ".csv"; 
    std::string ftype;
    if (jobinfo.ftype == Jobinfo::Ftypes::all)
        ftype = "all";
    else if (jobinfo.ftype == Jobinfo::Ftypes::cross)
        ftype = "cross";
    else if (jobinfo.ftype == Jobinfo::Ftypes::nocross)
        ftype = "nocross";

    auto out_csv = fopen(filename.c_str(),"a+");

    if (!out_csv) 
    {
        printf("\nERROR: could not open %s for output\n",filename.c_str());
        exit(1);
    }
    //Generate formatting for output 
    std::string header = "Method,Ftype";
    std::string values_f = "%s, %s";

    header   += ",npairs,num_point_keep";
    values_f += ", %6d, %5d";

    header   += ",Point1,Point2";
    values_f += ", %6d, %6d";

    header   += ",Fval,Sval";
    values_f += ", %e, %e";

    header   += ",Mu";
    values_f += ", %e";

    header   += ",E2C,E2X";
    values_f += ", %e, %e";

    header   += ",E4C,E4X";
    values_f += ", %e, %e";

    header   += ",E8C,E8X";
    values_f += ", %e, %e";

    values_f += "\n";
    
    if (std::filesystem::is_empty(filename))
        fprintf(out_csv,"%s\n",header.c_str());
    // fprintf(out_csv, "---------------------------------------------------\n");
    for (auto pair : range(jobinfo.num_pairs))
    {
        fprintf(out_csv,values_f.c_str(),
                // jobinfo.ftype, 
                jobinfo.method.c_str(), ftype.c_str(), 
                jobinfo.num_pairs, ps,
                pairs[pair].first, pairs[pair].second,
                Fo[pair],          Svals[pair],
                Mu[pair],              
                E2C,               E2X,
                E4C[pair],         E4X[pair],
                E8C[pair],         E8X[pair]);
    }
    // fprintf(out_csv, "\n\n\n\n");

    // close the file if it's the last element in the points set vector. 
    fclose(out_csv);
}

// template <typename T>
void output_to_csv(const Jobinfo& jobinfo,
                   const pair_list& pairs,
                   int ps,
                   const view<1>& Fo,  const view<1>& Svals,
                   const view<1>& Mu,
                   const view<1>& E8C, const view<1>& E8X)
{
    std::string filename = "energyAnalysis_" + jobinfo.method + ".csv"; 
    
    // get the fype string
    std::string ftype;
    if (jobinfo.ftype == Jobinfo::Ftypes::all)
        ftype = "all";
    else if (jobinfo.ftype == Jobinfo::Ftypes::cross)
        ftype = "cross";
    else if (jobinfo.ftype == Jobinfo::Ftypes::nocross)
        ftype = "nocross";
    auto out_csv = fopen(filename.c_str(),"a+");

    if (!out_csv) 
    {
        printf("\nERROR: could not open %s for output\n",filename.c_str());
        exit(1);
    }
    //Generate formatting for output 
    std::string header = "Method,Ftype";
    std::string values_f = "%s, %s";

    header   += ",npairs,num_point_keep";
    values_f += ", %6d, %5d";

    header   += ",Point1,Point2";
    values_f += ", %6d, %6d";

    header   += ",Fval,Sval";
    values_f += ", %e, %e";

    header   += ",Mu";
    values_f += ", %e";

    header   += ",E2C,E2X";
    values_f += ", %e, %e";

    header   += ",E4C,E4X";
    values_f += ", %e, %e";

    header   += ",E8C,E8X";
    values_f += ", %e, %e";

    values_f += "\n";
    
    if (std::filesystem::is_empty(filename))
        fprintf(out_csv,"%s\n",header.c_str());
    // fprintf(out_csv, "---------------------------------------------------\n");
    for (auto pair : range(jobinfo.num_pairs))
    {
        fprintf(out_csv,values_f.c_str(),
                jobinfo.method.c_str(), ftype.c_str(), 
                jobinfo.num_pairs, ps,
                pairs[pair].first, pairs[pair].second,
                Fo[pair],          Svals[pair],
                Mu[pair],              
                0.f, 0.f,
                0.f, 0.f,
                E8C[pair],         E8X[pair]);
    }
    //fprintf(out_csv, "\n\n\n\n");

    // close the file if it's the last element in the points set vector. 
    fclose(out_csv);
}
#endif
