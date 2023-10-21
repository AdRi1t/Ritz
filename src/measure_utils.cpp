#include <algorithm>
#include <iomanip>
#include <iso646.h>
#include <string>
#include <math.h>
#include <fstream>
#include <mpi.h>
#include "measure_utils.hpp"

build_stats::build_stats() {}

build_stats::~build_stats() {}

void build_stats::add_data(std::string data_name, std::string unit, double* measure, int problem_size, int nb_mesaure)
{
  makeStats(measure, nb_mesaure);
  this->name.push_back(data_name);
  this->unit.push_back(unit);
  this->problem_size.push_back(problem_size);
}

void build_stats::printStats() {}

void build_stats::writeAll(std::string file_name)
{
  int comm_rank = 0;
  int comm_size = 0;
  std::fstream dataFile;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  if(comm_rank == 0)
  {
    dataFile.open(file_name, std::ios_base::out | std::ios_base::app);
    dataFile << std::setw(12) << "Nb_process" << "|" 
             << std::setw(14) << "Measure name" << "|" 
             << std::setw(14) << "Problem size" << "|"
             << std::setw(8)  << "Unit" << "|" 
             << std::setw(12) << "Average" << "|" 
             << std::setw(12) << "Min" << "|" 
             << std::setw(12) << "Max" << "|" 
             << std::setw(14) << "std deviation" << "\n";
    for(size_t i = 0; i < name.size(); i++)
    {
      dataFile << std::setw(12) << comm_size << "|" 
               << std::setw(14) << name[i] << "|" 
               << std::setw(14) << problem_size[i] << "|"
               << std::setw(8)  << unit[i] << "|" 
               << std::setw(12) << std::setprecision(6) << average[i] << "|"
               << std::setw(12) << std::setprecision(6) << min[i] << "|"
               << std::setw(12) << std::setprecision(6) << max[i] << "|"
               << std::setw(14) << std::setprecision(6) << std_deviation[i] << "\n";
    }
  dataFile.flush();
  dataFile.close();
  }
}

void build_stats::makeStats(double* values, int nb_value)
{
  double average = 0;
  double std_dev = 0;
  double min = values[0];
  double max = values[0];
  for(size_t i = 0; i < nb_value; i++)
  {
    average += values[i];
    min = std::min(min, values[i]);
    max = std::max(max, values[i]);
  }
  average /= nb_value;
  for(size_t i = 0; i < nb_value; i++)
  {
    std_dev += pow((values[i]-average),2);
  }
  std_dev /= nb_value;
  std_dev = sqrt(std_dev);
  this->average.push_back(average);
  this->min.push_back(min);
  this->max.push_back(max);
  this->std_deviation.push_back(std_dev);
}


