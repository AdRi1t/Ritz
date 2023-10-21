#include <memory>
#include <string>
#include <vector>

#pragma once

class build_stats
{
private:
  std::vector<int> problem_size;
  std::vector<std::string> name;
  std::vector<std::string> unit;
  std::vector<double> min;
  std::vector<double> max;
  std::vector<double> average;
  std::vector<double> std_deviation;
  void makeStats(double* values, int nb_value);

public:
  build_stats();
  ~build_stats();
  void add_data(std::string data_name, std::string unit, double* measure, int problem_size, int nb_mesaure);
  void printStats();
  void writeAll(std::string file_name);
};


