#pragma once
#include <string>
class Mtx
{
private:
  double* data;
  unsigned int alloc_size;
  unsigned int* upper_id_rank;
  unsigned int global_shift; 
  int nb_cols;
  int nb_rows;
  int lower_id;
  int upper_id;
public:
  Mtx(int n_rows, int n_cols);
  Mtx();
  ~Mtx();
  void initFromFile(std::string file_name);
  int getNb_rows() const { return this->nb_rows; }
  int getNb_cols() const { return this->nb_cols; }
  int getLower_id() const { return this->lower_id; }
  int getUpper_id() const { return this->upper_id; }
  unsigned int getAllocatedSize() const { return this->alloc_size; }
  // Operators A(i,j)
  double& operator()(int i ,int j)
  {
    return (data[(i*nb_cols + j) - this->global_shift]);
  }
  double& operator()(unsigned int i, unsigned int j)
  {
    return (data[(i*nb_cols + j) - this->global_shift]);
  }
  double operator()(int i, int j) const 
  {
    return (data[(i*nb_cols + j) - this->global_shift]); 
  }
  double operator()(unsigned int i, unsigned int j) const 
  { 
    return (data[(i*nb_cols + j) - this->global_shift]);
  }
  Mtx& operator=(Mtx mtx);
  int getUpperRank(int rank) const;
  void fillRandom(int seed = 0);
  void fillConst(double value);
  void fillTestA3();
  void fillTestA9(double a, double b);
  void fillResultA3();
  void fillResultA9(double a, double b);
  void writeFile(std::string file_name);
  void debug();
  void printValue();
};


