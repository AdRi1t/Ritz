#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <math.h>
#include "mat.hpp"

extern "C"
{
  #include "../lib/mmio/mmio.h"
}

Mtx::Mtx(int i, int j)
{
  if (i > 0 && j > 0)
  {
    this->nb_rows = i;
    this->nb_cols = j;
    int comm_size;
    int comm_rank;
    int local_data;
    int extra_data;
    unsigned int allocSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    this->upper_id_rank = new unsigned int[comm_size]();

    if (i != 1 && j != 1)
    {
      local_data = this->nb_rows / comm_size;
      extra_data = this->nb_rows % comm_size;
      allocSize = this->nb_cols;

      if(comm_rank < extra_data)
      {
        this->lower_id = local_data * comm_rank + comm_rank;
        this->upper_id = local_data * (comm_rank + 1) + comm_rank;
      }
      else
      {
        this->lower_id = local_data * comm_rank + extra_data;
        this->upper_id = local_data * (comm_rank + 1) + extra_data - 1;
      }
      allocSize *= (this->upper_id - this->lower_id + 1);
      this->data = new double[allocSize]();
      MPI_Allgather(&(this->upper_id),  1,  MPI_INT,  this->upper_id_rank, 1,  MPI_INT,  MPI_COMM_WORLD);
    }
    else  // Vecteur
    {
      this->lower_id = 0; 
      this->upper_id = std::max(this->nb_rows,this->nb_cols) - 1;
      allocSize = i*j;
      this->data = new double[allocSize]();
      MPI_Allgather(&(this->upper_id),  1,  MPI_INT, this->upper_id_rank, 1,  MPI_INT,  MPI_COMM_WORLD);
    }
    this->alloc_size = allocSize;
    this->global_shift = (this->nb_cols) * (this->lower_id);
  }
  else
  {
    std::cerr << "Can't creat object in : " << __FILE__ << ":" << __LINE__ << std::endl;
  }
}

Mtx::Mtx()
{
  this->data = new double[1]();
  this->upper_id_rank = new unsigned int[1]();
  this->alloc_size = 1;
  this->global_shift = 0; 
  this->nb_rows = 0;
  this->nb_cols = 0;
  this->lower_id = 0;
  this->upper_id = 0;
}

Mtx::~Mtx()
{ 
  if(this->alloc_size > 1)
  {
    delete[] this->data;
    delete[] this->upper_id_rank;
  }
}

int Mtx::getUpperRank(int rank) const
{
  int comm_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if(rank >= comm_size)
  {
    throw std::runtime_error("Bad Rank for getUpperRank");
  }
  return this->upper_id_rank[rank];
}

void Mtx::fillRandom(int seed) {
  std::srand(seed);
  for (unsigned int i = this->lower_id; i <= this->upper_id; i++)
  {
    for (unsigned int j = 0; j < this->nb_cols; j++)
    {
      (*this)(i,j) =  ((double)std::rand() / (double)RAND_MAX) * 10.0;
    }
  }
}

void Mtx::fillConst(double value)
{
  for (unsigned int i = this->lower_id; i <= this->upper_id; i++)
  {
    for (unsigned int j = 0; j < this->nb_cols; j++)
    {
      (*this)(i,j) = value;
    }
  }
}


void Mtx::initFromFile(std::string file_name)
{
  FILE* mm_file;
  MM_typecode matcode;
  bool is_symmetric = false;
  int nb_row, nb_column, nb_data;
  long cursor;
  long end_cursor;
  char* big_buffer;
  char* p_big_buffer;
  mm_file = fopen(file_name.c_str(), "r");
  if(mm_file == NULL)
  {
    std::cerr << "Could not open file: ";
  }
  if(mm_read_banner(mm_file, &matcode) != 0)
  {
    std::cerr << "Could not process Matrix Market banner" << std::endl;
  }
  if(mm_read_mtx_crd_size(mm_file, &nb_row, &nb_column, &nb_data) != 0)
  {
    std::cerr << "Could not process Matrix Market dimensions" << std::endl;
  }
  if(mm_is_symmetric(matcode))
  {
    is_symmetric = true;
  }

  if (nb_row > 0 && nb_column > 0)
  {
    this->nb_rows = nb_row;
    this->nb_cols = nb_column;
    int comm_size;
    int comm_rank;
    int local_data;
    int extra_data;
    unsigned int allocSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    this->upper_id_rank = new unsigned int[comm_size]();

    if (nb_row != 1 && nb_column != 1)
    {
      local_data = this->nb_rows / comm_size;
      extra_data = this->nb_rows % comm_size;
      allocSize = this->nb_cols;

      if(comm_rank < extra_data)
      {
        this->lower_id = local_data * comm_rank + comm_rank;
        this->upper_id = local_data * (comm_rank + 1) + comm_rank;
      }
      else
      {
        this->lower_id = local_data * comm_rank + extra_data;
        this->upper_id = local_data * (comm_rank + 1) + extra_data - 1;
      }
      allocSize *= (this->upper_id - this->lower_id + 1);
      this->data = new double[allocSize]();
      MPI_Allgather(&(this->upper_id),  1,  MPI_INT,  this->upper_id_rank, 1,  MPI_INT,  MPI_COMM_WORLD);
    }
    else  // Vecteur
    {
      this->lower_id = 0; 
      this->upper_id = std::max(this->nb_rows,this->nb_cols) - 1;
      allocSize = nb_row*nb_column;
      this->data = new double[allocSize]();
      MPI_Allgather(&(this->upper_id),  1,  MPI_INT, this->upper_id_rank, 1,  MPI_INT,  MPI_COMM_WORLD);
    }
    this->alloc_size = allocSize;
    this->global_shift = (this->nb_cols) * (this->lower_id);
  }

  cursor = ftell(mm_file);
  fseek(mm_file, 0, SEEK_END);
  end_cursor = ftell(mm_file);
  fseek(mm_file, cursor, SEEK_SET);
  big_buffer = new char[end_cursor - cursor];
  fread(big_buffer, sizeof(char), end_cursor - cursor, mm_file);
  p_big_buffer = big_buffer;

  int row = 0;
  int col = 0;
  double value = 0;
  if(is_symmetric == true)
  {
    for (int i = 0; i < nb_data; i++)
    {
      row = (int)strtol(p_big_buffer, &p_big_buffer, 10) - 1;
      col = (int)strtol(p_big_buffer, &p_big_buffer, 10) - 1;
      value = strtod(p_big_buffer, &p_big_buffer);
      if(row >= this->lower_id && row <= this->upper_id)
      {
        (*this)(row,col) = value;
      }
      if(col >= this->lower_id && col <= this->upper_id)
      {
        (*this)(col, row) = value;
      }
    }
  }
  else
  {
    for (int i = 0; i < nb_data; i++)
    {
      row = (int)strtol(p_big_buffer, &p_big_buffer, 10) - 1;
      col = (int)strtol(p_big_buffer, &p_big_buffer, 10) - 1;
      value = strtod(p_big_buffer, &p_big_buffer);
      if(row >= this->lower_id && row <= this->upper_id)
      {
        (*this)(row,col) = value;
      }
    }
  }
  free(big_buffer);
}

void Mtx::writeFile(std::string file_name)
{
  FILE* mm_file;
  MM_typecode matcode;
  int comm_rank = 0;
      
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  
  std::string new_file_name(file_name);
  new_file_name.append(".");
  new_file_name += std::to_string(comm_rank);
  new_file_name.append(".mtx");
  
  mm_file = fopen(new_file_name.c_str(), "w");
  if(mm_file == NULL)
  {
    std::cerr << "Could not open file: " << std::endl;
    exit(EXIT_FAILURE);
  }
  
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_dense(&matcode);
  mm_set_real(&matcode);
  mm_set_general(&matcode);
  mm_write_banner(mm_file, matcode);
  mm_write_mtx_array_size(mm_file, this->nb_rows, this->nb_rows);
  
  for (unsigned int i = this->lower_id; i <= this->upper_id; i++)
  {
    for (unsigned int j = 0; j < this->nb_cols; j++)
    {      
      fprintf(mm_file, "%d %d %20.16e\n", i + 1, j + 1, (*this)(i,j));
    }
  }
  fclose(mm_file);
}

void Mtx::debug()
{
  std::cout << "Global line : " << nb_rows << "\n";
  std::cout << "Global cols : " << nb_cols << "\n";
  std::cout << "Bound : [ " << lower_id << " ; " << upper_id << " ]\n";
  std::cout << "Global shift : " << global_shift << "\n";
  std::cout << "Allocated size : " << alloc_size << "\n";
  int comm_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  std::cout << "Upper rank : ";
  for (int i = 0; i < comm_size; i++)
  {
    std::cout << upper_id_rank[i] << ", ";
  }
  std::cout << "\n"; 
}

void Mtx::printValue()
{
  std::cout << "\n";
  for (unsigned int i = this->lower_id; i <= this->upper_id; i++)
  {
    for (unsigned int j = 0; j < this->nb_cols; j++)
    {
      std::cout << std::setw(10) << std::setprecision(4) << (*this)(i,j) << " | ";
    }
    std::cout << "\n";
  }
}

Mtx& Mtx::operator=(Mtx mtx)
{
  int comm_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if(this->alloc_size > 0)
  {
    delete[] this->data;
    delete[] this->upper_id_rank;
  }
  this->alloc_size = mtx.alloc_size;
  this->data = new double[this->alloc_size]();
  this->upper_id_rank = new unsigned int[comm_size]();
  this->global_shift = mtx.global_shift;
  this->nb_rows = mtx.nb_rows;
  this->nb_cols = mtx.nb_cols;
  this->lower_id = mtx.lower_id;
  this->upper_id = mtx.upper_id;
  std::copy(mtx.data, mtx.data + mtx.alloc_size, this->data);
  std::copy(mtx.upper_id_rank, mtx.upper_id_rank + comm_size, this->upper_id_rank);
  return *this;
}

void Mtx::fillTestA3()
{
  int n = this->nb_rows;
  for (unsigned int j = 0; j < this->nb_cols; j++)
  {
    for (unsigned int i = 0; i <= j; i++)
    {
      if(i >= this->lower_id && i <= this->upper_id)
      {
        (*this)(i,j) =  n + 1 - j;
      }
    }
  }

  for (unsigned int j = 0; j < this->nb_cols - 1; j++)
  {
    for (unsigned int i = j + 1; i < this->nb_rows; i++)
    {
      if(i >= this->lower_id && i <= this->upper_id)
      {
        (*this)(i,j) =  n + 1 - i;
      }
    }
  }
}

void Mtx::fillResultA3()
{
  int n = this->nb_rows;
  for (int i = 0; i < n; i++)
  {
    double tmp = ((2*i-1)*3.1415926535)/(2*n+1) ;
    tmp = 2 - 2*cos(tmp); 
    tmp = 1/tmp;
    (*this)(i,0) = tmp;
  }
}

void Mtx::fillResultA9(double a, double b)
{
  int n = this->nb_rows;
  for(int i = 0; i < n; i++)
  {
    (*this)(i,0) = a+(2*b)*cos((i * 3.1415926535) / (n + 1));
  }
}

void Mtx::fillTestA9(double a, double b)
{
  int n = this->nb_rows;
  for (unsigned int i = this->lower_id; i <= this->upper_id; i++)
  {
    if(i == 0)
    {
      (*this)(i,i)   = a ;
      (*this)(i,i+1) = b ;
    }
    else if(i == n - 1)
    {
      (*this)(i,i-1) = b ;
      (*this)(i,i)   = a ;
    }
    else
    {
      (*this)(i,i-1) = b ;
      (*this)(i,i)   = a ;
      (*this)(i,i+1) = b ;
    }
  }
}

/*
Mtx& Mtx::operator=(const Mtx &mtx)
{
  int comm_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  this->alloc_size = mtx.alloc_size;
  this->data = new double[this->alloc_size]();
  this->upper_id_rank = new unsigned int[comm_size]();
  this->global_shift = mtx.global_shift;
  this->nb_rows = mtx.nb_rows;
  this->nb_cols = mtx.nb_cols;
  this->lower_id = mtx.lower_id;
  this->upper_id = mtx.upper_id;
  std::copy(mtx.data, mtx.data + mtx.alloc_size,this->data);
  std::copy(mtx.upper_id_rank, mtx.upper_id_rank + comm_size,this->upper_id_rank);
  return *this;
}
*/