#include <iostream>
#include <mpi.h>
#include <complex.h>
#include "mat.hpp"
#include "arnoldiEnv.h"
#include "operation.hpp"
#include "../lib/BLAS/include/blas.hh"
#include "../lib/lapackpp/include/lapack.hh"


void computeEigen(const Mtx& H, std::complex<double>** eigenValue, std::complex<double>** eigenVectors)
{
  if (H.getNb_cols() != H.getNb_rows())
  {
    throw std::logic_error("Not a square Matrix");
  }

  int n = H.getNb_rows();
  int m = H.getNb_cols();
  assert(n == m);
  int comm_size = 0;
  int comm_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  std::complex<double>* A = new std::complex<double>[n*n];
  *eigenVectors = new std::complex<double>[n*n];
  *eigenValue = new std::complex<double>[m];

  int nb_own_values  = H.getAllocatedSize();
  double* own_buffer = new double[nb_own_values];
  int k = 0;
  for (unsigned int i = H.getLower_id(); i <= H.getUpper_id(); i++)
  {
    for (unsigned int j = 0; j < m; j++)
    {
      own_buffer[k] = H(i,j); 
      A[i*m+j].real(own_buffer[k]);
      A[i*m+j].imag(0);
      k++;
    }
  }
  for (int i = 0; i < comm_size; i++)
  {
    int tmp_nb_values = 0;
    double* tmp_buffer = nullptr;
    int upper_id = 0;
    int lower_id = 0;
    if(comm_rank == i)
    {
      tmp_nb_values = nb_own_values;
      MPI_Bcast(&tmp_nb_values, 1, MPI_INT, i ,MPI_COMM_WORLD);
      tmp_buffer = new double[tmp_nb_values];
      MPI_Bcast(own_buffer, tmp_nb_values, MPI_DOUBLE, i ,MPI_COMM_WORLD);
    }
    else
    {
      MPI_Bcast(&tmp_nb_values, 1, MPI_INT, i ,MPI_COMM_WORLD);
      tmp_buffer = new double[tmp_nb_values];
      MPI_Bcast(tmp_buffer, tmp_nb_values, MPI_DOUBLE, i ,MPI_COMM_WORLD);
    }
    if(i == 0)
    {
      lower_id = 0;
      upper_id = H.getUpperRank(i);
    }
    else
    {
      lower_id = H.getUpperRank(i-1);
      upper_id = H.getUpperRank(i);
    }
    if(i != comm_rank)
    {
      std::cout << __LINE__ << std::endl;
      k = 0;
      for (unsigned int l = lower_id; l <= upper_id; l++)
      {
        for (unsigned int c = 0; c < m; c++)
        {
          A[l*m+c].real(tmp_buffer[k]);
          A[l*m+c].imag(0);
          k++;
        }
      }
    }
    delete[] tmp_buffer;
  }
  delete[] own_buffer;
  
  int return_code = 0;
  return_code = lapack::geev(lapack::Job::Vec, lapack::Job::NoVec, n, A, n, *eigenValue, *eigenVectors, n, nullptr, 1);
  if (return_code != 0)
  {
    std::cerr << "[Lapack] QR algorithm failed to compute all the eigenvalues \n"; 
  }

  // Transpose
  std::complex<double>* tmp_Vectors = new std::complex<double>[n*n];
  for (size_t i = 0; i < n; i++)
  { 
    for (size_t j = 0; j < n; j++)
    {
      tmp_Vectors[i*n+j] = (*eigenVectors)[j*n+i];
    }
  }
  for (size_t i = 0; i < n; i++)
  { 
    for (size_t j = 0; j < n; j++)
    {
      (*eigenVectors)[i*n+j] = tmp_Vectors[i*n+j];
    }
  }
  delete[] tmp_Vectors;
  delete[] A;
  return;
}

void sortEigenValue(std::complex<double>** eigenValue, std::complex<double>** eigenVectors, int n, int s)
{
  int* index = new int[n];
  for (size_t i = 0; i < n; i++)
  {
    index[i] = i;
  }
  for (size_t i = 0; i < n; i++)
  { 
    for (size_t j = i; j < n; j++)
    {
      // i < j si v(i) < v(j)
      if( std::abs((*eigenValue)[i]) < std::abs((*eigenValue)[j]) )
      {
        std::complex<double> tmp = (*eigenValue)[i];
        (*eigenValue)[i] = (*eigenValue)[j];
        (*eigenValue)[j] = tmp;
        int tmp_pos = i;
        index[i] = j;
        index[j] = tmp_pos;
      }
    }
  }
  std::complex<double>* tmp_Vectors = new std::complex<double>[n*n];
  for (size_t i = 0; i < n; i++)
  { 
    for (size_t j = 0; j < n; j++)
    {
      tmp_Vectors[i*n+j] = (*eigenVectors)[i*n+j];
    }
  }
  for (size_t i = 0; i < n; i++)
  { 
    for (size_t j = 0; j < n; j++)
    {
      (*eigenVectors)[i*n+j] = tmp_Vectors[index[i]*n + j];
    }
  }
  std::complex<double>* new_values = new std::complex<double>[s];
  std::complex<double>* new_vectors = new std::complex<double>[n*s];
  for (size_t i = 0; i < n; i++)
  { 
    for (size_t j = 0; j < s; j++)
    {
      new_vectors[i*s+j] =  (*eigenVectors)[i*n+j];
    }
  }
  for (size_t j = 0; j < s; j++)
  {
    new_values[j] = (*eigenValue)[j];
  }
  // XD
  delete[] (*eigenValue);
  delete[] (*eigenVectors);
  (*eigenValue) = new_values;
  (*eigenVectors) = new_vectors;
  
  delete[] index;
  delete[] tmp_Vectors;
}

void sortEigenValue(std::complex<double>** eigenValue, std::complex<double>** eigenVectors, std::complex<double>** mu, int m, int s)
{
  assert(m >= s);
  int* index = new int[m];
  for (size_t i = 0; i < m; i++)
  {
    index[i] = i;
  }
  for (size_t i = 0; i < m; i++)
  { 
    for (size_t j = i; j < m; j++)
    {
      // i < j si v(i) < v(j)
      if( std::abs((*eigenValue)[i]) < std::abs((*eigenValue)[j]) )
      {
        std::complex<double> tmp = (*eigenValue)[i];
        (*eigenValue)[i] = (*eigenValue)[j];
        (*eigenValue)[j] = tmp;
        int tmp_pos = i;
        index[i] = j;
        index[j] = tmp_pos;
      }
    }
  }
  std::complex<double>* tmp_Vectors = new std::complex<double>[m*m];
  for (size_t i = 0; i < m; i++)
  { 
    for (size_t j = 0; j < m; j++)
    {
      tmp_Vectors[i*m+j] = (*eigenVectors)[i*m+j];
    }
  }
  for (size_t i = 0; i < m; i++)
  { 
    for (size_t j = 0; j < m; j++)
    {
      (*eigenVectors)[i*m+j] = tmp_Vectors[index[i]*m + j];
    }
  }
  std::complex<double>* new_values = new std::complex<double>[s];
  std::complex<double>* new_vectors = new std::complex<double>[m*s];
  *mu = new std::complex<double>[m-s];
  for (size_t i = 0; i < m; i++)
  { 
    for (size_t j = 0; j < s; j++)
    {
      new_vectors[i*s+j] = (*eigenVectors)[i*m+j];
    }
  }
  for (size_t j = 0; j < s; j++)
  {
    new_values[j] = (*eigenValue)[j];
  }
  for (size_t i = 0; i < (m-s); i++)
  {
    (*mu)[i] = (*eigenValue)[s+i];
  }
  
  // XD
  delete[] (*eigenValue);
  delete[] (*eigenVectors);
  (*eigenValue) = new_values;
  (*eigenVectors) = new_vectors;
  
  delete[] index;
  delete[] tmp_Vectors;
}

void printEigenValue(const std::complex<double> *eigenValues, int n)
{
  for (size_t i = 0; i < n; i++)
  {
    std::cout << eigenValues[i] << " | " ;
  }
  std::cout << "\n" ;
}

void printEigenVectors(const std::complex<double> *eigenVectors, int n)
{
  for (size_t i = 0; i < n; i++)
  { 
    for (size_t j = 0; j < n; j++)
    {
      std::cout << eigenVectors[i*n+j] << " | " ;
    }
    std::cout << "\n" ;
  }
}

void printEigenVectors(const std::complex<double> *eigenVectors, int n, int m)
{
  for (size_t i = 0; i < n; i++)
  { 
    for (size_t j = 0; j < m; j++)
    {
      std::cout << eigenVectors[i*m+j] << " | " ;
    }
    std::cout << "\n" ;
  }
}


void computeUs(std::complex<double> **Y , const Mtx& Vm, std::complex<double> **U, int s)
{
  int n = Vm.getNb_rows();
  int m = Vm.getNb_cols();
  int comm_size = 0;
  int comm_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  std::complex<double>* V_m = new std::complex<double>[n*m]();
  *U = new std::complex<double>[n*s]();
  
  int nb_own_values  = Vm.getAllocatedSize();
  double* own_buffer = new double[nb_own_values];
  int k = 0;
  for (unsigned int i = Vm.getLower_id(); i <= Vm.getUpper_id(); i++)
  {
    for (unsigned int j = 0; j < m; j++)
    {
      own_buffer[k] = Vm(i,j); 
      V_m[i*m+j].real(own_buffer[k]);
      V_m[i*m+j].imag(0);
      k++;
    }
  }
  for (int i = 0; i < comm_size; i++)
  {
    int tmp_nb_values = 0;
    double* tmp_buffer = nullptr;
    int upper_id = 0;
    int lower_id = 0;
    if(comm_rank == i)
    {
      tmp_nb_values = nb_own_values;
      MPI_Bcast(&tmp_nb_values, 1, MPI_INT, i ,MPI_COMM_WORLD);
      tmp_buffer = new double[tmp_nb_values];
      MPI_Bcast(own_buffer, tmp_nb_values, MPI_DOUBLE, i ,MPI_COMM_WORLD);
    }
    else
    {
      MPI_Bcast(&tmp_nb_values, 1, MPI_INT, i ,MPI_COMM_WORLD);
      tmp_buffer = new double[tmp_nb_values];
      MPI_Bcast(tmp_buffer, tmp_nb_values, MPI_DOUBLE, i ,MPI_COMM_WORLD);
    }
    if(i == 0)
    {
      lower_id = 0;
      upper_id = Vm.getUpperRank(i);
    }
    else
    {
      lower_id = Vm.getUpperRank(i-1);
      upper_id = Vm.getUpperRank(i);
    }
    if(i != comm_rank)
    {
      k = 0;
      for (unsigned int l = lower_id; l <= upper_id; l++)
      {
        for (unsigned int c = 0; c < m; c++)
        {
          V_m[l*m+c].real(tmp_buffer[k]);
          V_m[l*m+c].imag(0);
          k++;
        }
      }
    }
    delete[] tmp_buffer;
  }
  delete[] own_buffer;
  
  std::complex<double> alpha(1.0 , 0.0);
  std::complex<double> beta(0.0 , 0.0);
  // u = vm * y
  std::cout << "\n";
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, n, s, m, alpha, V_m, m,*Y, s, beta, *U, s);

  delete[] V_m;
  return;
}