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
  int comm_size = 0;
  int comm_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  std::complex<double>* A = new std::complex<double>[n*n];
  std::complex<double>* VR = new std::complex<double>[n*n];
  std::complex<double>* W = new std::complex<double>[n];

  int nb_own_values  = H.getAllocatedSize();
  double* own_buffer = new double[nb_own_values];
  int k = 0;
  for (unsigned int i = H.getLower_id(); i <= H.getUpper_id(); i++)
  {
    for (unsigned int j = 0; j < n; j++)
    {
      own_buffer[k] = H(i,j); 
      A[i*n+j].real(own_buffer[k]);
      A[i*n+j].imag(0);
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
      k = 0;
      for (unsigned int l = lower_id; l <= upper_id; l++)
      {
        for (unsigned int c = 0; c < n; c++)
        {
          A[l*n+c].real(tmp_buffer[k]);
          A[l*n+c].imag(0);
          k++;
        }
      }
    }
    delete[] tmp_buffer;
  }
  delete[] own_buffer;

  int return_code = 0;
  return_code = lapack::geev(lapack::Job::NoVec, lapack::Job::Vec, n, A, n, W, nullptr, 1, VR, n);
  if (return_code != 0)
  {
    std::cerr << "[Lapack] QR algorithm failed to compute all the eigenvalues \n"; 
  }
  *eigenValue = W;
  *eigenVectors = VR;
  return;
}

void sortEigenValue(std::complex<double>** eigenValue, std::complex<double>** eigenVectors, int n)
{
  for (size_t i = 0; i < n; i++)
  {
    std::cout << std::abs((*eigenValue)[i]) <<"\n"; 
    if (std::abs((*eigenValue)[i]))
    {
    }
  }
}

