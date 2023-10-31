#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <complex.h>
#include "configuration.hpp"
#include "mat.hpp"
#include "operation.hpp"
#include "../lib/BLAS/include/blas.hh"
#include "../lib/lapackpp/include/lapack.hh"

// y = A.v
Mtx multVect(const Mtx &A, const Mtx &v)
{
  int comm_size = 0;
  int send_count = A.getUpper_id() - A.getLower_id() + 1 ;
  double* send_buf = new double[send_count]();
  double* recv_buf = new double[v.getNb_rows()];
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int* nb_data_rank = new int[comm_size];
  int* shift = new int[comm_size];

  unsigned int nb_cols =  A.getNb_cols();

  for (unsigned int i = A.getLower_id(); i <= A.getUpper_id(); i++)
  {
    unsigned int index_send_buf = i - A.getLower_id();
    send_buf[index_send_buf] = 0;
    for (unsigned int j = 0; j < nb_cols; j++)
    {
      send_buf[index_send_buf] += A(i,j) * v(j, (unsigned int) 0); 
    }
  }

  MPI_Allgather(&send_count, 1, MPI_INT, nb_data_rank, 1, MPI_INT, MPI_COMM_WORLD);
  shift[0] = 0;
  for (unsigned int i = 1; i < comm_size; i++)
  {
    shift[i] = shift[i - 1] + nb_data_rank[i - 1];
  }
  MPI_Allgatherv(send_buf, send_count, MPI_DOUBLE, recv_buf, nb_data_rank, shift, MPI_DOUBLE, MPI_COMM_WORLD);

  Mtx y(v.getNb_rows(), v.getNb_cols());
  unsigned int v_nb_rows = v.getNb_rows();

  for (unsigned int i = 0; i < v_nb_rows; i++)
  {
    y(i, (unsigned int) 0) = recv_buf[i];
  }

  delete[] send_buf;
  delete[] recv_buf;
  delete[] nb_data_rank;
  delete[] shift;
  return y;
}

// z = y + v
Mtx addVect(const Mtx &v,const Mtx &y)
{
  Mtx z(v.getNb_rows(), v.getNb_cols());
  if( y.getNb_rows() != v.getNb_rows() || y.getNb_cols() != v.getNb_cols())
  {
    std::cerr << "Bad dimension for z = x + y" << std::endl;
    return z; 
  }
  #pragma omp parallel for num_threads(2)
  {
    for (int i = 0; i < v.getAllocatedSize(); i++)
    {
      z(i,0) = y(i,0) + v(i,0);
    }
  }
  return z;
}

// k = x.y
double dotProduct(const Mtx &x, const Mtx &y)
{
  double k = 0.0;
  if( x.getNb_rows() != y.getNb_rows() || x.getNb_cols() != y.getNb_cols())
  {
    std::cerr << "Bad dimension for z = x + y" << std::endl;
    return k; 
  }
  int n = x.getAllocatedSize();
  for (int i = 0; i < n; i++)
  {
      k += x(i,0) * y(i,0);
  }
  return k;
}

// y = k.v
Mtx scaleV(double k, const Mtx &v)
{
  Mtx y(v.getNb_rows(), v.getNb_cols());
  
  int n = v.getAllocatedSize();
  #pragma omp parallel for num_threads(2)
  {
    for (int i = 0; i < n; i++)
    {
      y(i,0) = k * v(i,0);
    }
  }
  return y;
}

// k = ||x||
double norm(const Mtx &x)
{
  double k = 0.0;
  k = dotProduct(x,x);
  k = sqrt(k);
  return k;
}

double frobeniusNorm(const Mtx &A)
{
  double local_norm = 0.0;
  for (unsigned int i = A.getLower_id(); i <= A.getUpper_id(); i++)
  {
    for (unsigned int j = 0; j < A.getNb_cols(); j++)
    {
      local_norm += pow( fabs(A(i,j)), 2); 
    }
  }
  double global_norm = 0.0;
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Allreduce(&local_norm, &global_norm, comm_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global_norm;
}

// z = A.y + v
Mtx multYaddV(const Mtx &A, const Mtx &y, const Mtx &v)
{
  int send_count = A.getUpper_id() - A.getLower_id() + 1 ;
  double* send_buf = new double[send_count];
  double* recv_buf = new double[v.getNb_rows()];
  int* nb_data_rank = new int[v.getNb_rows()];
  int* shift = new int[v.getNb_rows()];
  
  for (unsigned int i = A.getLower_id(); i <= A.getUpper_id(); i++)
  {
    send_buf[i - A.getLower_id()] = v(i, (unsigned int) 0);
    for (unsigned int j = 0; j < A.getNb_cols(); j++)
    {
      send_buf[i - A.getLower_id()] += A(i,j) * y(j, (unsigned int) 0); 
    }
  }
  
  MPI_Allgather(&send_count, 1, MPI_INT, nb_data_rank, 1, MPI_INT, MPI_COMM_WORLD);
  
  shift[0] = 0;
  for (unsigned int i = 1; i < v.getNb_rows(); i++)
  {
    shift[i] = shift[i - 1] + nb_data_rank[i - 1];
  }
  MPI_Allgatherv(send_buf, send_count, MPI_DOUBLE, recv_buf, nb_data_rank, shift, MPI_DOUBLE, MPI_COMM_WORLD);
  
  Mtx z(v.getNb_rows(), v.getNb_cols());
  #pragma omp parallel for num_threads(2) 
  {
    for (unsigned int i = 0; i < v.getNb_rows(); i++)
    {
      z(i, (unsigned int) 0) = recv_buf[i];
    }
  }
  delete[] send_buf;
  delete[] recv_buf;
  delete[] nb_data_rank;
  delete[] shift;
  return z;
}

std::complex<double> complexVectorMultiply(const std::complex<double>**Y, const Mtx& Vm)
{
  int n = Vm.getAllocatedSize();
  std::complex<double> result = 0;
  std::complex<double>* V_m = new std::complex<double>[n*n];
  for (int i = 0; i < n; i++)
  { 
    for (int j = 0; j < n; j++)
    {
      V_m[i*n+j].real(Vm(i,j));
    }
  }
  result = blas::dotu(n,V_m,1,*Y,1);
  return result;
}

double summVect(const Mtx& vect)
{
  double summ = 0.0;
  int n = vect.getNb_rows();
  for (int i = 0; i < n; i++)
  {
    summ += vect(i,0);
  }
  return summ;
}


