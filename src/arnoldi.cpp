#include <mpi.h>
#include <iostream>
#include "mat.hpp"
#include "arnoldiEnv.h"
#include "operation.hpp"
#include "../lib/BLAS/include/blas.hh"
#include "../lib/lapackpp/include/lapack.hh"


Mtx ArnoldiProjection(const Mtx &A, const Mtx &v_0, unsigned int m)
{
  Mtx* v = new Mtx[m + 1];
  Mtx v_result = Mtx(A.getNb_cols(), m);
  Mtx H_result = Mtx(m + 1 , m);
  v[0] = scaleV(1.0/norm(v_0), v_0);

  for (int j = 0; j < m; j++)
  {
    v[j+1] = multVect(A , v[j]);
    Mtx h = Mtx(A.getNb_cols() + 1, 1);
    for (int i = 0; i <= j ; i++)
    {
      h(i, 0) = dotProduct(v[j+1], v[i]);
      Mtx tmp = scaleV(-h(i, 0), v[i]);
      v[j+1] = addVect(v[j+1], tmp);
    }
    h(j+1, 0) = norm(v[j+1]);
    v[j+1] = scaleV(1.0/h(j+1, 0), v[j+1]);
    if(h(j+1,0) == 0.0)
    {
      break;
    }
    for (int i = 0; i <= j + 1 ; i++)
    {
      if(i >= H_result.getLower_id() && i <= H_result.getUpper_id())
      {
        H_result(i,j) = h(i,0); 
      }
    }
    for (int i = 0; i < A.getNb_rows() ; i++)
    {
      if(i >= v_result.getLower_id() && i <= v_result.getUpper_id())
      {
        v_result(i,j) = v[j](i,0); 
      }
    }
  }
  return H_result; 
}


void reductionArnoldi(const ArnoldiInput& input, ArnoldiOutput *out)
{
  Mtx* v = new Mtx[input.m + 1];
  out->V = Mtx(input.n, input.m);
  out->H = Mtx(input.m , input.m);
  out->v_m = Mtx(input.n, 1);

  v[0] = scaleV(1.0/norm(input.v), input.v);

  for (int j = 0; j < input.m; j++)
  {
    v[j+1] = multVect(input.A , v[j]);
    Mtx h = Mtx(input.n + 1, 1);
    for (int i = 0; i <= j ; i++)
    {
      h(i, 0) = dotProduct(v[j+1], v[i]);
      Mtx tmp = scaleV(-h(i, 0), v[i]);
      v[j+1] = addVect(v[j+1], tmp);
    }
    h(j+1, 0) = norm(v[j+1]);
    v[j+1] = scaleV((1.0/h(j+1, 0)), v[j+1]);
    if(h(j+1,0) == 0.0)
    {
      break;
    }
    for (int i = 0; i <= j + 1 ; i++)
    {
      if(i >= out->H.getLower_id() && i <= out->H.getUpper_id())
      {
        out->H(i,j) = h(i,0); 
      }
    }
    for (int i = 0; i < input.n ; i++)
    {
      if(i >= out->V.getLower_id() && i <= out->V.getUpper_id())
      {
        out->V(i,j) = v[j](i,0); 
      }
    }
    out->v_m = v[input.m];
    out->h = h(input.m, 0);
  }
  return;
}

Mtx computeResiduals(const Mtx& mtx_A, std::complex<double> **eigenValues, std::complex<double> **eigenVectors, int s, int m)
{
  assert(s > 0);
  assert(m >= s);
  int n = mtx_A.getNb_rows();
  int comm_size = 0;
  int comm_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  Mtx residuals = Mtx(s, 1);
  std::complex<double>* A = new std::complex<double>[n*n];

  int nb_own_values  = mtx_A.getAllocatedSize();
  double* own_buffer = new double[nb_own_values];
  int k = 0;
  for (unsigned int i = mtx_A.getLower_id(); i <= mtx_A.getUpper_id(); i++)
  {
    for (unsigned int j = 0; j < n; j++)
    {
      own_buffer[k] = mtx_A(i,j); 
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
      upper_id = mtx_A.getUpperRank(i);
    }
    else
    {
      lower_id = mtx_A.getUpperRank(i-1);
      upper_id = mtx_A.getUpperRank(i);
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
  std::complex<double> alpha(1.0 , 0.0);
  for (int i = 0; i < s; i++)
  {
    std::complex<double> beta = -(*eigenValues)[i];
    std::complex<double>* vector = new std::complex<double>[n];
    blas::copy(n, &((*eigenVectors)[i]), s, vector, 1);
    blas::gemv(blas::Layout::RowMajor, blas::Op::NoTrans, n, n, alpha, A, n, vector, 1, beta, vector, 1);
    residuals(i,0) = blas::nrm2(n, vector, 1);
    delete[] vector;
  }
  delete[] A;
  return residuals;
}

Mtx computeResiduals2(double hm1, std::complex<double> **eigenVectors, int m, int s)
{
  Mtx residuals = Mtx(s,1);
  for(int i = 0 ; i < s ; i++)
  {
    residuals(i,0) = hm1 * abs((*eigenVectors)[m*(s-1) + i].real());
  }
  return residuals;
}

Mtx newV(std::complex<double> **EigenVectors, int n, int s)
{
  Mtx v = Mtx(n,1);
  double* norm = new double[s];
  for(int j = 0; j < s; j++)
  {
    norm[j] = 0.0;
    for(int i = 0; i < n; i++)
    {
      norm[j] += pow(std::norm((*EigenVectors)[i*s+j]),2);
    }
    norm[j] = sqrt(norm[j]);
  }
  
  for(int i = 0; i < n; i++)
  {
    v(i,0) = 0;
    for(int j = 0; j < s; j++)
    {
      v(i,0) +=  (*EigenVectors)[i*s+j].real();
    }
  } 
  /*
  for(int i = 0; i < n; i++)
  {
    v(i,0) = (*EigenVectors)[i*s].real();
  }
  */
  delete[] norm;
  return v;
}

