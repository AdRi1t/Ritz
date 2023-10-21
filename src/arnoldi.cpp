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
  out->v_m = Mtx(input.n,1);

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
    v[j+1] = scaleV(1.0/h(j+1, 0), v[j+1]);
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
    out->v_m = v[j+1];
    out->h = h(j+1, 0);
  }
  return;
}

Mtx computeResiduals(double h_m, const Mtx& )
{
  Mtx residuals = Mtx(0,0);
  return residuals;
}
