#ifndef ARNOLDI_ENV_H
#define ARNOLDI_ENV_H

#include "mat.hpp"

struct ArnoldiInput
{
  Mtx A;
  Mtx v;
  int n;
  int m;
};

struct ArnoldiOutput
{
  Mtx V;
  Mtx H;
  double h;
  Mtx v_m;
};

#endif