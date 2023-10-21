#include <iostream>
#include <vector>
#include <mpi.h>
#include <complex.h>
#include "plf_nanotimer.h"
#include "configuration.hpp"
#include "mat.hpp"
#include "arnoldiEnv.h"
#include "operation.hpp"
#include "measure_utils.hpp"

int main(int argc, char *argv[])
{
  parseCommand(argc, argv);
  if(getConfig()->configIsOK() == false)
  {
    std::cerr << "\nCan't continue\n";
    exit(0);
  }
  if(getConfig()->getVerboseLevel() > 1)
  {
    getConfig()->printConfig();
  }
  MPI_Init(&argc, &argv);
  unsigned int l,c,m;
  l = getConfig()->getNb_lines();
  c = getConfig()->getNb_cols();
  m = getConfig()->getArnlodiDegree();

  Mtx resultA9 = Mtx(l,1);
  resultA9.fillResultA9(10, 100);
  
  //mat_A.printValue();

  ArnoldiInput input;
  ArnoldiOutput output;
  std::complex<double>* eigenValues;
  std::complex<double>* eigenVectors;
  input.A = Mtx(l,c);
  input.v = Mtx(l,1);
  input.A.fillRandom(10);
  input.v.fillRandom(42);
  input.m = m;
  input.n = l;

  std::cout << __LINE__ << std::endl;

  reductionArnoldi(input, &output);
  
    std::cout << __LINE__ << std::endl;

  computeEigen(output.H, &eigenValues, &eigenVectors);
  sortEigenValue(&eigenValues,&eigenVectors,m);
    std::cout << __LINE__ << std::endl;
  output.H.printValue();
  output.V.printValue();
  for (size_t i = 0; i < input.m; i++)
  {
    std::cout << "Valeurs propre " << eigenValues[i] << "\n";
    for (size_t j = 0; j < input.m; j++)
    {
     // std::cout << eigenVectors[i * input.m + j] << " "; 
    }
    std::cout << "\n" ;
  }
  // output.v_m = scaleV(output.h, output.v_m);
  //output.v_m.printValue();
  plf::millisecond_delay(100);

  MPI_Finalize();
  
  return 0;
}

// std::cout << __LINE__ << std::endl;

/*
  Mtx mat_A = Mtx(n,m);
  Mtx vect_x = Mtx(m,1);
  mat_A.fillRandom(42);
  vect_x.fillRandom(42);
  Mtx result_y = multVect(mat_A, vect_x);
  Mtx result_z = addVect(result_y, vect_x);
  Mtx vect_result = multYaddV(mat_A, result_y, result_z);
  
  std::cout << "Norme vect : " << norm(vect_result) << "\n";
  std::cout << "Frobenius : " << frobeniusNorm(mat_A) << "\n";

*/