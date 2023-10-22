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

  
  Mtx resultA3 = Mtx(l,1);
  resultA3.fillResultA9(10,20);
  resultA3.printValue();
  
  
  ArnoldiInput input;
  ArnoldiOutput output;
  std::complex<double>* eigenValues;
  std::complex<double>* eigenVectors;
  std::complex<double>* Us;
  input.A = Mtx(l,c);
  input.v = Mtx(l,1);
  input.A.fillTestA9(10,20);
  input.v.fillRandom(42);
  input.m = m;
  input.n = l;

  //input.A.printValue();
  
  reductionArnoldi(input, &output);
  
  computeEigen(output.H, &eigenValues, &eigenVectors);

  //output.H.printValue();
  //output.V.printValue();
  
  sortEigenValue(&eigenValues,&eigenVectors, m);
  printEigenValue(eigenValues,input.m);
  printEigenVectors(eigenVectors, input.m);
  std::cout << __LINE__ << std::endl;
  computeUs(&eigenVectors, output.V, &Us);
  std::cout << __LINE__ << std::endl;
  printEigenVectors(Us,input.n, input.m);

  // output.v_m = scaleV(output.h, output.v_m);
  //output.v_m.printValue();


  delete[] eigenValues;
  delete[] eigenVectors;
  delete[] Us;
  
  std::cout << __LINE__ << std::endl;
  
  MPI_Finalize();
  
  plf::millisecond_delay(100);
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