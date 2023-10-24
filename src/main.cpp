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
  Configuration config;
  config.parseCommand(argc, argv);
  if(config.configIsOK() == false)
  {
    std::cerr << "\nCan't continue\n";
    exit(0);
  }
  if(config.getVerboseLevel() > 1)
  {
   config.printConfig();
  }
  MPI_Init(&argc, &argv);
  unsigned int l,c,m;
  l = config.getNb_lines();
  c = config.getNb_cols();
  m = config.getArnlodiDegree();

  ArnoldiInput input;
  ArnoldiOutput output;
  build_stats b_stats;
  std::complex<double>* eigenValues;
  std::complex<double>* eigenVectors;
  std::complex<double>* Us;

  Mtx resultA3 = Mtx(l,1);
  Mtx residuals;
  resultA3.fillResultA9(10,20);
  //resultA3.printValue();

  input.A = Mtx(l,c);
  input.v = Mtx(l,1);
  input.A.fillTestA3();
  input.v.fillRandom(42);
  input.m = m;
  input.n = l;

  double t1,t2;
  double t;
  plf::nanotimer timer;
  timer.start();

  t1 = timer.get_elapsed_ms();
  reductionArnoldi(input, &output);
  t2 = timer.get_elapsed_ms();
  t = (t2 - t1);
  b_stats.add_data("Arnoldi", "ms",&t,input.m,1); 

  t1 = timer.get_elapsed_ms();
  computeEigen(output.H, &eigenValues, &eigenVectors);
  t2 = timer.get_elapsed_ms();
  t = (t2 - t1);
  b_stats.add_data("Eigen", "ms",&t,input.m,1);

  t1 = timer.get_elapsed_ms();  
  sortEigenValue(&eigenValues,&eigenVectors, m);
  t2 = timer.get_elapsed_ms();
  t = (t2 - t1);
  b_stats.add_data("Sort", "ms",&t,input.m,1); 

  t1 = timer.get_elapsed_ms();
  computeUs(&eigenVectors, output.V, &Us);
  t2 = timer.get_elapsed_ms();
  t = (t2 - t1);
  b_stats.add_data("Us", "ms",&t,input.m,1); 

  residuals = computeResiduals(input.A, &eigenValues, &eigenVectors, input.m, input.m);
  // output.v_m = scaleV(output.h, output.v_m);
  
  /*
  output.H.printValue();
  output.V.printValue();
  residuals.printValue();
  printEigenVectors(Us, input.n,input.m);
  std::cout << "h_m : " << output.h << "\n";
  printEigenValue(eigenValues, input.m);
    std::cout << "h_m : " << output.h << "\n";
  printEigenVectors(eigenVectors,input.m);
  */
  
  b_stats.writeAll(config.getBench_file_name());

  delete[] eigenValues;
  delete[] eigenVectors;
  delete[] Us;
  
  
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