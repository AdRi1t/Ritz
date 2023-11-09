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
  unsigned int l,c,m,s;
  l = config.getNb_lines();
  c = config.getNb_cols();
  m = config.getArnlodiDegree();
  s = config.getNb_eigen();
  ArnoldiInput input;
  ArnoldiOutput output;
  build_stats b_stats;
  std::complex<double>* eigenValues;
  std::complex<double>* eigenVectors;
  std::complex<double>* Us;
  std::complex<double>* mu;

  Mtx resultA3 = Mtx(l,1);
  Mtx residuals = Mtx(m,1);
  resultA3.fillResultA9(10,20);
  //resultA3.printValue();

  input.A = Mtx(l,c);
  input.v = Mtx(l,1);
  input.A.fillTestA3();
  input.v.fillConst(0.5);
  input.m = m;
  input.s = s;
  input.n = l;

  double t1,t2,t3,t4;
  double t;
  double* t_arnoldi = new double[config.getMax_iter()];
  double* t_iter = new double[config.getMax_iter()];
  plf::nanotimer timer;
  timer.start();
  double rho = 1.0;
  int iter = 0;
  //input.A.printValue();
  while( rho > config.getRelative_error() && iter < config.getMax_iter())
  {
  
    t1 = timer.get_elapsed_ms();
    reductionArnoldi(input, &output);
    computeEigen(output.H, &eigenValues, &eigenVectors);
    sortEigenValue(&eigenValues,&eigenVectors, &mu, input.m, input.s);
    Vm_QR(output.H ,mu, m-s);
    computeUs(&eigenVectors, output.V, &Us, input.s);
    residuals = computeResiduals2(output.h, &eigenVectors, input.m, input.s);
    rho = summVect(residuals);
    t2 = timer.get_elapsed_ms();
    t_iter[iter] = (t2 - t1);
    std::cout << "Iter : " << iter + 1  << "\n";
    std::cout << "Error : " << rho << "\n";
    input.v = newV(&Us, input.n, input.s);
    iter += 1;  
    
  }
  //std::cout << "Us : " << "\n";
  //printEigenVectors(Us,input.n,input.s);
  //std::cout << "values : " <<"\n";
  //printEigenValue(eigenValues,input.s);

  //b_stats.add_data("Arnoldi", "ms",t_arnoldi,input.m, iter); 
  b_stats.add_data("Iteration", "ms",t_iter,input.m, iter); 
  b_stats.writeAll(config.getBench_file_name());

  delete[] eigenValues;
  delete[] eigenVectors;
  delete[] Us;
  
  
  MPI_Finalize();

  return 0;
}

// std::cout << __LINE__ << std::endl;

    /*
    std::cout << "input v" << "\n";
    input.v.printValue();
    std::cout << "\n"; 

    t1 = timer.get_elapsed_ms();
    reductionArnoldi(input, &output);
    t2 = timer.get_elapsed_ms();
    t = (t2 - t1);
    //b_stats.add_data("Arnoldi", "ms",&t,input.m,1); 

    t1 = timer.get_elapsed_ms();
    computeEigen(output.H, &eigenValues, &eigenVectors);
    t2 = timer.get_elapsed_ms();
    t = (t2 - t1);
    //b_stats.add_data("Eigen", "ms",&t,input.m,1);

    t1 = timer.get_elapsed_ms();  
    sortEigenValue(&eigenValues,&eigenVectors, input.m, input.s);
    t2 = timer.get_elapsed_ms();
    t = (t2 - t1);
    //b_stats.add_data("Sort", "ms",&t,input.m,1); 

    std::cout << "A" << "\n";
    input.A.printValue();
    std::cout << "\n"; 

    std::cout << "H m" << "\n";
    output.H.printValue();
    std::cout << "\n"; 

    std::cout << "V m" << "\n";
    output.V.printValue();
    std::cout << "\n"; 

    std::cout << "output v" << "\n";
    output.v_m.printValue();
    std::cout << "\n"; 

    std::cout << "Eigen \n";
    printEigenVectors(eigenVectors,input.m, input.s);
    std::cout << "\n";
    printEigenValue(eigenValues, input.s);
    std::cout << "\n";

    t1 = timer.get_elapsed_ms();
    computeUs(&eigenVectors, output.V, &Us, input.s);
    t2 = timer.get_elapsed_ms();
    t = (t2 - t1);
    //b_stats.add_data("Us", "ms",&t,input.m,1); 
    
    std::cout << "Us "<< "\n";
    printEigenVectors(Us,input.n, input.s);
    std::cout << "\n";

    std::cout << "h "<< output.h <<"\n";

    //residuals = computeResiduals(input.A, &eigenValues , &Us, input.s, input.m);
    residuals = computeResiduals2(output.h, &eigenVectors, input.m, input.s);
    std::cout << "residuals" << "\n";
    residuals.printValue();
    std::cout << "\n";
    
    rho = summVect(residuals);
    std::cout << "Error : " << rho << "\n";
    
    input.v = newV(&Us, input.n, input.s);
    iter += 1;
    plf::millisecond_delay(200);

    */


   // ERAM

   /*
    t1 = timer.get_elapsed_ms();
    reductionArnoldi(input, &output);
    computeEigen(output.H, &eigenValues, &eigenVectors);
    sortEigenValue(&eigenValues,&eigenVectors, &mu, input.m, input.s);

    std::cout << "Eigen \n";
    printEigenValue(mu, m-s);
    std::cout << "\n";

    computeUs(&eigenVectors, output.V, &Us, input.s);
    residuals = computeResiduals2(output.h, &eigenVectors, input.m, input.s);
    rho = summVect(residuals);
    t2 = timer.get_elapsed_ms();
    t_iter[iter] = (t2 - t1);
    std::cout << "Iter : " << iter + 1  << "\n";
    std::cout << "Error : " << rho << "\n";
    input.v = newV(&Us, input.n, input.s);
    iter += 1;  
   */