#pragma once
#include "arnoldiEnv.h"
#include <complex.h>

// y = A.v
Mtx multVect(const Mtx& A, const Mtx& v);

// z = y + v
Mtx addVect(const Mtx &y, const Mtx &v);

// z = A.y + v
Mtx multYaddV(const Mtx &A, const Mtx &y, const Mtx &v);

// k = x.y
double dotProduct(const Mtx &x, const Mtx &y);

// y = k.v
Mtx scaleV(double k, const Mtx &v);

// k = ||x||
double norm(const Mtx &x);

double frobeniusNorm(const Mtx &A);

Mtx ArnoldiProjection(const Mtx &A, const Mtx &v, unsigned int m);

// Si j'instenci des matrices puis que je les copies dans la structure ArnoldiOutput
// elle sont détruite à la fin de cette fonction.
void reductionArnoldi(const ArnoldiInput& input, ArnoldiOutput *out);

void computeEigen(const Mtx& H, std::complex<double>** eigenValue, std::complex<double>** eigenVectors);

void sortEigenValue(std::complex<double>** eigenValue, std::complex<double>** eigenVectors, int n);

void printEigenValue(const std::complex<double> *eigenValues, int n);

void printEigenVectors(const std::complex<double> *eigenVectors, int n);
void printEigenVectors(const std::complex<double> *eigenVectors, int n, int m);

void computeUs(std::complex<double> **Y , const Mtx& Vm, std::complex<double> **U);

