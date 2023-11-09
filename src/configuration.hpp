#include <fstream>
#include <string>
#pragma once

// Balise pour le fichier de configuration xml
#define TAG_TITLE "title"
#define TAG_BASE "ritz-config"
#define TAG_OUT_FILE "out-file"
#define TAG_MAX_ITER "max-iter"
#define TAG_NB_COLS "matrix-colums"
#define TAG_NB_LINES "matrix-lines"
#define TAG_RELATIVE_ERROR "relative-error"
#define TAG_ARNOLDI_SIZE "arnoldi-size"
#define TAG_NB_EIGEN "s"
#define TAG_MATRIX_FILE "matrix-file"
#define TAG_BENCHMARK "benchmark"
#define TAG_MEASURE_ITER "measure-iter"
#define TAG_VERBOSE_LEVEL "verbose-level"

class Configuration
{
private:
  bool dump_enable;
  bool make_benchmark;
  bool matrix_source_file;
  int verbose_level;
  int matrix_nb_lines;
  int matrix_nb_cols;
  int max_iter;
  int measure_iter;
  int arnoldi_degree;
  int nb_eigen;
  double relative_error;
  std::string matrix_file_name;
  std::string title;
  std::string data_file;

  void parseFile(std::string file_name);

public:
  Configuration();
  ~Configuration();
  bool configIsOK();
  void printConfig();
  void parseCommand(int argc, char* argv[]);
  bool getMake_benchmark() const;
  bool haveMtxFile() const;
  double getRelative_error() const;
  int getNb_lines() const;
  int getNb_cols() const;
  int getMax_iter() const;
  int getMeasure_iter() const;
  int getVerboseLevel() const;
  int getArnlodiDegree() const;
  int getNb_eigen() const;
  std::string getBench_file_name() const;
  std::string getMatrix_file_name() const;
};

void printUsage();

