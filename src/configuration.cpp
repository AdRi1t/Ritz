#include <exception>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <regex>
#include <filesystem>
#include "../lib/tinyxml2/tinyxml2.h"
#include "configuration.hpp"

Configuration::Configuration()
{
  dump_enable = true;
  make_benchmark = false;
  matrix_source_file = false;
  verbose_level = 0;
  matrix_nb_lines = 0;
  matrix_nb_cols = 0;
  max_iter = 100;
  measure_iter = 10;
  relative_error = 1e-10;
  matrix_file_name = std::string();
  title = std::string();
  data_file = std::string();
}

Configuration::~Configuration() { }


void Configuration::parseCommand(int argc, char* argv[])
{
  extern char* optarg;
  int opt;
  if(argc == 1)
  {
    printUsage();
    exit(EXIT_SUCCESS);
  }
  else if(argc >= 2)
  {
    try
    {
      this->parseFile(argv[1]);
    }
    catch(const std::exception& e)
    {
      std::cerr << e.what() << '\n';
    }
  }
  while((opt = getopt(argc, argv, "hrdl:c:f:s:i:e:m:")) != -1)
  {
    switch(opt)
    {
      case 'h':
        printUsage();
        exit(EXIT_SUCCESS);
        exit(0);
        break;
      case 'l':
        this->matrix_nb_lines = atoi(optarg);
        break;
      case 'c':
        this->matrix_nb_cols = atoi(optarg);
        break;
      case 'd':
        this->dump_enable = true;
        break;
      case 'm':
        this->arnoldi_degree = atoi(optarg);
        break;
      case 'i':
        this->max_iter = atoi(optarg);
        break;
      case 'e':
        this->relative_error = atof(optarg);
        break;
      case 'r':
        this->make_benchmark = true;
        break;
      case 'f':
        this->matrix_source_file = true;
        this->matrix_file_name.assign(optarg);
        break;
    }
  }
  return;
}

void Configuration::parseFile(std::string file_name)
{
  tinyxml2::XMLError error;
  tinyxml2::XMLDocument xml_document;
  error = xml_document.LoadFile(file_name.c_str());
  if(error != tinyxml2::XML_SUCCESS)
  {
    std::cout << "No file to parse" << std::endl;
    return;
  }
  
  tinyxml2::XMLElement* BASE_element =  xml_document.FirstChildElement(TAG_BASE);
  
  if(BASE_element == nullptr)
  {
    std::string error("Missing root tag in XML FILE : ");
    error.append(TAG_BASE);
    throw std::runtime_error(error);
  }

  tinyxml2::XMLElement* out_file_element = BASE_element->FirstChildElement(TAG_OUT_FILE);
  tinyxml2::XMLElement* title_element = BASE_element->FirstChildElement(TAG_TITLE);
  tinyxml2::XMLElement* matrix_file_element = BASE_element->FirstChildElement(TAG_MATRIX_FILE);
  tinyxml2::XMLElement* nb_lines_element = BASE_element->FirstChildElement(TAG_NB_LINES);
  tinyxml2::XMLElement* nb_cols_element = BASE_element->FirstChildElement(TAG_NB_COLS);
  tinyxml2::XMLElement* max_iter_element = BASE_element->FirstChildElement(TAG_MAX_ITER);
  tinyxml2::XMLElement* arnoldi_size_element = BASE_element->FirstChildElement(TAG_ARNOLDI_SIZE);
  tinyxml2::XMLElement* relative_error_element = BASE_element->FirstChildElement(TAG_RELATIVE_ERROR);
  tinyxml2::XMLElement* make_benchmark_element = BASE_element->FirstChildElement(TAG_BENCHMARK);
  tinyxml2::XMLElement* measure_iter_element = BASE_element->FirstChildElement(TAG_MEASURE_ITER);
  tinyxml2::XMLElement* verbose_element = BASE_element->FirstChildElement(TAG_VERBOSE_LEVEL);

  //checkPath();
  if(out_file_element != nullptr)
  {
    data_file = std::string(out_file_element->GetText());
  }
  else
  {
    FILE* test = fopen("result/data_file", "a");
    if(test == NULL)
    {
      std::cerr << "Ne peut pas ouvrir de fichier de sortie" << "\n";
    }
    else
    {
      fclose(test);
    }
  }
  if(title_element != nullptr)
  {
    title = std::string(title_element->GetText());
  }
  if(matrix_file_element != nullptr)
  {
    matrix_file_name = std::string(matrix_file_element->GetText());
    matrix_source_file = true;
  }
  if(nb_lines_element != nullptr)
  {
    nb_lines_element->QueryIntText(&matrix_nb_lines);
  }
  if(nb_cols_element != nullptr)
  {
    nb_cols_element->QueryIntText(&matrix_nb_cols);
  }
  if(arnoldi_size_element != nullptr)
  {
    arnoldi_size_element->QueryIntText(&arnoldi_degree);
  }
  if(relative_error_element != nullptr)
  {
    relative_error_element->QueryDoubleText(&relative_error);
  }
  if(max_iter_element != nullptr)
  {
    max_iter_element->QueryIntText(&max_iter);
  }
  if(make_benchmark_element != nullptr)
  {
    std::string bench = std::string(make_benchmark_element->GetText());
    if(bench.compare("yes") == 0 || bench.compare("1") == 0)
    {
      make_benchmark = true;
    }
  }
  if(measure_iter_element != nullptr)
  {
    measure_iter_element->QueryIntText(&measure_iter);
  }
  if(verbose_element != nullptr)
  {
    verbose_element->QueryIntText(&verbose_level);
  }
}

bool Configuration::getMake_benchmark() const { return make_benchmark; }

bool Configuration::haveMtxFile() const { return matrix_source_file; }

double Configuration::getRelative_error() const { return relative_error; }

int Configuration::getNb_lines() const { return matrix_nb_lines; }

std::string Configuration::getMatrix_file_name() const { return matrix_file_name; }

int Configuration::getNb_cols() const { return matrix_nb_cols; }

int Configuration::getMax_iter() const { return max_iter; }

int Configuration::getMeasure_iter() const { return measure_iter; }

int Configuration::getVerboseLevel() const { return verbose_level; }

int Configuration::getArnlodiDegree() const { return arnoldi_degree; }

std::string Configuration::getBench_file_name() const { return data_file; }

bool Configuration::configIsOK()
{
  if(matrix_source_file == false && (matrix_nb_cols == 0 || matrix_nb_lines == 0 ))
  {
    std::cerr << "\nIncorrect matrix size\n";
    return false;
  }
  if(arnoldi_degree > matrix_nb_lines)
  {
    std::cerr << "\nIncorrect arnoldi degree \n";
    return false;
  }
  if(relative_error == 0.0)
  {
    std::cerr << "\nMust specify a relative error\n";
    return false;
  }
  if(FILE* file = fopen(matrix_file_name.c_str(), "r"))
  {
    fclose(file);
  }
  else
  {
    if(matrix_source_file == true)
    {
      std::cerr << "\nCannot open : " << matrix_file_name << "\n";
      return false;
    }
  }
  if(verbose_level < 0 || verbose_level > 2)
  {
    std::cerr << "\nBad verbose level must be 0,1 or 2\n";
    return false;
  }
  auto WorkingDir = std::filesystem::current_path();
  auto resultDir = WorkingDir.append("result");
  if(!std::filesystem::is_directory(resultDir))
  {
    std::filesystem::create_directory(resultDir);
  }

  const std::regex txt_regex("result/.*");
  if(std::regex_match(data_file, txt_regex) == false)
  {
    data_file = std::string("result/" + data_file);
  }
  return true;
}

void Configuration::printConfig()
{
  std::cout << "\n";
  std::cout.width(24);
  std::cout << std::left << "Title : " << std::right << title << std::endl;
  std::cout.width(24);
  std::cout << std::left << "Matrix nb lines : " << std::right << matrix_nb_lines << std::endl;
  std::cout.width(24);
  std::cout << std::left << "Matrix nb cols : " << std::right << matrix_nb_cols << std::endl;
  std::cout.width(24);
  std::cout << std::left << "Matrix file name : " << std::right << matrix_file_name
            << std::endl;
  std::cout.width(24);
  std::cout << std::left << "Arnoldi degree : " << std::right << arnoldi_degree
            << std::endl;
  std::cout.width(24);
  std::cout << std::left << "Maximum iterations : " << std::right << max_iter
            << std::endl;
  std::cout.width(24);
  std::cout << std::left << "Relative error : " << std::right << relative_error
            << std::endl;
  std::cout.width(24);
  std::cout << std::left << "Out data file : " << std::right << data_file << std::endl;
  std::cout.width(24);
  std::cout << std::left << "Verbose level : " << std::right << verbose_level
            << std::endl;
}

void printUsage()
{
  std::cout << "Arnoldi [file.xml] -l[nb lines] -c[nb cols]" << "\n";
}
