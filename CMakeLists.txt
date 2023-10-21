cmake_minimum_required(VERSION 3.20.0)

project(ritz)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(flag -Wall -g3 -O0 -march=native -mtune=native -ftree-vectorize -funroll-loops)

set(SRCS
    lib/mmio/mmio.c
    lib/tinyxml2/tinyxml2.cpp
    src/configuration.cpp
    src/measure_utils.cpp
    src/mat.cpp
    src/operation.cpp
    src/arnoldi.cpp
    src/eigen.cpp
    src/main.cpp
    )
    
set(HEADERS
    lib/mmio/mmio.h
    lib/tinyxml2/tinyxml2.h
    src/plf_nanotimer.h
    src/mat.hpp
    src/configuration.hpp
    src/measure_utils.hpp
    src/arnoldiEnv.h
    src/operation.hpp
    )

add_executable(ritz ${SRCS} ${HEADERS})

find_package(MPI REQUIRED)
target_link_libraries(ritz MPI::MPI_CXX)

add_compile_options(${flag})
target_compile_options(ritz PUBLIC ${flag})

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++20")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -llapacke -llapack -lblas")
target_link_libraries(ritz OpenMP::OpenMP_CXX)
target_include_directories(ritz PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})

target_include_directories(ritz PUBLIC ${CMAKE_SOURCE_DIR}/lib/lapackpp/include)
target_link_libraries (ritz ${CMAKE_SOURCE_DIR}/lib/lapackpp/lib/liblapackpp.a)

target_include_directories(ritz PUBLIC ${CMAKE_SOURCE_DIR}/lib/BLAS/include)
target_link_libraries (ritz ${CMAKE_SOURCE_DIR}/lib/BLAS/lib/libblaspp.a)


find_package(BLAS REQUIRED)
target_link_libraries(ritz ${BLAS_LIB})
target_link_libraries(ritz BLAS::BLAS) 
target_include_directories(ritz PUBLIC ${BLAS_LIB_DIR})




