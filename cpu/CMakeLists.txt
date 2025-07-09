cmake_minimum_required(VERSION 3.10)

# Nombre del proyecto
project(BPE)

# Buscar OpenMP
find_package(OpenMP REQUIRED)

# Establecer el estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Agregar el ejecutable principal
add_executable(bpe bpe.cpp)

# Linkear con OpenMP
target_link_libraries(bpe PRIVATE OpenMP::OpenMP_CXX)
