cmake_minimum_required(VERSION 3.10)
project(torchtest VERSION 0.1
        DESCRIPTION "Testing making PyTorch C++ extensions"
        LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Read variables from file
file(STRINGS dir_info.txt ConfigContents)
foreach(KEYVALUE ${ConfigContents})
    string(REGEX MATCH "^[^=]+" KEY ${KEYVALUE})
    string(REPLACE "${KEY}=" "" VALUE ${KEYVALUE})
    if(DEFINED ${KEY})
        message(FATAL_ERROR "${KEY} already exists")
    else()
        set(${KEY} "${VALUE}")
    endif()
endforeach()

# file should define LIBTORCH and PYTHON
if(NOT DEFINED LIBTORCH)
    message(FATAL_ERROR "LIBTORCH key not defined in dir_info.txt. Should specify where to find LibTorch; see https://pytorch.org/cppdocs/installing.html")
endif()
if(NOT DEFINED PYTHON)
    message(FATAL_ERROR "PYTHON key not defined in dir_info.txt. Should specify where to find the Python header files; e.g. /usr/include/python3.6")
endif()

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${LIBTORCH}")
find_package(Torch REQUIRED)

add_executable(torchtest main.cpp)
target_link_libraries(torchtest PUBLIC "${TORCH_LIBRARIES}")
target_include_directories(torchtest PUBLIC "${PYTHON}")