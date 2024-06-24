#
# Non-metric Space Library
#
# Authors: Bilegsaikhan Naidan, Leonid Boytsov.
#
# This code is released under the
# Apache License Version 2.0 http://www.apache.org/licenses/.
#
#

cmake_minimum_required (VERSION 2.8)

project (NonMetricSpaceLib)

#
# Runs compiler with "-dumpversion" and parses major/minor
# version with a regex.
#
# Taken&Modified from Boost.cmake
#
function(CXX_COMPILER_DUMPVERSION _OUTPUT_VERSION)

  exec_program(${CMAKE_CXX_COMPILER}
    ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
    OUTPUT_VARIABLE COMPILER_VERSION
  )
  #string(REGEX REPLACE "([0-9])\\.([0-9])(\\.[0-9])?" "\\1\\2"
  #   COMPILER_VERSION ${COMPILER_VERSION})

  set(${_OUTPUT_VERSION} ${COMPILER_VERSION} PARENT_SCOPE)
endfunction()

if(NOT WIN32)
    CXX_COMPILER_DUMPVERSION(CXX_COMPILER_VERSION)
endif()
#message(FATAL_ERROR "stopping... compiler version is: ${CMAKE_CXX_COMPILER_ID} ${CXX_COMPILER_VERSION}")

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc|powerpc")
    set(SIMD_FLAGS " -mtune=native")
else()
    set(SIMD_FLAGS " -march=native")
endif()
#set(SIMD_FLAGS "-march=x86-64")
#set(SIMD_FLAGS "-march=core2")
#set(SIMD_FLAGS "-fpic -msse4.2")

set(WARN_FLAGS "-Wall -Wunreachable-code -Wcast-align") 

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # require at least gcc 4.7
    if (CXX_COMPILER_VERSION VERSION_LESS 4.7)
        message(FATAL_ERROR "GCC version must be at least 4.7!")
    endif()
    set (CMAKE_CXX_FLAGS_RELEASE "${WARN_FLAGS} -Ofast -lm -DNDEBUG -std=c++11 -pthread  -DHAVE_CXX0X ${SIMD_FLAGS} -Wl,--no-as-needed -fpic")
    set (CMAKE_CXX_FLAGS_DEBUG   "${WARN_FLAGS} -ggdb  -lm -DNDEBUG -std=c++11 -pthread  -DHAVE_CXX0X ${SIMD_FLAGS} -Wl,--no-as-needed -fpic")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    if (CXX_COMPILER_VERSION VERSION_LESS 14.0.1)
        message(FATAL_ERROR "Intel version must be at least 14.0.1!")
    endif()
    set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wunreachable-code -Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -pthread ${SIMD_FLAGS} -fpic")
    set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wunreachable-code -ggdb  -DNDEBUG -std=c++11 -DHAVE_CXX0X -pthread ${SIMD_FLAGS} -fpic")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    if (CMAKE_SYSTEM_NAME MATCHES Darwin)
        # MACOSX
        set (CMAKE_CXX_FLAGS_RELEASE "${WARN_FLAGS} -O3 -DNDEBUG -std=c++11 -DHAVE_CXX0X -pthread -fpic ${SIMD_FLAGS}")
        set (CMAKE_CXX_FLAGS_DEBUG   "${WARN_FLAGS} -ggdb  -DNDEBUG -std=c++11 -DHAVE_CXX0X -pthread -fpic ${SIMD_FLAGS}")
    else()
        set (CMAKE_CXX_FLAGS_RELEASE "${WARN_FLAGS} -O3 -DNDEBUG -std=c++11 -DHAVE_CXX0X -pthread ${SIMD_FLAGS} -Wl,--no-as-needed -fpic")
        set (CMAKE_CXX_FLAGS_DEBUG   "${WARN_FLAGS} -ggdb  -DNDEBUG -std=c++11 -DHAVE_CXX0X -pthread ${SIMD_FLAGS} -Wl,--no-as-needed -fpic")
    endif()
elseif(WIN32)
    # TODO how can we add support for later versions? There seems to be no flag with the semantics "version x or higher"
    if(MSVC_VERSION LESS 1800) 
         message(FATAL_ERROR "On Windows, only MSVC version should be >= 12, code 1800, current version ${MSVC_VERSION}!") 
    endif()
else ()
    message(FATAL_ERROR "Unrecognized compiler (use GCC, Clang, Intel compiler, or MSVC (on Windows)!")
endif()

if (WITH_EXTRAS)
    message(STATUS "******************************")
    message(STATUS "Will build with extra stuff...")
    message(STATUS "******************************")
    add_definitions (-DWITH_EXTRAS=1)
endif()

if (WIN32)
    # With MSVC build types are useless, it's all handled by MSVC itself,
    # which creates build-specific output folders. However, they will all
    # be placed inside the debug subfolder.
    # To avoid this, we set these output variables.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else()
    if (NOT CMAKE_BUILD_TYPE)
        set (CMAKE_BUILD_TYPE Release CACHE STRING "Build types: Release Debug" FORCE)
    endif (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if (WITH_EXTRAS)
  find_package(Eigen3 3 REQUIRED)
  if (EIGEN3_FOUND)
    message (STATUS "Found Eigen3.")
    include_directories (${EIGEN3_INCLUDE_DIR})
  else ()
    message (FATAL_ERROR "Could not locate Eigen3.")
  endif ()
endif()

add_subdirectory (src)
add_subdirectory (apps)
if (NOT WITHOUT_TESTS)
    add_subdirectory (test)
endif()

message(STATUS "OS:                ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CXX_COMPILER_VERSION}")
message(STATUS "CXX flags release: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CXX flags debug:   ${CMAKE_CXX_FLAGS_DEBUG}")
