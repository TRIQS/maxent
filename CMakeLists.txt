if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} -------------")

set (MAXENT_VERSION "1.1.0")

# start configuration
cmake_minimum_required(VERSION 2.8)
# We use TRIQS by default
option(USE_TRIQS "Use TRIQS" ON)

# Load TRIQS, including all predefined variables from TRIQS installation
if(TRIQS_PATH) # TRIQS 1.4
  if(NOT ${USE_TRIQS})
      message(FATAL_ERROR "Either use TRIQS_PATH or USE_TRIQS=OFF, not both.")
  endif()
  project(maxent CXX)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  # First append triqs installed files to the cmake load path
  message(STATUS "--------- Installing as TRIQS 1.4 package ---------")
  list(APPEND CMAKE_MODULE_PATH ${TRIQS_PATH}/share/triqs/cmake)
  find_package(TRIQS REQUIRED)
  set(TRIQS_ROOT ${TRIQS_PATH})
  set(TRIQS_V1 ON)
  set(TRIQS_V2 OFF)
  set(python_destination triqs_maxent)
  set(PYTHON_LIB_DEST ${TRIQS_PYTHON_LIB_DEST_ROOT}/${python_destination})
elseif(${USE_TRIQS}) # search for TRIQS 3.1 - if not found default to USE_TRIQS=OFF
  # Try to find TRIQS and Cpp2Py
  project(maxent CXX)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  find_package(TRIQS 3.1 QUIET)
  if (TRIQS_FOUND)
	message(STATUS "--------- Installing as TRIQS ${TRIQS_VERSION} package ---------")
	set(TRIQS_V1 OFF)
	set(TRIQS_V2 ON)
	set(PYTHON_LIB_DEST ${CPP2PY_PYTHON_LIB_DEST_ROOT}/triqs_maxent)
  else()
	message(STATUS "Cpp2Py / TRIQS not found.")
	message(STATUS "Defaulting back to installation without TRIQS")
	set(USE_TRIQS OFF)
  endif()
else()
  # Handle USE_TRIQS= blank case
  set(USE_TRIQS OFF)
endif()

if(NOT ${USE_TRIQS})
	message(STATUS "--------- Installing WITHOUT TRIQS ---------")
	project(maxent NONE)
	set(TRIQS_V1 OFF)
	set(TRIQS_V2 OFF)
endif()

if(${USE_TRIQS})
    if (NOT ${TRIQS_WITH_PYTHON_SUPPORT})
      MESSAGE(FATAL_ERROR "maxent requires python support enabled in TRIQS")
    endif()
endif()

# We want to be installed in the TRIQS tree
# Default Install directory to TRIQS_ROOT if not given. Checks an absolute name is given.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  if(NOT ${USE_TRIQS})
      message(STATUS "No CMAKE_INSTALL_PREFIX given.")
      set(CMAKE_INSTALL_PREFIX /usr CACHE PATH "maxent install path" FORCE)
  else()
    message(STATUS " No install prefix given (or invalid). Defaulting to TRIQS_ROOT")
    set(CMAKE_INSTALL_PREFIX ${TRIQS_ROOT} CACHE PATH "default install path" FORCE)
  endif()
endif()
message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} -------------")
if(NOT ${USE_TRIQS})
    set(PYTHON_LIB_DEST "lib/python2.7/site-packages/triqs_maxent" CACHE PATH "where to install the python library (path relative to CMAKE_INSTALL_PREFIX)")
endif()
set(DOC_DEST "share/doc/maxent" CACHE PATH "where to install the doc (path relative to CMAKE_INSTALL_PREFIX)")
set(BIN_DEST "bin" CACHE PATH "where to install the bin files (path relative to CMAKE_INSTALL_PREFIX)")

message(STATUS "'make install' will install maxent as python package to: ${CMAKE_INSTALL_PREFIX}/${PYTHON_LIB_DEST}")
message(STATUS "'make install' will install bin files to: ${CMAKE_INSTALL_PREFIX}/${BIN_DEST}")

if(${TRIQS_V1})
  set(TRIQS_PATH "${CMAKE_INSTALL_PREFIX}/bin/triqs")
endif()

if(NOT ${USE_TRIQS})
  find_package(Git)
  if(GIT_FOUND)
      message(STATUS "git found: ${GIT_EXECUTABLE}")
      execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
          WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
          RESULT_VARIABLE   GIT_RESULT
          OUTPUT_VARIABLE   "MAXENT_GIT_HASH"
          ERROR_VARIABLE    GIT_ERROR
          OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
elseif(${TRIQS_V1})
  triqs_get_git_hash(${CMAKE_SOURCE_DIR} "MAXENT")
elseif(${TRIQS_V2})
  triqs_get_git_hash_of_source_dir(MAXENT_GIT_HASH)
endif()

message(STATUS "Git hash: ${MAXENT_GIT_HASH}")

add_subdirectory(python)
add_subdirectory(shells)

option(TEST_COVERAGE "Analyze the coverage of tests" OFF)

# perform tests with coverage info
if (${TEST_COVERAGE})
  # we try to locate the coverage program
  find_program(PYTHON_COVERAGE python-coverage)
  find_program(PYTHON_COVERAGE coverage)
  if(NOT PYTHON_COVERAGE)
      message(FATAL_ERROR "Program coverage (or python-coverage) not found.\nEither set PYTHON_COVERAGE explicitly or disable TEST_COVERAGE!\nYou need to install the python package coverage, e.g. with\n  pip install coverage\nor with\n  apt install python-coverage")
  endif()

  message(STATUS "Setting up test coverage")
  if(${TRIQS_V1})
    set(TRIQS_PATH "${CMAKE_BINARY_DIR}/build_triqs")
    file(READ ${CMAKE_SOURCE_DIR}/shells/triqs-coverage.sh.in BUILD_TRIQS_COVERAGE_SCRIPT)
    string(CONFIGURE "${BUILD_TRIQS_COVERAGE_SCRIPT}" BUILD_TRIQS_COVERAGE_SCRIPT @ONLY)
    file(WRITE ${CMAKE_BINARY_DIR}/build_triqs-coverage "${BUILD_TRIQS_COVERAGE_SCRIPT}")
    execute_process(COMMAND chmod 755 ${CMAKE_BINARY_DIR}/build_triqs-coverage)
    set(TRIQS_PATH "${CMAKE_INSTALL_PREFIX}/bin/triqs")
    set(TRIQS_COVERAGE_EXECUTABLE ${CMAKE_BINARY_DIR}/build_triqs-coverage)
  endif()
  add_custom_target(coverage ${PYTHON_COVERAGE} html COMMAND echo "Open ${CMAKE_BINARY_DIR}/test/python/htmlcov/index.html in browser!" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/python)
endif()

# Add tests
enable_testing()
option(Build_Tests "Enable Tests" ON)
if (Build_Tests)
 message( STATUS "-------- Preparing tests  -------------")
 add_subdirectory(test)
endif()

option(Build_Documentation "Build documentation" OFF)
if(${Build_Documentation})
 message( STATUS "-------- Preparing doc -------------")
  if(NOT ${USE_TRIQS} OR NOT TRIQS_WITH_DOCUMENTATION)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    find_package(Sphinx REQUIRED)
    set(TRIQS_SPHINXBUILD_EXECUTABLE ${SPHINX_EXECUTABLE})
    if(${USE_TRIQS})
        message(STATUS "TRIQS was not compiled with documentation.")
    endif()
    message(STATUS "Cannot use TRIQS-style documentation, will compile documentation without TRIQS.")
    set(DOC_WITH_TRIQS OFF)
  else()
    set(DOC_WITH_TRIQS ON)
  endif()
  add_subdirectory(doc)
  message(STATUS "'make install' will install documentation to: ${CMAKE_INSTALL_PREFIX}/${DOC_DEST}")
endif(${Build_Documentation})

if(NOT ${USE_TRIQS})
    message(STATUS "")
    message(STATUS "***********************************************")
    message(STATUS "* Restrictions apply for usage without TRIQS. *")
    message(STATUS "***********************************************")
    message(STATUS "")
    message(STATUS "In order to use the library, please don't forget to")
    message(STATUS "export PYTHONPATH=\"${CMAKE_INSTALL_PREFIX}:$PYTHONPATH\"")
    message(STATUS "")
endif()

# #############
# Debian Package

option(BUILD_DEBIAN_PACKAGE "Build a deb package" OFF)
if(BUILD_DEBIAN_PACKAGE)
  if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be /usr for packaging")
  endif()
  set(CPACK_PACKAGE_NAME triqs_maxent)
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_VERSION ${MAXENT_VERSION})
  set(CPACK_PACKAGE_CONTACT "https://github.com/TRIQS/maxent")
  execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CMAKE_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "triqs (>= 3.1)")
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
  include(CPack)
endif()
