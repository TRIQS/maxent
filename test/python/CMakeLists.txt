# Copy h5-files to binary dir
FILE(GLOB all_h5_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h5)
FILE(GLOB all_dat_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.dat)
FILE(GLOB all_npz_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.npz)
FILE(GLOB all_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_h5_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_dat_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_ref_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_npz_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/matplotlibrc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(general_tests omega_meshes minimizer_cubefun minimizer_circlefun
    minimizer_sin cached_view plus_minus_entropy maxent_cost_function_lazy
    maxent_cost_function_d functions_different_grid tau_kernel alpha_meshes
    default_models logtaker matrix_maxent_result maxent_loop tau_maxent
    bryan_cost_function huge_alpha cov elementwise_maxent elementwise_set_G
    pickle_maxent_result complex_A)

set(triqs_tests maxent_result analyzers set_G_tau set_G_tau_complex_matrix
    tau_maxent_off srvo3_mesh_and_ALPS complex_elementwise_maxent G_w_from_A_w
    G_tau_from_A_w sigma_continuation)

if(${USE_TRIQS})
    set(all_tests ${general_tests} ${triqs_tests})
else()
    set(all_tests ${general_tests})
endif()

if(${TRIQS_V1})
  find_package(TriqsTest)

  if(${TEST_COVERAGE})
    set(PythonBuildExecutable ${TRIQS_COVERAGE_EXECUTABLE})
    include(${CMAKE_SOURCE_DIR}/cmake/triqs_add_python_test_coverage.cmake)
  endif()

  foreach(t ${all_tests})
    triqs_add_python_test(${t})
  endforeach()
else() #TRIQS 2.1 and USE_TRIQS=OFF
  set(python_executable python3)

  if(${TEST_COVERAGE})
      set(python_executable ${PYTHON_COVERAGE} run --append --source "${CMAKE_BINARY_DIR}/" )
  endif()

  foreach(t ${all_tests})
    add_test(NAME ${t} COMMAND ${python_executable} ${CMAKE_CURRENT_SOURCE_DIR}/${t}.py)
  endforeach()

  set_property(TEST ${all_tests} PROPERTY ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH} )
endif()
