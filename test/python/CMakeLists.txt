# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h5 *.dat *.npz *.ref)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
set(all_tests
  # general_tests
  omega_meshes minimizer_cubefun minimizer_circlefun
  minimizer_sin cached_view plus_minus_entropy maxent_cost_function_lazy
  maxent_cost_function_d functions_different_grid tau_kernel alpha_meshes
  default_models logtaker matrix_maxent_result maxent_loop tau_maxent
  bryan_cost_function huge_alpha cov elementwise_maxent elementwise_set_G
  pickle_maxent_result complex_A
  #triqs_tests
  maxent_result analyzers set_G_tau set_G_tau_complex_matrix
  tau_maxent_off srvo3_mesh_and_ALPS complex_elementwise_maxent G_w_from_A_w
  G_tau_from_A_w sigma_continuation)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name} COMMAND ${TRIQS_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH} ${SANITIZER_RT_PRELOAD})
endforeach()
