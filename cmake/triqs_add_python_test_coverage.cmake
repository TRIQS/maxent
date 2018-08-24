macro(triqs_add_python_test testname)
 enable_testing()

 set(testcmd ${PythonBuildExecutable})
 set(testref ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.ref)

 # run this test via mpirun if TEST_MPI_NUMPROC is set
 if(TEST_MPI_NUMPROC)
  set(testname_ ${testname}_np${TEST_MPI_NUMPROC})
  set(testcmd ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_MPI_NUMPROC} ${testcmd})
 else(TEST_MPI_NUMPROC)
  set(testname_ ${testname})
 endif(TEST_MPI_NUMPROC)

 if (EXISTS ${testref})

  file( COPY ${testref} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  add_test(${testname_}
   ${CMAKE_COMMAND}
   -Dname=${testname_}
   -Dcmd=${PythonBuildExecutable}
   -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/${testname}.py
   -Dreference=${testref}
   -P ${CMAKE_SOURCE_DIR}/cmake/run_python_test.cmake
  )

 else (EXISTS ${testref})

  add_test(${testname_}
   ${CMAKE_COMMAND}
   -Dname=${testname_}
   -Dcmd=${PythonBuildExecutable}
   -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/${testname}.py
   -P ${CMAKE_SOURCE_DIR}/cmake/run_python_test.cmake
  )

 endif (EXISTS ${testref})

 if(TEST_MPI_NUMPROC)
  set_tests_properties(${testname_} PROPERTIES PROCESSORS ${TEST_MPI_NUMPROC})
 endif(TEST_MPI_NUMPROC)

endmacro(triqs_add_python_test)
