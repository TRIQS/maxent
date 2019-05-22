#add version file
configure_file(version.py.in triqs_maxent/version.py)
configure_file(triqs_support.py.in triqs_maxent/triqs_support.py)

# where will the python end up in triqs python tree ?
if(${TRIQS_V1})
  # site_customize for build
  set(package_name "triqs_maxent")
  #configure_file(${CMAKE_SOURCE_DIR}/cmake/sitecustomize.py ${CMAKE_CURRENT_BINARY_DIR}/sitecustomize.py @ONLY)
  # make a local pytriqs copy to be able to run in the current dir without installation
  triqs_prepare_local_pytriqs(${python_destination})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/triqs_maxent/version.py DESTINATION ${PYTHON_LIB_DEST})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/triqs_maxent/triqs_support.py DESTINATION ${PYTHON_LIB_DEST})
else() #TRIQS 2.1 and USE_TRIQS=OFF
  set(package_name "triqs_maxent")

  # Create a temporary copy of the python modules so that we can run before installation with the test
  FILE(GLOB_RECURSE PYTHON_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py )
  foreach(f ${PYTHON_SOURCES})
   configure_file(${f} triqs_maxent/${f} COPYONLY)
   get_filename_component(PARENT_DIR ${f} DIRECTORY)
   install(FILES ${f} DESTINATION ${PYTHON_LIB_DEST}/${PARENT_DIR})
  endforeach()

  # install files
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/triqs_maxent/version.py DESTINATION ${PYTHON_LIB_DEST})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/triqs_maxent/triqs_support.py DESTINATION ${PYTHON_LIB_DEST})
endif()
