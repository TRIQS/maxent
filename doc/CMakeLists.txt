# generate the conf.py
if(${DOC_WITH_TRIQS})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY)
else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf_notriqs.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY)
endif()

# all rst files of the documentation
file(GLOB_RECURSE doc_sources *.rst)

# create documentation target
set(sphinx_top ${CMAKE_CURRENT_BINARY_DIR}/html/index.html)
if(${TRIQS_V1})
  add_custom_command(OUTPUT ${sphinx_top} DEPENDS ${doc_sources} py_copy
    COMMAND ${CMAKE_BINARY_DIR}/build_pytriqs ${TRIQS_SPHINXBUILD_EXECUTABLE} -c . -b html ${CMAKE_CURRENT_BINARY_DIR} html)
  add_custom_target(doc_sphinx ALL DEPENDS ${sphinx_top})
else() # TRIQS 2.0 and DOC_WITH_TRIQS=OFF
  add_custom_command(OUTPUT ${sphinx_top} DEPENDS ${doc_sources}
    COMMAND env PYTHONPATH=${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH} ${TRIQS_SPHINXBUILD_EXECUTABLE} -c . -j8 -b html ${CMAKE_CURRENT_BINARY_DIR} html)
  add_custom_target(doc_message
           COMMAND ${CMAKE_COMMAND} -E echo ""
           COMMAND ${CMAKE_COMMAND} -E echo "   Documentation can be found at"
           COMMAND ${CMAKE_COMMAND} -E echo "   ${sphinx_top}"
           COMMAND ${CMAKE_COMMAND} -E echo "   and will be installed to"
           COMMAND ${CMAKE_COMMAND} -E echo "   ${CMAKE_INSTALL_PREFIX}/${DOC_DEST}"
           COMMAND ${CMAKE_COMMAND} -E echo ""
           DEPENDS ${sphinx_top})
  add_custom_target(doc_sphinx ALL DEPENDS ${sphinx_top} ${CMAKE_CURRENT_BINARY_DIR} doc_message)
endif()

# ------------------------------------------------------------------------------------------------
# Copy the directory substructure and link all files with relevant extensions
# cp_rs is a script in cpp2py/bin, it mimics cp -rs on Linux
# and filters the relevant extension
# ------------------------------------------------------------------------------------------------
set(EXT_FOR_DOC "rst png txt css_t conf css js gif jpg py html bib md")
execute_process(COMMAND cp_rs ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${EXT_FOR_DOC})

# install
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ COMPONENT documentation DESTINATION ${DOC_DEST}
 FILES_MATCHING
 REGEX "\\.(html|pdf|png|gif|jpg|js|xsl|css|py|txt|inv|bib)$"
 PATTERN "_*"
)
