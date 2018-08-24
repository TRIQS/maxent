
if(${TRIQS_V1})
    set(PYTRIQS_PATH ${CMAKE_INSTALL_PREFIX}/bin/pytriqs)
elseif(${TRIQS_V2})
    set(PYTRIQS_PATH "python")
endif()
configure_file(plot_maxent.bash.in plot_maxent)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/plot_maxent DESTINATION ${BIN_DEST}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
