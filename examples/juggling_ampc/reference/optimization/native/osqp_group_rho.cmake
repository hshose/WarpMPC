# Private ABI bridge for OSQP 1572ae068e9ce9ca723cf8223548ade1ff7acc29.
# Version strings are insufficient: this pinned source's installed package can
# report 0.0.0. Verify headers/implementation/config and archive provenance.
set(JUGGLING_OSQP_PINNED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../resources/OSQP" CACHE PATH "Pinned OSQP private sources")
set(JUGGLING_OSQP_PINNED_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build/osqp_native" CACHE PATH "Build tree which produced the installed OSQP archive")
function(juggling_verify_sha path expected)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Group-rho pin verification: missing ${path}")
  endif()
  file(SHA256 "${path}" actual)
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "Group-rho pin verification failed for ${path}: ${actual}")
  endif()
endfunction()
set(_group_rho_pins
  "include/private/types.h|8d76c2cfda30e92b4a531c0be53384e85f446a6c9901f1979109e60d7c079dec"
  "include/private/algebra_vector.h|dd2feba7b43788c0bbfd96efe9e007935a734d7010f1672cebc783fb9cf57e10"
  "include/private/algebra_matrix.h|b1b4ec679c970c98bf90357a970bd3c43ecb7d84722afe08f407652ec8975ccd"
  "include/private/glob_opts.h|1b218caf0d3749804019682e8f1fbb402563a2037c7fbc11514b4c72d3b796c7"
  "include/public/osqp_api_types.h|c65577b9c9682aa611f0b5beed009d52f1b45cbddcfb1d4daf5379860ac7be9b"
  "include/public/osqp_api_constants.h|b10318c4284d08e9c4ad1cd078bb7d0a152b08e07878a1b098bb2a73a652a4d9"
  "src/auxil.c|b682e3fba83dc2aba7d39d08c1a0f3059d538d74814e464cffbefa6a35cc5591"
  "src/osqp_api.c|f5921f7941581335f44622902ed21b9b6813db69d19c6879484d08a509b8d143")
file(STRINGS "${JUGGLING_OSQP_PINNED_BUILD_DIR}/CMakeCache.txt" _group_rho_cache_source REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" _group_rho_cache_source "${_group_rho_cache_source}")
if(NOT IS_DIRECTORY "${_group_rho_cache_source}")
  message(FATAL_ERROR "Group-rho requires the original OSQP build cache and source tree")
endif()
foreach(pin IN LISTS _group_rho_pins)
  string(REPLACE "|" ";" parts "${pin}")
  list(GET parts 0 relative)
  list(GET parts 1 expected)
  juggling_verify_sha("${JUGGLING_OSQP_PINNED_SOURCE_DIR}/${relative}" "${expected}")
  juggling_verify_sha("${_group_rho_cache_source}/${relative}" "${expected}")
endforeach()
get_target_property(_group_rho_includes osqp::osqpstatic INTERFACE_INCLUDE_DIRECTORIES)
set(_group_rho_public "")
foreach(directory IN LISTS _group_rho_includes)
  if(EXISTS "${directory}/osqp_configure.h")
    set(_group_rho_public "${directory}")
  endif()
endforeach()
if(NOT _group_rho_public)
  message(FATAL_ERROR "Group-rho cannot locate the imported OSQP configuration")
endif()
set(_group_rho_config_hash "3b76a0e1e1d31d6490278a982bf5e114b31dafd95b81e709a452f5ef701e923c")
juggling_verify_sha("${_group_rho_public}/osqp_configure.h" "${_group_rho_config_hash}")
juggling_verify_sha("${JUGGLING_OSQP_PINNED_BUILD_DIR}/include/public/osqp_configure.h" "${_group_rho_config_hash}")
juggling_verify_sha("${_group_rho_public}/osqp_api_types.h" "c65577b9c9682aa611f0b5beed009d52f1b45cbddcfb1d4daf5379860ac7be9b")
juggling_verify_sha("${_group_rho_public}/osqp_api_constants.h" "b10318c4284d08e9c4ad1cd078bb7d0a152b08e07878a1b098bb2a73a652a4d9")
get_target_property(_group_rho_archive osqp::osqpstatic IMPORTED_LOCATION_RELEASE)
if(NOT EXISTS "${_group_rho_archive}" OR NOT EXISTS "${JUGGLING_OSQP_PINNED_BUILD_DIR}/out/libosqpstatic.a")
  message(FATAL_ERROR "Group-rho requires the pinned Release OSQP static archive and its original build artifact")
endif()
file(SHA256 "${JUGGLING_OSQP_PINNED_BUILD_DIR}/out/libosqpstatic.a" _group_rho_archive_hash)
juggling_verify_sha("${_group_rho_archive}" "${_group_rho_archive_hash}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/osqp_group_rho_provenance.json"
  "{\n  \"source_commit\": \"1572ae068e9ce9ca723cf8223548ade1ff7acc29\",\n  \"archive_sha256\": \"${_group_rho_archive_hash}\",\n  \"configure_sha256\": \"${_group_rho_config_hash}\"\n}\n")
target_sources(juggling_controller PRIVATE osqp_group_rho.c)
target_compile_features(juggling_controller PRIVATE c_std_11)
# PUBLIC propagates the Controller class layout to C++ consumers; private OSQP
# headers stay PRIVATE and are included only by osqp_group_rho.c.
target_compile_definitions(juggling_controller PUBLIC JUGGLING_EXPERIMENTAL_GROUP_RHO=1)
target_include_directories(juggling_controller BEFORE PRIVATE "${_group_rho_public}" "${JUGGLING_OSQP_PINNED_SOURCE_DIR}/include/private")
add_executable(juggling_group_rho_check osqp_group_rho_check.c osqp_group_rho.c)
target_compile_features(juggling_group_rho_check PRIVATE c_std_11)
target_include_directories(juggling_group_rho_check PRIVATE . "${_group_rho_public}" "${JUGGLING_OSQP_PINNED_SOURCE_DIR}/include/private")
target_link_libraries(juggling_group_rho_check PRIVATE osqp::osqpstatic m)
