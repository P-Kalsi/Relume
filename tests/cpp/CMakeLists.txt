# Native unit tests (GoogleTest). None of them needs a camera; GPU tests render
# offscreen with Dawn.
relume_fetch_googletest()
include(GoogleTest)

# relume_add_gtest(<name> SOURCES <files...> LIBS <targets...>)
function(relume_add_gtest name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE ${ARG_LIBS} GTest::gtest_main)
  relume_set_warnings(${name})
  gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 120)
endfunction()

relume_add_gtest(test_pixel_convert SOURCES test_pixel_convert.cpp LIBS relume_core)
relume_add_gtest(test_frame_queue SOURCES test_frame_queue.cpp LIBS relume_core)
