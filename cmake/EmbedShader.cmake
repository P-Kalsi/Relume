# Script mode: cmake -DINPUT=<file.wgsl> -DOUTPUT=<file.hpp> -DNAME=<kName> -P EmbedShader.cmake
# Writes a header holding the shader source as a constexpr string, so shaders
# are compiled into the binary and never loaded from disk at runtime.
file(READ "${INPUT}" source)
string(FIND "${source}" ")wgsl\"" collision)
if(NOT collision EQUAL -1)
  message(FATAL_ERROR "${INPUT} contains the raw-string delimiter )wgsl\"")
endif()
file(WRITE "${OUTPUT}.tmp"
  "// Generated from ${INPUT} by cmake/EmbedShader.cmake. Do not edit.\n"
  "#pragma once\n"
  "inline constexpr char ${NAME}[] = R\"wgsl(${source})wgsl\";\n")
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.tmp")
