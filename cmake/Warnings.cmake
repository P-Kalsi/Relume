# relume_set_warnings(<target>) applies Relume's warning policy to our own
# targets only (third-party code is compiled with its own flags).
function(relume_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
  endif()
endfunction()
