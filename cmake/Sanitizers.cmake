# RELUME_SANITIZERS selects runtime sanitizers for every target in the build.
#
#   ""                   no sanitizers (default)
#   "address,undefined"  AddressSanitizer + UndefinedBehaviorSanitizer
#   "thread"             ThreadSanitizer (cannot be combined with address)
#
# Apple Clang on arm64 supports address, undefined and thread. LeakSanitizer
# and MemorySanitizer are not supported on macOS. MSVC needs different flags
# (/fsanitize=address only), so this module is Clang/GCC-only for now.
set(RELUME_SANITIZERS "" CACHE STRING "Comma-separated sanitizers: address,undefined or thread")

if(RELUME_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "RELUME_SANITIZERS is not implemented for MSVC yet")
  endif()
  if(RELUME_SANITIZERS MATCHES "thread" AND RELUME_SANITIZERS MATCHES "address")
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer")
  endif()
  message(STATUS "Relume sanitizers: ${RELUME_SANITIZERS}")
  add_compile_options(-fsanitize=${RELUME_SANITIZERS} -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${RELUME_SANITIZERS})
  if(RELUME_SANITIZERS MATCHES "undefined")
    # Make UB reports fail the process instead of only printing.
    add_compile_options(-fno-sanitize-recover=undefined)
  endif()
endif()
