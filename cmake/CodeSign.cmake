# Code signing for local development builds (macOS). Release signing and
# notarization are out of scope.
#
# RELUME_CODESIGN_IDENTITY
#   "-"     ad-hoc signature (default). The signature changes on every build,
#           so macOS asks for camera permission again after each rebuild.
#   <name>  a code-signing identity in your keychain, e.g. a self-signed
#           "Relume Dev" certificate (see docs/signing.md). The designated
#           requirement then stays the same across rebuilds, so the camera
#           grant persists. The identity is checked at configure time.
#
# No certificate, key or password is stored in the repository.
set(RELUME_CODESIGN_IDENTITY "-" CACHE STRING "codesign identity for Relume.app ('-' = ad-hoc)")

if(APPLE AND NOT RELUME_CODESIGN_IDENTITY STREQUAL "-")
  # Without -v so that a self-signed certificate that is not marked trusted
  # is still listed; codesign can sign with it.
  execute_process(COMMAND security find-identity -p codesigning
    OUTPUT_VARIABLE _relume_identities RESULT_VARIABLE _relume_security_result)
  string(FIND "${_relume_identities}" "\"${RELUME_CODESIGN_IDENTITY}\"" _relume_by_name)
  string(FIND "${_relume_identities}" "${RELUME_CODESIGN_IDENTITY} " _relume_by_hash)
  if(NOT _relume_security_result EQUAL 0 OR (_relume_by_name EQUAL -1 AND _relume_by_hash EQUAL -1))
    message(FATAL_ERROR
      "RELUME_CODESIGN_IDENTITY='${RELUME_CODESIGN_IDENTITY}' is not a code-signing identity in your keychain.\n"
      "List identities with: security find-identity -p codesigning\n"
      "Use -DRELUME_CODESIGN_IDENTITY=- for ad-hoc signing, or see docs/signing.md.")
  endif()
  message(STATUS "Relume.app will be signed with '${RELUME_CODESIGN_IDENTITY}'")
endif()

# relume_codesign(<bundle target>) signs the whole bundle after each build so
# Info.plist (with the camera usage string) is sealed into the signature.
# --timestamp=none: never contact a timestamp server.
function(relume_codesign target)
  if(NOT APPLE)
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND codesign --force --timestamp=none --sign "${RELUME_CODESIGN_IDENTITY}" "$<TARGET_BUNDLE_DIR:${target}>"
    VERBATIM)
endfunction()
