add_executable(RelumeCameraProbe MACOSX_BUNDLE main.mm)
set_target_properties(RelumeCameraProbe PROPERTIES
  MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist")
target_compile_options(RelumeCameraProbe PRIVATE -fobjc-arc)
target_link_libraries(RelumeCameraProbe PRIVATE
  "-framework Foundation" "-framework AVFoundation" "-framework CoreMedia" "-framework CoreVideo")
relume_set_warnings(RelumeCameraProbe)

# Ad-hoc sign the whole bundle so Info.plist (and its camera usage string) is
# sealed into the signature that macOS privacy checks (TCC) look at.
add_custom_command(TARGET RelumeCameraProbe POST_BUILD
  COMMAND codesign --force --sign - "$<TARGET_BUNDLE_DIR:RelumeCameraProbe>"
  VERBATIM)

# Not registered with CTest: camera permission requires a person to approve
# the macOS prompt, so this probe is run manually (see docs/build.md).
