#pragma once

// CPU reference colour conversions. These define the expected output for the
// GPU upload path and are the conversion later preprocessing builds on.
//
// YCbCr -> RGB (NV12), per pixel, in float:
//   video range:  y = (Y - 16) / 219,  cb = (Cb - 128) / 224,  cr = (Cr - 128) / 224
//   full range:   y = Y / 255,         cb = (Cb - 128) / 255,  cr = (Cr - 128) / 255
//   R = y + 2 (1 - Kr) cr
//   B = y + 2 (1 - Kb) cb
//   G = (y - Kr R - Kb B) / (1 - Kr - Kb)
//   BT.601: Kr = 0.299,  Kb = 0.114      BT.709: Kr = 0.2126, Kb = 0.0722
// Each channel is clamped to [0, 1] and quantized as floor(255 v + 0.5).
//
// Chroma is nearest-neighbour: every pixel of a 2x2 luma block uses that
// block's single Cb/Cr sample, whatever the stream's chroma location. This is
// an approximation; it can show 2-pixel colour stair-steps at sharp colour
// edges compared with interpolated chroma (see docs/architecture.md).

#include "relume/video_frame.hpp"

#include <cstdint>
#include <vector>

namespace relume {

// Converts `frame` to tightly packed RGB (3 bytes per pixel, top row first).
// Returns false, leaving `out` empty, if the frame's layout is invalid.
bool ToRgb8(const VideoFrame& frame, std::vector<std::uint8_t>& out);

// As ToRgb8 but with a fourth byte of 255 per pixel.
bool ToRgba8(const VideoFrame& frame, std::vector<std::uint8_t>& out);

// Encodes tightly packed RGB into `out` with the given format (test patterns
// and the file source). NV12 chroma for each 2x2 block is computed from the
// block's mean RGB. `range` and `matrix` are used only for kNv12.
void EncodeRgb8(const std::uint8_t* rgb, std::uint32_t width, std::uint32_t height, PixelFormat format,
                YuvRange range, YuvMatrix matrix, VideoFrame& out);

}  // namespace relume
