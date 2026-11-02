#include "relume/layout.hpp"

#include <gtest/gtest.h>

namespace relume {
namespace {

void ExpectRect(PixelRect r, std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h) {
  EXPECT_EQ(r.x, x);
  EXPECT_EQ(r.y, y);
  EXPECT_EQ(r.width, w);
  EXPECT_EQ(r.height, h);
}

TEST(FitRect, SameAspectFillsExactly) {
  ExpectRect(FitRect(1280, 720, 1280, 720), 0, 0, 1280, 720);
  ExpectRect(FitRect(1920, 1080, 1280, 720), 0, 0, 1280, 720);
  // Retina: a 1280x720 output in a 2560x1440 framebuffer.
  ExpectRect(FitRect(1280, 720, 2560, 1440), 0, 0, 2560, 1440);
}

TEST(FitRect, FourThreeIntoSixteenNineIsPillarboxed) {
  ExpectRect(FitRect(640, 480, 1280, 720), 160, 0, 960, 720);
}

TEST(FitRect, WideIntoTallIsLetterboxed) {
  ExpectRect(FitRect(1280, 720, 800, 1000), 0, 275, 800, 450);
}

TEST(FitRect, OddFramebufferSizesStayCentredAndUnstretched) {
  // Window dragged to an arbitrary size on a 2x display.
  const PixelRect r = FitRect(1280, 720, 2001, 1333);
  EXPECT_EQ(r.width, 2001u);
  EXPECT_EQ(r.height, 1126u);  // 2001 * 9 / 16 = 1125.56 -> 1126
  EXPECT_EQ(r.y, (1333u - 1126u) / 2);
  // Aspect error is below one pixel.
  EXPECT_NEAR(static_cast<double>(r.width) / r.height, 16.0 / 9.0, 2.0 / r.height);
}

TEST(FitRect, DegenerateSizes) {
  ExpectRect(FitRect(0, 720, 1280, 720), 0, 0, 0, 0);
  ExpectRect(FitRect(1280, 720, 0, 0), 0, 0, 0, 0);
  ExpectRect(FitRect(1, 1, 1, 1), 0, 0, 1, 1);
  // Extremely wide source still yields at least one row.
  ExpectRect(FitRect(10000, 1, 10, 10), 0, 4, 10, 1);
}

// Light drag on the Controls preview: the preview shows the 1280x720 output
// at any size; the camera image may be pillarboxed inside it.
TEST(PreviewToImageUv, MapsAResizedPreviewOfA16x9Image) {
  PreviewGeometry p{100.0f, 50.0f, 640.0f, 360.0f, 1280, 720, FitRect(1280, 720, 1280, 720)};
  const ImageUv centre = PreviewToImageUv(420.0f, 230.0f, p);
  EXPECT_FLOAT_EQ(centre.u, 0.5f);
  EXPECT_FLOAT_EQ(centre.v, 0.5f);
  const ImageUv top_left = PreviewToImageUv(100.0f, 50.0f, p);
  EXPECT_FLOAT_EQ(top_left.u, 0.0f);
  EXPECT_FLOAT_EQ(top_left.v, 0.0f);
  const ImageUv quarter = PreviewToImageUv(260.0f, 140.0f, p);
  EXPECT_FLOAT_EQ(quarter.u, 0.25f);
  EXPECT_FLOAT_EQ(quarter.v, 0.25f);
}

TEST(PreviewToImageUv, MapsAPillarboxed4x3ImageAndClampsInTheBars) {
  // A 4:3 camera: 960x720 at x = 160 in the output. Preview at half size.
  PreviewGeometry p{0.0f, 0.0f, 640.0f, 360.0f, 1280, 720, FitRect(640, 480, 1280, 720)};
  ASSERT_EQ(p.image.x, 160u);
  const ImageUv left_edge = PreviewToImageUv(80.0f, 180.0f, p);  // output x 160
  EXPECT_FLOAT_EQ(left_edge.u, 0.0f);
  EXPECT_FLOAT_EQ(left_edge.v, 0.5f);
  const ImageUv in_bar = PreviewToImageUv(10.0f, 180.0f, p);
  EXPECT_FLOAT_EQ(in_bar.u, 0.0f);
  const ImageUv right_bar = PreviewToImageUv(639.0f, 400.0f, p);  // also below the preview
  EXPECT_FLOAT_EQ(right_bar.u, 1.0f);
  EXPECT_FLOAT_EQ(right_bar.v, 1.0f);
  const ImageUv centre = PreviewToImageUv(320.0f, 180.0f, p);
  EXPECT_FLOAT_EQ(centre.u, 0.5f);
}

TEST(PreviewToImageUv, RoundTripsThroughTheMarkerPosition) {
  for (const PixelRect image : {FitRect(1280, 720, 1280, 720), FitRect(640, 480, 1280, 720), FitRect(720, 1280, 1280, 720)}) {
    PreviewGeometry p{37.0f, 11.0f, 517.0f, 291.0f, 1280, 720, image};
    for (const float u : {0.0f, 0.13f, 0.5f, 0.9f, 1.0f}) {
      for (const float v : {0.0f, 0.4f, 1.0f}) {
        float sx = 0.0f, sy = 0.0f;
        ImageUvToPreview({u, v}, p, sx, sy);
        const ImageUv back = PreviewToImageUv(sx, sy, p);
        EXPECT_NEAR(back.u, u, 1e-5f);
        EXPECT_NEAR(back.v, v, 1e-5f);
      }
    }
  }
}

TEST(PreviewToImageUv, EmptyGeometryIsTheCentre) {
  const ImageUv uv = PreviewToImageUv(5.0f, 5.0f, PreviewGeometry{});
  EXPECT_FLOAT_EQ(uv.u, 0.5f);
  EXPECT_FLOAT_EQ(uv.v, 0.5f);
}

}  // namespace
}  // namespace relume
