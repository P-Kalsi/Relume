// OnnxDepthEstimator with the real ONNX Runtime CPU provider. Tests that need
// the exported model are skipped when it is missing (run scripts/export.py).

#include "relume/depth_estimator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>

namespace relume {
namespace {

bool ModelExists() { return std::filesystem::exists(RELUME_TEST_MODEL); }

EstimatorConfig CpuConfig() {
  EstimatorConfig config;
  config.model_path = RELUME_TEST_MODEL;
  config.intra_op_threads = 4;
  return config;
}

void FillSmooth(float* tensor) {
  for (std::size_t i = 0; i < kNetTensorFloats; ++i) {
    tensor[i] = std::sin(static_cast<float>(i % 4481) * 0.01f);
  }
}

TEST(OnnxDepth, LoadsChecksSignatureAndRuns) {
  if (!ModelExists()) {
    GTEST_SKIP() << "model missing: " << RELUME_TEST_MODEL;
  }
  std::unique_ptr<IDepthEstimator> estimator = MakeOnnxDepthEstimator(CpuConfig());
  std::string error;
  ASSERT_TRUE(estimator->Load(error)) << error;
  const EstimatorInfo& info = estimator->info();
  EXPECT_EQ(info.input.name, "image");
  EXPECT_EQ(info.input.type, "float");
  EXPECT_EQ(info.input.shape, (std::vector<std::int64_t>{1, 3, 448, 448}));
  EXPECT_EQ(info.output.name, "depth");
  EXPECT_EQ(info.output.shape, (std::vector<std::int64_t>{1, 448, 448}));
  EXPECT_EQ(info.description, "CPU, 4 threads");
  EXPECT_EQ(info.ort_version, "1.31.0");

  FillSmooth(estimator->input());
  ASSERT_TRUE(estimator->Run(error)) << error;
  const std::vector<float> first(estimator->output(), estimator->output() + kNetDepthFloats);
  EXPECT_TRUE(std::all_of(first.begin(), first.end(), [](float v) { return std::isfinite(v) && v >= 0.0f; }));
  EXPECT_GT(*std::max_element(first.begin(), first.end()), 0.0f);

  // Same buffers, same input: identical output, and the output buffer is
  // the one ONNX Runtime wrote into (no reallocation between runs).
  const float* output_before = estimator->output();
  ASSERT_TRUE(estimator->Run(error)) << error;
  EXPECT_EQ(estimator->output(), output_before);
  EXPECT_TRUE(std::equal(first.begin(), first.end(), estimator->output()));
}

TEST(OnnxDepth, MissingModelIsAnErrorNotACrash) {
  EstimatorConfig config = CpuConfig();
  config.model_path = "/nonexistent/relume_model.onnx";
  std::unique_ptr<IDepthEstimator> estimator = MakeOnnxDepthEstimator(config);
  std::string error;
  EXPECT_FALSE(estimator->Load(error));
  EXPECT_NE(error.find("ONNX Runtime"), std::string::npos) << error;
  EXPECT_FALSE(estimator->Run(error));
  EXPECT_EQ(error, "the model is not loaded");
}

TEST(OnnxDepth, CorruptModelIsAnErrorNotACrash) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "relume_corrupt_model.onnx";
  {
    std::ofstream out(path, std::ios::binary);
    out << "this is not an ONNX model";
  }
  EstimatorConfig config = CpuConfig();
  config.model_path = path.string();
  std::unique_ptr<IDepthEstimator> estimator = MakeOnnxDepthEstimator(config);
  std::string error;
  EXPECT_FALSE(estimator->Load(error));
  EXPECT_FALSE(error.empty());
  std::filesystem::remove(path);
}

TEST(OnnxDepth, TerminateStopsARunInProgressAndLaterRuns) {
  if (!ModelExists()) {
    GTEST_SKIP() << "model missing: " << RELUME_TEST_MODEL;
  }
  EstimatorConfig config = CpuConfig();
  config.intra_op_threads = 1;  // about 1 s per run, so Terminate lands mid-run
  std::unique_ptr<IDepthEstimator> estimator = MakeOnnxDepthEstimator(config);
  std::string error;
  ASSERT_TRUE(estimator->Load(error)) << error;
  FillSmooth(estimator->input());

  bool ok = true;
  std::string run_error;
  const auto start = std::chrono::steady_clock::now();
  std::thread runner([&] { ok = estimator->Run(run_error); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  estimator->Terminate();
  runner.join();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  EXPECT_FALSE(ok);
  EXPECT_EQ(run_error, "stopped");
  // Measured for information: how soon ONNX Runtime checks the flag.
  std::printf("[terminate] Run returned %.0f ms after start (terminate at ~100 ms)\n", ms);

  EXPECT_FALSE(estimator->Run(error));
  EXPECT_EQ(error, "stopped");
}

TEST(OnnxDepth, ProviderNamesRoundTrip) {
  EXPECT_EQ(ParseExecutionProvider("cpu"), ExecutionProvider::kCpu);
  EXPECT_EQ(ParseExecutionProvider("coreml"), ExecutionProvider::kCoreML);
  EXPECT_FALSE(ParseExecutionProvider("cuda").has_value());
  for (const CoreMLUnits units :
       {CoreMLUnits::kAll, CoreMLUnits::kCPUOnly, CoreMLUnits::kCPUAndGPU, CoreMLUnits::kCPUAndNeuralEngine}) {
    EXPECT_EQ(ParseCoreMLUnits(ToString(units)), units);
  }
  EXPECT_FALSE(ParseCoreMLUnits("GPU").has_value());
}

}  // namespace
}  // namespace relume
