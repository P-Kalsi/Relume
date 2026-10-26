// relume_depth: the native depth pipeline from the command line, for parity
// checks and benchmarks. No camera, no GPU.
//
//   relume_depth run        --image in.ppm --out-depth d.npy [--out-tensor t.npy] [--method stretch|letterbox]
//   relume_depth run-tensor --tensor t.npy --out-depth d.npy
//   relume_depth bench      [--image in.ppm] [--method m] [--warmup 5] [--runs 50]
//
// Common options: --model <onnx> --provider cpu|coreml --units ALL|CPUOnly|CPUAndGPU|CPUAndNeuralEngine
//   --strict (CoreML diagnostic: no CPU fallback) --threads N --profile <prefix> --cache <dir>
//   --compute-plan (log CoreML's planned device per op) --verbose (ONNX Runtime verbose log)
//
// `run` prints one JSON line with the transform and timings. Depth and tensor
// files are float32 .npy (C order).

#include "relume/depth_estimator.hpp"
#include "relume/image_io.hpp"
#include "relume/preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#ifndef RELUME_DEFAULT_MODEL
#define RELUME_DEFAULT_MODEL ""
#endif

namespace {

using relume::kNetDepthFloats;
using relume::kNetSize;
using relume::kNetTensorFloats;
using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// Minimal .npy (version 1.0) float32 writer and reader.
bool WriteNpy(const std::string& path, const float* data, const std::vector<std::size_t>& shape) {
  std::string dims;
  std::size_t count = 1;
  for (const std::size_t d : shape) {
    dims += std::to_string(d) + ", ";
    count *= d;
  }
  if (shape.size() > 1) {
    dims.erase(dims.size() - 2);
  } else {
    dims.pop_back();
  }
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
  const std::size_t unpadded = 10 + header.size() + 1;
  header.append((64 - unpadded % 64) % 64, ' ');
  header.push_back('\n');
  std::ofstream out(path, std::ios::binary);
  const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  out.write(magic, sizeof(magic));
  const auto len = static_cast<std::uint16_t>(header.size());
  const char len_bytes[2] = {static_cast<char>(len & 0xFF), static_cast<char>(len >> 8)};
  out.write(len_bytes, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
  return static_cast<bool>(out);
}

bool ReadNpyFloats(const std::string& path, std::vector<float>& out, std::size_t expected, std::string& error) {
  std::ifstream in(path, std::ios::binary);
  char magic[8] = {};
  in.read(magic, 8);
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0 || magic[6] != 1) {
    error = path + ": not a version 1 .npy file";
    return false;
  }
  unsigned char len_bytes[2] = {};
  in.read(reinterpret_cast<char*>(len_bytes), 2);
  std::string header(static_cast<std::size_t>(len_bytes[0] | (len_bytes[1] << 8)), '\0');
  in.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (header.find("'<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos) {
    error = path + ": expected little-endian float32 in C order";
    return false;
  }
  out.resize(expected);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expected * sizeof(float)));
  if (!in || in.peek() != std::char_traits<char>::eof()) {
    error = path + ": expected exactly " + std::to_string(expected) + " floats";
    return false;
  }
  return true;
}

struct Args {
  std::string command;
  std::map<std::string, std::string> values;
  bool Has(const std::string& key) const { return values.count(key) != 0; }
  std::string Get(const std::string& key, const std::string& fallback = "") const {
    const auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  }
};

std::optional<Args> ParseArgs(int argc, char** argv) {
  if (argc < 2) {
    return std::nullopt;
  }
  Args args;
  args.command = argv[1];
  const char* flags[] = {"--strict", "--compute-plan", "--verbose"};
  for (int i = 2; i < argc; ++i) {
    const std::string key = argv[i];
    if (std::find_if(std::begin(flags), std::end(flags), [&](const char* f) { return key == f; }) !=
        std::end(flags)) {
      args.values[key] = "1";
    } else if (key.rfind("--", 0) == 0 && i + 1 < argc) {
      args.values[key] = argv[++i];
    } else {
      std::fprintf(stderr, "unexpected argument: %s\n", argv[i]);
      return std::nullopt;
    }
  }
  return args;
}

int Usage() {
  std::fprintf(stderr,
               "usage: relume_depth run --image in.ppm --out-depth d.npy [--out-tensor t.npy] [--method m]\n"
               "       relume_depth run-tensor --tensor t.npy --out-depth d.npy\n"
               "       relume_depth bench [--image in.ppm] [--method m] [--warmup N] [--runs N]\n"
               "options: --model path --provider cpu|coreml --units ALL|CPUOnly|CPUAndGPU|CPUAndNeuralEngine\n"
               "         --strict --threads N --profile prefix --cache dir --compute-plan --verbose\n");
  return 2;
}

std::optional<relume::EstimatorConfig> MakeConfig(const Args& args) {
  relume::EstimatorConfig config;
  config.model_path = args.Get("--model", RELUME_DEFAULT_MODEL);
  const auto provider = relume::ParseExecutionProvider(args.Get("--provider", "cpu"));
  const auto units = relume::ParseCoreMLUnits(args.Get("--units", "ALL"));
  if (!provider || !units) {
    std::fprintf(stderr, "unknown --provider or --units\n");
    return std::nullopt;
  }
  config.provider = *provider;
  config.coreml_units = *units;
  config.coreml_strict = args.Has("--strict");
  config.coreml_cache_dir = args.Get("--cache");
  config.coreml_profile_compute_plan = args.Has("--compute-plan");
  config.intra_op_threads = std::atoi(args.Get("--threads", "0").c_str());
  config.profile_prefix = args.Get("--profile");
  config.verbose_log = args.Has("--verbose");
  return config;
}

std::string ShapeText(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    text += (i ? "," : "") + std::to_string(shape[i]);
  }
  return text + "]";
}

void PrintInfo(const relume::IDepthEstimator& estimator) {
  const relume::EstimatorInfo& info = estimator.info();
  std::fprintf(stderr, "onnxruntime %s | %s | session created in %.1f ms\n", info.ort_version.c_str(),
               info.description.c_str(), info.load_ms);
  std::fprintf(stderr, "signature: %s %s %s -> %s %s %s\n", info.input.name.c_str(), info.input.type.c_str(),
               ShapeText(info.input.shape).c_str(), info.output.name.c_str(), info.output.type.c_str(),
               ShapeText(info.output.shape).c_str());
}

std::optional<relume::RgbImage> LoadImage(const std::string& path) {
  std::string error;
  std::optional<relume::RgbImage> image = relume::ReadPpm(path, error);
  if (!image) {
    std::fprintf(stderr, "cannot read %s: %s\n", path.c_str(), error.c_str());
  }
  return image;
}

// A deterministic smooth 1280x720 image for benchmarks without an input file.
relume::RgbImage SyntheticImage() {
  relume::RgbImage image;
  image.width = 1280;
  image.height = 720;
  image.pixels.resize(std::size_t{image.width} * image.height * 3);
  for (std::uint32_t y = 0; y < image.height; ++y) {
    for (std::uint32_t x = 0; x < image.width; ++x) {
      std::uint8_t* p = &image.pixels[(std::size_t{y} * image.width + x) * 3];
      p[0] = static_cast<std::uint8_t>((x * 255) / (image.width - 1));
      p[1] = static_cast<std::uint8_t>((y * 255) / (image.height - 1));
      p[2] = static_cast<std::uint8_t>(((x + y) / 4) & 0xFF);
    }
  }
  return image;
}

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  // Linear interpolation between closest ranks (numpy's default).
  const double rank = p / 100.0 * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(rank);
  const std::size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (rank - static_cast<double>(lo));
}

int Run(const Args& args, relume::IDepthEstimator& estimator) {
  const std::string method_name = args.Get("--method", std::string(relume::ToString(relume::kDefaultResizeMethod)));
  const auto method = relume::ParseResizeMethod(method_name);
  if (!method || !args.Has("--image") || !args.Has("--out-depth")) {
    return Usage();
  }
  const std::optional<relume::RgbImage> image = LoadImage(args.Get("--image"));
  if (!image) {
    return 1;
  }
  relume::Preprocessor pre;
  const auto t0 = Clock::now();
  const auto transform = pre.Run(image->pixels.data(), image->width, image->height, *method, estimator.input());
  const auto t1 = Clock::now();
  if (!transform) {
    std::fprintf(stderr, "empty image\n");
    return 1;
  }
  if (args.Has("--out-tensor") &&
      !WriteNpy(args.Get("--out-tensor"), estimator.input(), {1, 3, kNetSize, kNetSize})) {
    std::fprintf(stderr, "cannot write %s\n", args.Get("--out-tensor").c_str());
    return 1;
  }
  std::string error;
  if (!estimator.Run(error)) {
    std::fprintf(stderr, "run failed: %s\n", error.c_str());
    return 1;
  }
  const auto t2 = Clock::now();
  if (!WriteNpy(args.Get("--out-depth"), estimator.output(), {1, kNetSize, kNetSize})) {
    std::fprintf(stderr, "cannot write %s\n", args.Get("--out-depth").c_str());
    return 1;
  }
  const relume::ResizeTransform& t = *transform;
  std::printf(
      "{\"transform\": {\"method\": \"%s\", \"src_w\": %u, \"src_h\": %u, \"new_w\": %u, \"new_h\": %u, "
      "\"pad_x\": %u, \"pad_y\": %u}, \"preprocess_ms\": %.3f, \"run_ms\": %.3f, \"provider\": \"%s\"}\n",
      std::string(relume::ToString(t.method)).c_str(), t.src_w, t.src_h, t.new_w, t.new_h, t.pad_x, t.pad_y,
      Ms(t0, t1), Ms(t1, t2), estimator.info().description.c_str());
  return 0;
}

int RunTensor(const Args& args, relume::IDepthEstimator& estimator) {
  if (!args.Has("--tensor") || !args.Has("--out-depth")) {
    return Usage();
  }
  std::vector<float> tensor;
  std::string error;
  if (!ReadNpyFloats(args.Get("--tensor"), tensor, kNetTensorFloats, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::copy(tensor.begin(), tensor.end(), estimator.input());
  if (!estimator.Run(error)) {
    std::fprintf(stderr, "run failed: %s\n", error.c_str());
    return 1;
  }
  if (!WriteNpy(args.Get("--out-depth"), estimator.output(), {1, kNetSize, kNetSize})) {
    std::fprintf(stderr, "cannot write %s\n", args.Get("--out-depth").c_str());
    return 1;
  }
  return 0;
}

int Bench(const Args& args, relume::IDepthEstimator& estimator) {
  const auto method = relume::ParseResizeMethod(
      args.Get("--method", std::string(relume::ToString(relume::kDefaultResizeMethod))));
  if (!method) {
    return Usage();
  }
  const int warmup = std::max(0, std::atoi(args.Get("--warmup", "5").c_str()));
  const int runs = std::max(1, std::atoi(args.Get("--runs", "50").c_str()));
  std::optional<relume::RgbImage> image = args.Has("--image") ? LoadImage(args.Get("--image")) : SyntheticImage();
  if (!image) {
    return 1;
  }
  relume::Preprocessor pre;
  relume::ResizeScratch scratch;
  std::vector<float> source_depth;
  std::vector<double> pre_ms, run_ms, post_ms, total_ms;
  std::string error;
  for (int i = 0; i < warmup + runs; ++i) {
    const auto t0 = Clock::now();
    const auto transform = pre.Run(image->pixels.data(), image->width, image->height, *method, estimator.input());
    const auto t1 = Clock::now();
    if (!estimator.Run(error)) {
      std::fprintf(stderr, "run %d failed: %s\n", i, error.c_str());
      return 1;
    }
    const auto t2 = Clock::now();
    relume::DepthToSource(estimator.output(), *transform, source_depth, scratch);
    const auto t3 = Clock::now();
    if (i >= warmup) {
      pre_ms.push_back(Ms(t0, t1));
      run_ms.push_back(Ms(t1, t2));
      post_ms.push_back(Ms(t2, t3));
      total_ms.push_back(Ms(t0, t3));
    }
  }
  std::printf("input %ux%u %s, %d warm-up + %d runs\n", image->width, image->height,
              std::string(relume::ToString(*method)).c_str(), warmup, runs);
  const auto row = [](const char* name, const std::vector<double>& v) {
    std::printf("  %-12s p50 %8.2f ms  p95 %8.2f ms  min %8.2f  max %8.2f\n", name, Percentile(v, 50),
                Percentile(v, 95), *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
  };
  row("preprocess", pre_ms);
  row("run", run_ms);
  row("postprocess", post_ms);
  row("total", total_ms);
  const std::string profile = estimator.EndProfiling();
  if (!profile.empty()) {
    std::printf("profile: %s\n", profile.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::optional<Args> args = ParseArgs(argc, argv);
  if (!args || (args->command != "run" && args->command != "run-tensor" && args->command != "bench")) {
    return Usage();
  }
  const std::optional<relume::EstimatorConfig> config = MakeConfig(*args);
  if (!config) {
    return Usage();
  }
  std::unique_ptr<relume::IDepthEstimator> estimator = relume::MakeOnnxDepthEstimator(*config);
  std::string error;
  if (!estimator->Load(error)) {
    std::fprintf(stderr, "load failed (%s): %s\n", config->model_path.c_str(), error.c_str());
    return 1;
  }
  PrintInfo(*estimator);
  if (args->command == "run") {
    return Run(*args, *estimator);
  }
  if (args->command == "run-tensor") {
    return RunTensor(*args, *estimator);
  }
  return Bench(*args, *estimator);
}
