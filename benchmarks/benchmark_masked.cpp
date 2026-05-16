#include <masked/masked.hpp>

#include <array>
#include <benchmark/benchmark.h>
#include <cstdint>

namespace {

struct joint_tag {};
using joint_domain = masked::index_domain<joint_tag, 16, std::uint16_t>;

auto make_values(double scale) -> std::array<double, joint_domain::size> {
  std::array<double, joint_domain::size> values{};
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = scale * static_cast<double>(index + 1U);
  }
  return values;
}

void BM_RuntimeSubsetMaterialize(benchmark::State& state) {
  auto lhs = make_values(1.0);
  auto rhs = make_values(0.5);
  const auto active =
      masked::subset<joint_domain>::from_bits_asserted(0b1011010110010110);

  for (auto _ : state) {
    auto result = masked::unchecked_materialize(
        masked::select(lhs, active) + 0.25 * masked::select(rhs, active));
    benchmark::DoNotOptimize(result);
  }
}

void BM_CompileTimeSubsetEvalArray(benchmark::State& state) {
  auto lhs = make_values(1.0);
  auto rhs = make_values(0.5);

  for (auto _ : state) {
    auto result = masked::unchecked_eval_array(
        masked::select<joint_domain, 0b1011010110010110>(lhs) +
        0.25 * masked::select<joint_domain, 0b1011010110010110>(rhs));
    benchmark::DoNotOptimize(result);
  }
}

void BM_CheckedScatterTo(benchmark::State& state) {
  auto base = make_values(1.0);
  const auto active =
      masked::subset<joint_domain>::from_bits_asserted(0b1111000011110000);

  for (auto _ : state) {
    auto out = base;
    [[maybe_unused]] const auto result =
        masked::checked_scatter_to(out, masked::select(out, active) + 3.0);
    benchmark::DoNotOptimize(out);
  }
}

void BM_CheckedGatherCompactFromRuntimeSubset(benchmark::State& state) {
  auto values = make_values(1.0);
  const auto active =
      masked::subset<joint_domain>::from_bits_asserted(0b1011010110010110);

  for (auto _ : state) {
    auto result = masked::checked_gather_compact_from(values, active);
    benchmark::DoNotOptimize(result);
  }
}

void BM_CheckedScatterCompactToRuntimeSubset(benchmark::State& state) {
  auto base = make_values(1.0);
  constexpr std::array<double, 9> compact{
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  const auto active =
      masked::subset<joint_domain>::from_bits_asserted(0b1011010110010110);

  for (auto _ : state) {
    auto out = base;
    [[maybe_unused]] const auto result =
        masked::checked_scatter_compact_to(out, compact, active);
    benchmark::DoNotOptimize(out);
  }
}

void BM_CheckedTransformSelectedRuntimeSubset(benchmark::State& state) {
  auto base = make_values(1.0);
  const auto active =
      masked::subset<joint_domain>::from_bits_asserted(0b1111000011110000);

  for (auto _ : state) {
    auto out = base;
    [[maybe_unused]] const auto result = masked::checked_transform_selected(
        out, active, [](auto index, double value) {
          return value + static_cast<double>(index.value());
        });
    benchmark::DoNotOptimize(out);
  }
}

void BM_CheckedDomainArrayRuntimeSubset(benchmark::State& state) {
  auto lhs = make_values(1.0);
  auto rhs = make_values(0.5);
  const auto active =
      masked::subset<joint_domain>::from_bits_asserted(0b1011010110010110);

  for (auto _ : state) {
    auto result = masked::checked_domain_array(
        masked::select(lhs, active) - masked::select(rhs, active), -1.0);
    benchmark::DoNotOptimize(result);
  }
}

void BM_UncheckedDotCompileTimeSubset(benchmark::State& state) {
  auto lhs = make_values(1.0);
  auto rhs = make_values(0.5);

  for (auto _ : state) {
    auto result = masked::unchecked_dot(
        masked::select<joint_domain, 0b1011010110010110>(lhs),
        masked::select<joint_domain, 0b1011010110010110>(rhs));
    benchmark::DoNotOptimize(result);
  }
}

void BM_CheckedMinRuntimeSubset(benchmark::State& state) {
  auto values = make_values(1.0);
  const auto active =
      masked::subset<joint_domain>::from_bits_asserted(0b1011010110010110);

  for (auto _ : state) {
    auto result = masked::checked_min(masked::select(values, active));
    benchmark::DoNotOptimize(result);
  }
}

void BM_CheckedFindIfRuntimeSubset(benchmark::State& state) {
  auto values = make_values(1.0);
  const auto active =
      masked::subset<joint_domain>::from_bits_asserted(0b1011010110010110);

  for (auto _ : state) {
    auto result = masked::checked_find_if(
        masked::select(values, active),
        [](double value) { return value > 10.0; });
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(BM_RuntimeSubsetMaterialize);
BENCHMARK(BM_CompileTimeSubsetEvalArray);
BENCHMARK(BM_CheckedScatterTo);
BENCHMARK(BM_CheckedGatherCompactFromRuntimeSubset);
BENCHMARK(BM_CheckedScatterCompactToRuntimeSubset);
BENCHMARK(BM_CheckedTransformSelectedRuntimeSubset);
BENCHMARK(BM_CheckedDomainArrayRuntimeSubset);
BENCHMARK(BM_UncheckedDotCompileTimeSubset);
BENCHMARK(BM_CheckedMinRuntimeSubset);
BENCHMARK(BM_CheckedFindIfRuntimeSubset);

} // namespace
