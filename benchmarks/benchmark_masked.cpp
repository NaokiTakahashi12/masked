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

void BM_RuntimeSubsetMaterialize(benchmark::State &state) {
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

void BM_CompileTimeSubsetEvalArray(benchmark::State &state) {
  auto lhs = make_values(1.0);
  auto rhs = make_values(0.5);

  for (auto _ : state) {
    auto result = masked::unchecked_eval_array(
        masked::select<joint_domain, 0b1011010110010110>(lhs) +
        0.25 * masked::select<joint_domain, 0b1011010110010110>(rhs));
    benchmark::DoNotOptimize(result);
  }
}

void BM_CheckedScatterTo(benchmark::State &state) {
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

BENCHMARK(BM_RuntimeSubsetMaterialize);
BENCHMARK(BM_CompileTimeSubsetEvalArray);
BENCHMARK(BM_CheckedScatterTo);

} // namespace
