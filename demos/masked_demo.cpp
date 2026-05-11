#include <masked/masked.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

struct joint_tag {};
using joint_domain = masked::index_domain<joint_tag, 5, std::uint8_t>;

struct actuator_tag {};
using actuator_domain = masked::index_domain<actuator_tag, 6, std::uint8_t>;

template <class Range>
void print_range(std::string_view label, const Range &values) {
  std::cout << label << ": [";
  auto separator = std::string_view{""};
  for (const auto &value : values) {
    std::cout << separator << value;
    separator = ", ";
  }
  std::cout << "]\n";
}

auto status_name(masked::eval_status status) -> std::string_view {
  switch (status) {
  case masked::eval_status::ok:
    return "ok";
  case masked::eval_status::mask_out_of_range:
    return "mask_out_of_range";
  case masked::eval_status::mask_mismatch:
    return "mask_mismatch";
  case masked::eval_status::source_too_small:
    return "source_too_small";
  case masked::eval_status::output_too_small:
    return "output_too_small";
  case masked::eval_status::compact_input_size_mismatch:
    return "compact_input_size_mismatch";
  case masked::eval_status::empty_selection:
    return "empty_selection";
  case masked::eval_status::no_selected_sequence:
    return "no_selected_sequence";
  }
  return "unknown";
}

void runtime_mask_demo() {
  std::array<double, joint_domain::size> position{0.0, 10.0, 20.0, 30.0, 40.0};
  std::array<double, joint_domain::size> velocity{1.0, 2.0, 3.0, 4.0, 5.0};
  std::array<double, 3> out{};
  const auto active = masked::subset<joint_domain>::from_bits_asserted(0b10110);

  const auto expr =
      masked::select(position, active) * masked::select(velocity, active) + 2.0;
  const auto result = masked::checked_eval_to(out, expr);

  std::cout << "runtime typed subset expression\n";
  std::cout << "  status: " << status_name(result.status) << "\n";
  print_range("  output", out);
  std::cout << "\n";
}

void scalar_and_materialize_demo() {
  std::vector<int> values{1, 2, 3, 4, 5, 6};
  const auto active =
      masked::subset<actuator_domain>::from_bits_asserted(0b101011);

  const auto out = masked::unchecked_materialize(
      -(masked::select(values, active) + masked::scalar(10)));

  std::cout << "scalar broadcast and materialize\n";
  print_range("  output", out);
  std::cout << "\n";
}

void compile_time_mask_demo() {
  std::array<int, joint_domain::size> values{1, 2, 3, 4, 5};

  const auto out = masked::unchecked_eval_array<double>(
      masked::select<joint_domain, 0b10101>(values) * 2.5);

  std::cout << "compile-time mask to std::array\n";
  std::cout << "  static output size: " << out.size() << "\n";
  print_range("  output", out);
  std::cout << "\n";
}

void validation_demo() {
  std::array<int, 3> values{1, 2, 3};
  std::array<int, 1> out{};
  using invalid_domain =
      masked::index_domain<struct invalid_tag, 3, std::uint8_t>;
  const auto invalid = masked::subset<invalid_domain>::unchecked(1U << 4);

  const auto result =
      masked::checked_eval_to(out, masked::select(values, invalid));

  std::cout << "checked validation failure\n";
  std::cout << "  status: " << status_name(result.status) << "\n";
  std::cout << "  selected size: " << result.selected_size << "\n";
  std::cout << "  required source size: " << result.required_source_size
            << "\n";
  std::cout << "\n";
}

void scatter_demo() {
  std::array<int, joint_domain::size> gains{10, 20, 30, 40, 50};
  const auto active = masked::subset<joint_domain>::from_indices({0, 2, 4});

  const auto result =
      masked::checked_scatter_to(gains, masked::select(gains, active) + 5);

  std::cout << "scatter back to original indices\n";
  std::cout << "  status: " << status_name(result.status) << "\n";
  print_range("  updated gains", gains);
  std::cout << "\n";
}

} // namespace

auto main() -> int {
  runtime_mask_demo();
  scalar_and_materialize_demo();
  compile_time_mask_demo();
  validation_demo();
  scatter_demo();
  return 0;
}
