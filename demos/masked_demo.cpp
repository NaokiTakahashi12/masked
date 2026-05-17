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
void print_range(std::string_view label, const Range& values) {
  std::cout << label << ": [";
  auto separator = std::string_view{""};
  for (const auto& value : values) {
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

void fault_aware_command_synthesis_demo() {
  std::array<double, actuator_domain::size> position{0.10, -0.20, 0.35,
                                                     0.05, -0.10, 0.40};
  std::array<double, actuator_domain::size> velocity{0.00, 0.30,  -0.20,
                                                     0.10, -0.40, 0.20};
  std::array<double, actuator_domain::size> target_position{0.20, -0.10, 0.10,
                                                            0.00, 0.15,  0.30};
  std::array<double, actuator_domain::size> target_velocity{0.00, 0.00, 0.00,
                                                            0.00, 0.00, 0.00};
  std::array<double, actuator_domain::size> stiffness{120.0, 90.0,  100.0,
                                                      80.0,  110.0, 95.0};
  std::array<double, actuator_domain::size> damping{8.0, 6.0, 7.0,
                                                    5.0, 9.0, 6.5};
  std::array<double, actuator_domain::size> feedforward{1.0, 0.0, -0.5,
                                                        0.2, 0.8, -0.1};

  // Actuators 1 and 4 are isolated, so only the healthy set receives commands.
  const auto healthy =
      masked::subset<actuator_domain>::from_bits_asserted(0b101101);

  const auto tracking_error = masked::select(target_position, healthy) -
                              masked::select(position, healthy);
  const auto damping_error = masked::select(target_velocity, healthy) -
                             masked::select(velocity, healthy);
  const auto torque_expr = masked::select(stiffness, healthy) * tracking_error +
                           masked::select(damping, healthy) * damping_error +
                           masked::select(feedforward, healthy);

  const auto command = masked::checked_domain_array(torque_expr, 0.0);
  const auto worst_error =
      masked::checked_arg_max(tracking_error * tracking_error);
  const auto overloaded =
      masked::checked_find_if(torque_expr, [](double torque) {
        return torque > 12.0 || torque < -12.0;
      });

  std::cout << "fault-aware command synthesis\n";
  std::cout << "  command status: " << status_name(command.result.status)
            << "\n";
  print_range("  full torque command", command.values);
  if (worst_error.value.has_value()) {
    std::cout << "  largest tracking error actuator: "
              << worst_error.value->value() << "\n";
  }
  if (overloaded.value.has_value()) {
    std::cout << "  first actuator over torque budget: "
              << overloaded.value->value() << "\n";
  } else {
    std::cout << "  first actuator over torque budget: none\n";
  }
  std::cout << "\n";
}

} // namespace

auto main() -> int {
  runtime_mask_demo();
  scalar_and_materialize_demo();
  compile_time_mask_demo();
  validation_demo();
  scatter_demo();
  fault_aware_command_synthesis_demo();
  return 0;
}
