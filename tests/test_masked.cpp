#include <masked/masked.hpp>

#include <array>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct joint_tag {};
using joint_domain = masked::index_domain<joint_tag, 5, std::uint8_t>;

struct short_joint_tag {};
using short_joint_domain =
    masked::index_domain<short_joint_tag, 4, std::uint8_t>;

TEST(MaskedTest, RuntimeMasksEvaluateByOriginalIndex) {
  std::array<double, joint_domain::size> position{0.0, 10.0, 20.0, 30.0, 40.0};
  std::array<double, joint_domain::size> velocity{1.0, 2.0, 3.0, 4.0, 5.0};
  std::array<double, 3> out{};
  const auto active = masked::subset<joint_domain>::from_bits_asserted(0b10110);

  const auto expr =
      masked::select(position, active) * masked::select(velocity, active);
  const auto result = masked::checked_eval_to(out, expr);

  EXPECT_EQ(result.status, masked::eval_status::ok);
  EXPECT_EQ(result.selected_size, 3U);
  EXPECT_DOUBLE_EQ(out[0], 20.0);
  EXPECT_DOUBLE_EQ(out[1], 60.0);
  EXPECT_DOUBLE_EQ(out[2], 200.0);
}

TEST(MaskedTest, SupportsScalarBroadcastAndUnaryMinus) {
  std::array<int, joint_domain::size> a{1, 2, 3, 4, 5};
  std::array<int, joint_domain::size> b{10, 20, 30, 40, 50};
  std::array<int, 3> out{};
  const auto active = masked::subset<joint_domain>::from_bits_asserted(0b10101);

  const auto expr =
      -(masked::select(a, active) + 2) + masked::select(b, active) / 10;
  masked::unchecked_eval_to(out.begin(), expr);

  EXPECT_EQ(out[0], -2);
  EXPECT_EQ(out[1], -2);
  EXPECT_EQ(out[2], -2);
}

TEST(MaskedTest, EmptyMaskWritesNoElements) {
  std::array<int, short_joint_domain::size> a{1, 2, 3, 4};
  std::array<int, 2> out{11, 22};

  const auto expr = masked::select(a, masked::none<short_joint_domain>()) * 10;
  const auto result = masked::checked_eval_to(out, expr);

  EXPECT_EQ(result.status, masked::eval_status::ok);
  EXPECT_EQ(result.selected_size, 0U);
  EXPECT_EQ(out[0], 11);
  EXPECT_EQ(out[1], 22);
}

TEST(MaskedTest, CheckedPathDetectsMaskOutsideDomainRange) {
  std::array<int, short_joint_domain::size> a{1, 2, 3, 4};
  std::array<int, 1> out{};
  const auto invalid = masked::subset<short_joint_domain>::unchecked(1U << 4);

  const auto expr = masked::select(a, invalid);
  const auto result = masked::checked_eval_to(out, expr);

  EXPECT_EQ(result.status, masked::eval_status::mask_out_of_range);
  EXPECT_EQ(result.selected_size, 1U);
  EXPECT_EQ(result.required_source_size, 0U);
}

TEST(MaskedTest, CheckedPathDetectsMaskMismatch) {
  std::array<int, short_joint_domain::size> a{1, 2, 3, 4};
  std::array<int, short_joint_domain::size> b{10, 20, 30, 40};
  std::array<int, 3> out{};

  const auto expr =
      masked::select(a,
                     masked::subset<short_joint_domain>::from_bits_asserted(
                         0b0111)) +
      masked::select(b,
                     masked::subset<short_joint_domain>::from_bits_asserted(
                         0b0011));
  const auto result = masked::checked_eval_to(out, expr);

  EXPECT_EQ(result.status, masked::eval_status::mask_mismatch);
}

TEST(MaskedTest, CheckedPathDetectsOutputCapacity) {
  std::array<int, short_joint_domain::size> a{1, 2, 3, 4};
  std::array<int, 2> out{};

  const auto expr =
      masked::select(a,
                     masked::subset<short_joint_domain>::from_bits_asserted(
                         0b111));
  const auto result = masked::checked_eval_to(out, expr);

  EXPECT_EQ(result.status, masked::eval_status::output_too_small);
  EXPECT_EQ(result.selected_size, 3U);
}

TEST(MaskedTest, ValidationDetectsSourceTooSmallForDynamicRange) {
  std::vector<int> a{1, 2, 3};

  const auto result =
      masked::validate(masked::select(a, masked::all<short_joint_domain>()));

  EXPECT_EQ(result.status, masked::eval_status::source_too_small);
  EXPECT_EQ(result.selected_size, short_joint_domain::size);
  EXPECT_EQ(result.required_source_size, short_joint_domain::size);
}

TEST(MaskedTest, MakeIndexReturnsOptionalForCheckedConstruction) {
  const auto valid = masked::make_index<short_joint_domain>(2);
  const auto invalid = masked::make_index<short_joint_domain>(5);

  ASSERT_TRUE(valid.has_value());
  EXPECT_EQ(valid->value(), 2U);
  EXPECT_FALSE(invalid.has_value());
}

TEST(MaskedTest, MaterializeCreatesIndependentStorage) {
  std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0};
  const auto selected = masked::subset<joint_domain>::from_bits_asserted(0b10101);

  auto materialized =
      masked::unchecked_materialize(masked::select(values, selected) + 0.5);
  values[0] = 100.0;
  values[2] = 200.0;

  ASSERT_EQ(materialized.size(), 3U);
  EXPECT_DOUBLE_EQ(materialized[0], 1.5);
  EXPECT_DOUBLE_EQ(materialized[1], 3.5);
  EXPECT_DOUBLE_EQ(materialized[2], 5.5);
}

TEST(MaskedTest, CheckedMaterializeDetectsMaskOutsideDomainRange) {
  std::array<int, short_joint_domain::size> values{1, 2, 3, 4};
  const auto invalid = masked::subset<short_joint_domain>::unchecked(1U << 4);

  const auto result =
      masked::checked_materialize(masked::select(values, invalid));

  EXPECT_EQ(result.result.status, masked::eval_status::mask_out_of_range);
  EXPECT_TRUE(result.values.empty());
}

TEST(MaskedTest, CheckedMaterializeDetectsMaskMismatch) {
  std::array<int, short_joint_domain::size> a{1, 2, 3, 4};
  std::array<int, short_joint_domain::size> b{10, 20, 30, 40};

  const auto result = masked::checked_materialize(
      masked::select(a,
                     masked::subset<short_joint_domain>::from_bits_asserted(
                         0b0111)) +
      masked::select(b,
                     masked::subset<short_joint_domain>::from_bits_asserted(
                         0b0011)));

  EXPECT_EQ(result.result.status, masked::eval_status::mask_mismatch);
  EXPECT_TRUE(result.values.empty());
}

TEST(MaskedTest, CompileTimeMaskProvidesStaticOutputArray) {
  std::array<int, joint_domain::size> a{1, 2, 3, 4, 5};

  const auto out =
      masked::unchecked_eval_array(masked::select<joint_domain, 0b10101>(a) *
                                   2);

  static_assert(std::tuple_size_v<decltype(out)> == 3);
  static_assert(std::is_same_v<typename decltype(out)::value_type, int>);
  EXPECT_EQ(out[0], 2);
  EXPECT_EQ(out[1], 6);
  EXPECT_EQ(out[2], 10);
}

TEST(MaskedTest, EvalArrayAllowsExplicitOutputType) {
  std::array<int, short_joint_domain::size> a{1, 2, 3, 4};

  const auto out = masked::unchecked_eval_array<double>(
      masked::select<short_joint_domain, 0b1010>(a) / 2.0);

  static_assert(std::is_same_v<typename decltype(out)::value_type, double>);
  EXPECT_DOUBLE_EQ(out[0], 1.0);
  EXPECT_DOUBLE_EQ(out[1], 2.0);
}

TEST(MaskedTest, CompileTimeMaskExposesEmptyAndFullProperties) {
  std::array<int, 3> a{1, 2, 3};
  using tiny_domain = masked::index_domain<struct tiny_tag, 3, std::uint8_t>;

  [[maybe_unused]] const auto empty = masked::select<tiny_domain, 0b000>(a);
  [[maybe_unused]] const auto full = masked::select<tiny_domain, 0b111>(a);
  [[maybe_unused]] const auto partial = masked::select<tiny_domain, 0b011>(a);

  static_assert(decltype(empty)::selected().empty());
  static_assert(!decltype(empty)::selected().is_full());
  static_assert(!decltype(full)::selected().empty());
  static_assert(decltype(full)::selected().is_full());
  static_assert(!decltype(partial)::selected().empty());
  static_assert(!decltype(partial)::selected().is_full());
}

TEST(MaskedTest, RestrictToCanNarrowASelectionExplicitly) {
  std::array<int, short_joint_domain::size> a{1, 2, 3, 4};
  std::array<int, 2> out{};

  masked::unchecked_eval_to(
      out.begin(),
      masked::restrict_to(
          masked::select_all<short_joint_domain>(a),
          masked::subset<short_joint_domain>::from_bits_asserted(0b1010)));

  EXPECT_EQ(out[0], 2);
  EXPECT_EQ(out[1], 4);
}

TEST(MaskedTest, SelectAllUsesEntireDomain) {
  std::array<int, short_joint_domain::size> a{4, 3, 2, 1};
  std::array<int, short_joint_domain::size> out{};

  masked::unchecked_eval_to(out.begin(),
                            masked::select_all<short_joint_domain>(a));

  EXPECT_EQ(out, a);
}

TEST(MaskedTest, CheckedScatterWritesBackToOriginalIndices) {
  std::array<int, joint_domain::size> base{10, 20, 30, 40, 50};
  const auto active = masked::subset<joint_domain>::from_bits_asserted(0b10011);

  const auto result =
      masked::checked_scatter_to(base, masked::select(base, active) + 5);

  EXPECT_EQ(result.status, masked::eval_status::ok);
  EXPECT_EQ(base, (std::array<int, joint_domain::size>{15, 25, 30, 40, 55}));
}

TEST(MaskedTest, SelectedSizeReportsNumberOfChosenIndices) {
  std::array<int, joint_domain::size> values{1, 2, 3, 4, 5};
  const auto expr =
      masked::select(values,
                     masked::subset<joint_domain>::from_bits_asserted(0b10011));

  EXPECT_EQ(masked::selected_size(expr), 3U);
}

} // namespace
