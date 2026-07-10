#pragma once

#include <array>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <masked/selection.hpp>

namespace masked {

template <class Out, detail::expression Expr>
constexpr auto unchecked_eval_to(Out out, const Expr& expr) -> Out {
  auto write = [&](auto, auto&& value) {
    *out = std::forward<decltype(value)>(value);
    ++out;
  };
  detail::unchecked_for_each_index_value(expr, write);
  return out;
}

template <class T, std::size_t Extent, detail::expression Expr>
constexpr auto checked_eval_to(std::span<T, Extent> out, const Expr& expr)
    -> eval_result {
  const auto validation = validate(expr);
  if (!validation) {
    return validation;
  }
  if (out.size() < validation.selected_size) {
    return {eval_status::output_too_small, validation.selected_size, 0,
            validation.selected_size};
  }
  unchecked_eval_to(out.begin(), expr);
  return validation;
}

template <detail::lvalue_contiguous_range R, detail::expression Expr>
constexpr auto checked_eval_to(R& out, const Expr& expr) -> eval_result {
  using element_type = detail::range_element_t<R>;
  return checked_eval_to(
      std::span<element_type>(std::data(out), std::size(out)), expr);
}

template <class T = void, detail::expression Expr>
[[nodiscard]] auto unchecked_materialize(const Expr& expr) {
  static_assert(
      Expr::has_sequence,
      "masked::unchecked_materialize requires an expression containing "
      "select()");
  using output_type =
      detail::requested_or_inferred_t<T, typename Expr::value_type>;
  std::vector<output_type> out(expr.size());
  unchecked_eval_to(out.begin(), expr);
  return out;
}

template <class T = void, detail::expression Expr>
[[nodiscard]] auto checked_materialize(const Expr& expr) {
  static_assert(
      Expr::has_sequence,
      "masked::checked_materialize requires an expression containing select()");
  using output_type =
      detail::requested_or_inferred_t<T, typename Expr::value_type>;
  auto validation = validate(expr);
  if (!validation) {
    return materialize_result<output_type>{validation, {}};
  }
  std::vector<output_type> out(validation.selected_size);
  unchecked_eval_to(out.begin(), expr);
  return materialize_result<output_type>{validation, std::move(out)};
}

template <class T = void, detail::expression Expr>
[[nodiscard]] constexpr auto unchecked_eval_array(const Expr& expr) {
  static_assert(Expr::has_sequence, "masked::unchecked_eval_array requires an "
                                    "expression containing select()");
  static_assert(Expr::static_extent != dynamic_extent,
                "masked::unchecked_eval_array requires compile-time selected "
                "length");
  using output_type =
      detail::requested_or_inferred_t<T, typename Expr::value_type>;
  std::array<output_type, Expr::static_extent> out{};
  unchecked_eval_to(out.begin(), expr);
  return out;
}

template <class T = void, detail::expression Expr>
[[nodiscard]] constexpr auto checked_eval_array(const Expr& expr) {
  static_assert(
      Expr::has_sequence,
      "masked::checked_eval_array requires an expression containing select()");
  static_assert(Expr::static_extent != dynamic_extent,
                "masked::checked_eval_array requires compile-time selected "
                "length");
  using output_type =
      detail::requested_or_inferred_t<T, typename Expr::value_type>;
  array_result<output_type, Expr::static_extent> result{};
  result.result = validate(expr);
  if (!result) {
    return result;
  }
  unchecked_eval_to(result.values.begin(), expr);
  return result;
}

template <detail::lvalue_contiguous_range R, detail::expression Expr, class Fn>
constexpr auto checked_update_selected(R& out, const Expr& expr, Fn&& fn)
    -> eval_result {
  static_assert(
      Expr::has_sequence,
      "masked::checked_update_selected requires an expression containing "
      "select()");
  using domain_type = typename Expr::domain_type;
  using element_type = detail::range_element_t<R>;

  auto validation = validate(expr);
  if (!validation) {
    return validation;
  }
  if (std::size(out) < domain_type::size) {
    return {eval_status::output_too_small, validation.selected_size, 0,
            domain_type::size};
  }

  auto out_span = std::span<element_type>(std::data(out), std::size(out));

  auto update = [&](auto index_token, auto&& value) {
    const auto index = detail::index_value(index_token);
    fn(out_span[index], std::forward<decltype(value)>(value));
  };

  detail::unchecked_for_each_index_value(expr, update);
  return validation;
}

template <detail::lvalue_contiguous_range R, detail::expression Expr>
constexpr auto checked_scatter_to(R& out, const Expr& expr) -> eval_result {
  return checked_update_selected(out, expr,
                                 [](auto& dst, auto&& value) { dst = value; });
}

template <detail::lvalue_contiguous_range R, detail::expression Expr>
constexpr auto checked_add_to(R& out, const Expr& expr) -> eval_result {
  return checked_update_selected(out, expr,
                                 [](auto& dst, auto&& value) { dst += value; });
}

template <detail::lvalue_contiguous_range R, detail::expression Expr>
constexpr auto checked_subtract_from(R& out, const Expr& expr) -> eval_result {
  return checked_update_selected(out, expr,
                                 [](auto& dst, auto&& value) { dst -= value; });
}

template <class T = void, detail::expression Expr>
[[nodiscard]] constexpr auto checked_domain_array(
    const Expr& expr,
    detail::requested_or_inferred_t<T, typename Expr::value_type> fill = {}) {
  static_assert(
      Expr::has_sequence,
      "masked::checked_domain_array requires an expression containing "
      "select()");
  using domain_type = typename Expr::domain_type;
  using output_type =
      detail::requested_or_inferred_t<T, typename Expr::value_type>;
  array_result<output_type, domain_type::size> result{};
  result.result = validate(expr);
  for (auto& value : result.values) {
    value = fill;
  }
  if (!result) {
    return result;
  }

  auto write = [&](auto index_token, auto&& value) {
    result.values[detail::index_value(index_token)] =
        std::forward<decltype(value)>(value);
  };
  detail::unchecked_for_each_index_value(expr, write);
  return result;
}

template <detail::lvalue_contiguous_range R, detail::domain Domain, class Value>
constexpr auto checked_fill_selected(R& out, subset<Domain> selected,
                                     Value&& value) -> eval_result {
  if (!selected.in_range()) {
    return {eval_status::mask_out_of_range, selected.count(), 0, 0};
  }
  if (std::size(out) < Domain::size) {
    return {eval_status::output_too_small, selected.count(), 0, Domain::size};
  }

  using element_type = detail::range_element_t<R>;
  auto out_span = std::span<element_type>(std::data(out), std::size(out));
  for_each_index(selected,
                 [&](auto index) { out_span[index.value()] = value; });
  return {eval_status::ok, selected.count(), 0, 0};
}

template <detail::lvalue_contiguous_range Out,
          detail::readable_contiguous_range Compact, detail::domain Domain>
constexpr auto checked_scatter_compact_to(Out& out, const Compact& compact,
                                          subset<Domain> selected)
    -> eval_result {
  if (!selected.in_range()) {
    return {eval_status::mask_out_of_range, selected.count(), 0, 0};
  }
  if (std::size(out) < Domain::size) {
    return {eval_status::output_too_small, selected.count(), 0, Domain::size};
  }
  if (std::size(compact) < selected.count()) {
    return {eval_status::compact_input_size_mismatch, selected.count(), 0, 0};
  }

  using out_element_type = detail::range_element_t<Out>;
  auto out_span = std::span<out_element_type>(std::data(out), std::size(out));
  const auto* compact_data = std::data(compact);

  std::size_t rank = 0;
  for_each_index(selected, [&](auto index) {
    out_span[index.value()] = compact_data[rank++];
  });
  return {eval_status::ok, selected.count(), 0, 0};
}

template <class T = void, detail::readable_contiguous_range R,
          detail::domain Domain>
[[nodiscard]] auto checked_gather_compact_from(const R& values,
                                               subset<Domain> selected) {
  using source_type = std::remove_pointer_t<decltype(std::data(values))>;
  using output_type =
      detail::requested_or_inferred_t<T, detail::output_value_t<source_type>>;

  if (!selected.in_range()) {
    return materialize_result<output_type>{
        {eval_status::mask_out_of_range, selected.count(), 0, 0}, {}};
  }
  if (std::size(values) < Domain::size) {
    return materialize_result<output_type>{
        {eval_status::source_too_small, selected.count(), Domain::size, 0}, {}};
  }

  materialize_result<output_type> result{};
  result.result = {eval_status::ok, selected.count(), 0, 0};
  result.values.resize(selected.count());

  const auto* values_data = std::data(values);
  std::size_t rank = 0;
  for_each_index(selected, [&](auto index) {
    result.values[rank++] = values_data[index.value()];
  });
  return result;
}

template <class T = void, detail::readable_contiguous_range R,
          detail::domain Domain, typename Domain::mask_type Mask>
[[nodiscard]] constexpr auto
checked_gather_array_from(const R& values, subset_c<Domain, Mask> = {}) {
  using source_type = std::remove_pointer_t<decltype(std::data(values))>;
  using output_type =
      detail::requested_or_inferred_t<T, detail::output_value_t<source_type>>;

  array_result<output_type, detail::popcount(Mask)> result{};
  if (std::size(values) < Domain::size) {
    result.result = {eval_status::source_too_small, detail::popcount(Mask),
                     Domain::size, 0};
    return result;
  }

  result.result = {eval_status::ok, detail::popcount(Mask), 0, 0};
  const auto* values_data = std::data(values);
  std::size_t rank = 0;
  auto gather_static = [&](auto index_constant) {
    constexpr auto index = decltype(index_constant)::value;
    result.values[rank++] = values_data[index];
  };
  detail::for_each_static_index<Domain, Mask>(gather_static);
  return result;
}

template <detail::lvalue_contiguous_range R, detail::domain Domain, class Fn>
constexpr auto checked_transform_selected(R& out, subset<Domain> selected,
                                          Fn&& fn) -> eval_result {
  if (!selected.in_range()) {
    return {eval_status::mask_out_of_range, selected.count(), 0, 0};
  }
  if (std::size(out) < Domain::size) {
    return {eval_status::output_too_small, selected.count(), 0, Domain::size};
  }
  using element_type = detail::range_element_t<R>;
  auto out_span = std::span<element_type>(std::data(out), std::size(out));
  for_each_index(selected, [&](auto index) {
    out_span[index.value()] =
        std::forward<Fn>(fn)(index, out_span[index.value()]);
  });
  return {eval_status::ok, selected.count(), 0, 0};
}

template <detail::lvalue_contiguous_range R, detail::domain Domain,
          typename Domain::mask_type Mask, class Fn>
constexpr auto checked_transform_selected(R& out, subset_c<Domain, Mask>,
                                          Fn&& fn) -> eval_result {
  if (std::size(out) < Domain::size) {
    return {eval_status::output_too_small, detail::popcount(Mask), 0,
            Domain::size};
  }
  using element_type = detail::range_element_t<R>;
  auto out_span = std::span<element_type>(std::data(out), std::size(out));
  auto transform_static = [&](auto index_constant) {
    constexpr auto index = decltype(index_constant)::value;
    out_span[index] = std::forward<Fn>(fn)(index_constant, out_span[index]);
  };
  detail::for_each_static_index<Domain, Mask>(transform_static);
  return {eval_status::ok, detail::popcount(Mask), 0, 0};
}

template <class T = void, detail::expression Expr>
[[nodiscard]] auto unchecked_sum(const Expr& expr) {
  static_assert(
      Expr::has_sequence,
      "masked::unchecked_sum requires an expression containing select()");
  using output_type =
      detail::requested_or_inferred_t<T, typename Expr::value_type>;
  output_type total{};

  auto accumulate = [&](auto, auto&& value) {
    total =
        static_cast<output_type>(total + std::forward<decltype(value)>(value));
  };
  detail::unchecked_for_each_index_value(expr, accumulate);
  return total;
}

template <class T = void, detail::expression Expr>
[[nodiscard]] auto checked_sum(const Expr& expr) {
  static_assert(Expr::has_sequence,
                "masked::checked_sum requires an expression containing "
                "select()");
  using output_type =
      detail::requested_or_inferred_t<T, typename Expr::value_type>;
  auto validation = validate(expr);
  if (!validation) {
    return reduce_result<output_type>{validation, {}};
  }
  return reduce_result<output_type>{validation,
                                    unchecked_sum<output_type>(expr)};
}

template <class T = void, detail::expression Expr, class Compare>
[[nodiscard]] auto checked_min_max(const Expr& expr, Compare&& compare) {
  static_assert(Expr::has_sequence,
                "masked::checked_min_max requires an expression containing "
                "select()");
  using output_type =
      detail::requested_or_inferred_t<T, typename Expr::value_type>;

  const auto validation = validate(expr);
  if (!validation) {
    return reduce_result<output_type>{validation, {}};
  }
  if (validation.selected_size == 0) {
    return reduce_result<output_type>{{eval_status::empty_selection, 0, 0, 0},
                                      {}};
  }

  output_type best{};
  bool first = true;
  detail::unchecked_for_each_index_value(expr, [&](auto, auto&& value) {
    auto candidate = static_cast<output_type>(std::forward<decltype(value)>(value));
    if (first || compare(candidate, best)) {
      best = std::move(candidate);
      first = false;
    }
  });
  if (first) {
    return reduce_result<output_type>{{eval_status::empty_selection, 0, 0, 0},
                                      {}};
  }
  return reduce_result<output_type>{validation, std::move(best)};
}

template <class T = void, detail::expression Expr>
[[nodiscard]] auto checked_min(const Expr& expr) {
  return checked_min_max<T>(
      expr, [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
}

template <class T = void, detail::expression Expr>
[[nodiscard]] auto checked_max(const Expr& expr) {
  return checked_min_max<T>(
      expr, [](const auto& lhs, const auto& rhs) { return rhs < lhs; });
}

template <detail::expression Expr, class Compare>
[[nodiscard]] auto checked_arg_min_max(const Expr& expr, Compare&& compare) {
  static_assert(Expr::has_sequence,
                "masked::checked_arg_min_max requires an expression "
                "containing select()");
  using domain_type = typename Expr::domain_type;
  using index_type = typed_index<domain_type>;
  using result_type = reduce_result<std::optional<index_type>>;

  const auto validation = validate(expr);
  if (!validation) {
    return result_type{validation, std::nullopt};
  }
  if (validation.selected_size == 0) {
    return result_type{{eval_status::empty_selection, 0, 0, 0}, std::nullopt};
  }

  index_type best_index = index_type::unchecked(0);
  typename Expr::value_type best_value{};
  bool first = true;
  detail::unchecked_for_each_index_value(
      expr, [&](auto index_token, auto&& value) {
        using value_ref_type = decltype(value);
        if (first) {
          best_index = index_type::unchecked(detail::index_value(index_token));
          best_value = static_cast<typename Expr::value_type>(
              std::forward<value_ref_type>(value));
          first = false;
          return;
        }

        const auto candidate = static_cast<typename Expr::value_type>(
            std::forward<value_ref_type>(value));
        if (compare(candidate, best_value)) {
          best_index = index_type::unchecked(detail::index_value(index_token));
          best_value = candidate;
        }
      });

  return result_type{validation, std::optional<index_type>(best_index)};
}

template <detail::expression Expr>
[[nodiscard]] auto checked_arg_min(const Expr& expr) {
  return checked_arg_min_max(
      expr, [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
}

template <detail::expression Expr>
[[nodiscard]] auto checked_arg_max(const Expr& expr) {
  return checked_arg_min_max(
      expr, [](const auto& lhs, const auto& rhs) { return rhs < lhs; });
}

template <detail::expression Expr, class Pred>
[[nodiscard]] auto checked_find_if(const Expr& expr, Pred&& pred) {
  static_assert(Expr::has_sequence,
                "masked::checked_find_if requires an expression containing "
                "select()");
  using domain_type = typename Expr::domain_type;
  using index_type = typed_index<domain_type>;
  using result_type = reduce_result<std::optional<index_type>>;

  const auto validation = validate(expr);
  if (!validation) {
    return result_type{validation, std::nullopt};
  }
  if (validation.selected_size == 0) {
    return result_type{{eval_status::empty_selection, 0, 0, 0}, std::nullopt};
  }

  std::optional<index_type> found{};
  if constexpr (Expr::has_static_mask) {
    detail::for_each_static_index<typename Expr::domain_type, Expr::static_mask>(
        [&](auto index_constant) {
          if (found.has_value()) {
            return;
          }
          constexpr auto index = decltype(index_constant)::value;
          if (pred(expr.unchecked_value_at(index))) {
            found = index_type::unchecked(index);
          }
        });
  } else {
    using mask_type = typename Expr::mask_type;
    if (expr.mask_value() == domain_type::full_mask()) {
      for (std::size_t index = 0; index < domain_type::size; ++index) {
        if (pred(expr.unchecked_value_at(index))) {
          found = index_type::unchecked(index);
          break;
        }
      }
    } else {
      auto cursor = detail::mask_cursor<mask_type>(expr.mask_value());
      while (cursor.has_next()) {
        const auto index = cursor.next();
        if (pred(expr.unchecked_value_at(index))) {
          found = index_type::unchecked(index);
          break;
        }
      }
    }
  }
  return result_type{validation, found};
}

template <class T = void, class Lhs, class Rhs>
requires((detail::expression<Lhs> || detail::expression<Rhs>))
    [[nodiscard]] auto checked_dot(Lhs&& lhs, Rhs&& rhs) {
  auto expr = std::forward<Lhs>(lhs) * std::forward<Rhs>(rhs);
  return checked_sum<T>(expr);
}

template <class T = void, class Lhs, class Rhs>
requires((detail::expression<Lhs> || detail::expression<Rhs>))
    [[nodiscard]] auto unchecked_dot(Lhs&& lhs, Rhs&& rhs) {
  auto expr = std::forward<Lhs>(lhs) * std::forward<Rhs>(rhs);
  return unchecked_sum<T>(expr);
}

} // namespace masked
