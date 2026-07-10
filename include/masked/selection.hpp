#pragma once

#include <type_traits>
#include <utility>

#include <masked/detail/expression_runtime.hpp>

namespace masked {

template <detail::lvalue_contiguous_range R, detail::domain Domain>
[[nodiscard]] constexpr auto select(R& values, subset<Domain> selected) {
  using element_type = detail::range_element_t<R>;
  constexpr auto source_static_extent = detail::static_extent_of_v<R>;
  if constexpr (source_static_extent != dynamic_extent) {
    static_assert(source_static_extent >= Domain::size,
                  "masked::select(values, subset<Domain>) requires values to "
                  "cover the whole index domain");
  }
  return subset_view<element_type, Domain>(
      std::span<element_type>(std::data(values), std::size(values)), selected);
}

template <detail::domain Domain, typename Domain::mask_type Mask,
          detail::lvalue_contiguous_range R>
[[nodiscard]] constexpr auto select(R& values, subset_c<Domain, Mask> = {}) {
  using element_type = detail::range_element_t<R>;
  constexpr auto source_static_extent = detail::static_extent_of_v<R>;
  if constexpr (source_static_extent != dynamic_extent) {
    static_assert(source_static_extent >= Domain::size,
                  "masked::select<Domain, Mask>(values) requires values to "
                  "cover the whole index domain");
  }
  return static_subset_view<element_type, Domain, Mask>(
      std::span<element_type>(std::data(values), std::size(values)));
}

template <detail::domain Domain, detail::lvalue_contiguous_range R>
[[nodiscard]] constexpr auto select_all(R& values) {
  return select(values, subset<Domain>::all());
}

template <detail::domain Domain, detail::lvalue_contiguous_range R>
[[nodiscard]] constexpr auto select_none(R& values) {
  return select(values, subset<Domain>::none());
}

template <detail::domain Domain, std::size_t Index,
          detail::lvalue_contiguous_range R>
[[nodiscard]] constexpr auto select_one(R& values) {
  static_assert(Domain::valid_index(Index),
                "masked::select_one index is outside its domain");
  constexpr auto mask = static_cast<typename Domain::mask_type>(
      typename Domain::mask_type{1} << Index);
  return select<Domain, mask>(values);
}

template <detail::expression Expr>
[[nodiscard]] constexpr auto
restrict_to(Expr&& expr,
            subset<typename std::remove_cvref_t<Expr>::domain_type> selected) {
  using expr_type = decltype(detail::as_expr(std::forward<Expr>(expr)));
  auto expr_value = detail::as_expr(std::forward<Expr>(expr));
  return restricted_expr<expr_type>(std::move(expr_value), selected);
}

template <detail::expression Expr, detail::domain Domain,
          typename Domain::mask_type Mask>
requires std::same_as<typename std::remove_cvref_t<Expr>::domain_type, Domain>
[[nodiscard]] constexpr auto restrict_to(Expr&& expr, subset_c<Domain, Mask>) {
  using expr_type = decltype(detail::as_expr(std::forward<Expr>(expr)));
  auto expr_value = detail::as_expr(std::forward<Expr>(expr));
  return static_restricted_expr<expr_type, Mask>(std::move(expr_value));
}

template <class T> [[nodiscard]] constexpr auto scalar(T value) {
  return scalar_expr<std::decay_t<T>>(std::move(value));
}

template <class Lhs, class Rhs>
requires((detail::expression<Lhs> || detail::expression<Rhs>))
    [[nodiscard]] constexpr auto
    operator+(Lhs&& lhs, Rhs&& rhs) {
  auto lhs_expr = detail::as_expr(std::forward<Lhs>(lhs));
  auto rhs_expr = detail::as_expr(std::forward<Rhs>(rhs));
  return binary_expr<decltype(lhs_expr), decltype(rhs_expr), detail::plus_op>(
      std::move(lhs_expr), std::move(rhs_expr));
}

template <class Lhs, class Rhs>
requires((detail::expression<Lhs> || detail::expression<Rhs>))
    [[nodiscard]] constexpr auto
    operator-(Lhs&& lhs, Rhs&& rhs) {
  auto lhs_expr = detail::as_expr(std::forward<Lhs>(lhs));
  auto rhs_expr = detail::as_expr(std::forward<Rhs>(rhs));
  return binary_expr<decltype(lhs_expr), decltype(rhs_expr), detail::minus_op>(
      std::move(lhs_expr), std::move(rhs_expr));
}

template <class Lhs, class Rhs>
requires((detail::expression<Lhs> || detail::expression<Rhs>))
    [[nodiscard]] constexpr auto
    operator*(Lhs&& lhs, Rhs&& rhs) {
  auto lhs_expr = detail::as_expr(std::forward<Lhs>(lhs));
  auto rhs_expr = detail::as_expr(std::forward<Rhs>(rhs));
  return binary_expr<decltype(lhs_expr), decltype(rhs_expr),
                     detail::multiply_op>(std::move(lhs_expr),
                                          std::move(rhs_expr));
}

template <class Lhs, class Rhs>
requires((detail::expression<Lhs> || detail::expression<Rhs>))
    [[nodiscard]] constexpr auto
    operator/(Lhs&& lhs, Rhs&& rhs) {
  auto lhs_expr = detail::as_expr(std::forward<Lhs>(lhs));
  auto rhs_expr = detail::as_expr(std::forward<Rhs>(rhs));
  return binary_expr<decltype(lhs_expr), decltype(rhs_expr), detail::divide_op>(
      std::move(lhs_expr), std::move(rhs_expr));
}

template <detail::expression Expr>
[[nodiscard]] constexpr auto operator-(Expr&& expr) {
  auto expr_value = detail::as_expr(std::forward<Expr>(expr));
  return unary_expr<decltype(expr_value), detail::negate_op>(
      std::move(expr_value));
}

template <detail::expression Expr>
[[nodiscard]] constexpr auto selected_size(const Expr& expr) noexcept
    -> std::size_t {
  static_assert(Expr::has_sequence,
                "masked::selected_size requires an expression containing "
                "select()");
  return expr.size();
}

template <detail::expression Expr>
[[nodiscard]] constexpr auto selected_mask(const Expr& expr) noexcept ->
    typename Expr::mask_type {
  static_assert(Expr::has_sequence,
                "masked::selected_mask requires an expression containing "
                "select()");
  return expr.mask_value();
}

template <detail::expression Expr>
[[nodiscard]] constexpr auto validate(const Expr& expr) noexcept
    -> eval_result {
  if constexpr (!Expr::has_sequence) {
    return {eval_status::no_selected_sequence, 0, 0, 0};
  } else {
    return expr.validate();
  }
}

} // namespace masked
