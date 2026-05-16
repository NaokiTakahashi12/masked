#pragma once

#include <masked/expressions.hpp>

namespace masked::detail {

template <class T, domain Domain>
struct is_expression<subset_view<T, Domain>> : std::true_type {};

template <class T, domain Domain, typename Domain::mask_type Mask>
struct is_expression<static_subset_view<T, Domain, Mask>> : std::true_type {};

template <class T> struct is_expression<scalar_expr<T>> : std::true_type {};

template <expression Lhs, expression Rhs, class Op>
struct is_expression<binary_expr<Lhs, Rhs, Op>> : std::true_type {};

template <expression Expr, class Op>
struct is_expression<unary_expr<Expr, Op>> : std::true_type {};

template <expression Expr>
struct is_expression<restricted_expr<Expr>> : std::true_type {};

template <expression Expr, typename Expr::mask_type Mask>
struct is_expression<static_restricted_expr<Expr, Mask>> : std::true_type {};

template <class T> [[nodiscard]] constexpr auto as_expr(T&& value) {
  if constexpr (expression<T>) {
    return std::forward<T>(value);
  } else {
    return scalar_expr<std::decay_t<T>>(std::forward<T>(value));
  }
}

template <std::size_t Index>
[[nodiscard]] constexpr auto
index_value(std::integral_constant<std::size_t, Index>) noexcept
    -> std::size_t {
  return Index;
}

template <domain Domain>
[[nodiscard]] constexpr auto index_value(typed_index<Domain> index) noexcept
    -> std::size_t {
  return index.value();
}

template <expression Expr, class Fn>
constexpr void unchecked_for_each_index_value(const Expr& expr, Fn&& fn) {
  static_assert(Expr::has_sequence,
                "masked::unchecked_for_each_index_value requires an "
                "expression containing select()");

  if constexpr (Expr::has_static_mask) {
    auto emit = [&](auto index_constant) {
      constexpr auto index = decltype(index_constant)::value;
      std::forward<Fn>(fn)(index_constant, expr.unchecked_value_at(index));
    };
    for_each_static_index<typename Expr::domain_type, Expr::static_mask>(emit);
  } else {
    using domain_type = typename Expr::domain_type;
    if (expr.mask_value() == domain_type::full_mask()) {
      for (std::size_t index = 0; index < domain_type::size; ++index) {
        std::forward<Fn>(fn)(typed_index<domain_type>::unchecked(index),
                             expr.unchecked_value_at(index));
      }
      return;
    }

    auto state = expr.make_state();
    auto cursor = mask_cursor<typename Expr::mask_type>(expr.mask_value());
    const auto size = expr.size();
    for (std::size_t rank = 0; rank < size; ++rank) {
      const auto index = cursor.next();
      std::forward<Fn>(fn)(typed_index<domain_type>::unchecked(index),
                           expr.value(state));
      if (rank + 1 < size) {
        expr.advance(state);
      }
    }
  }
}

} // namespace masked::detail
