#pragma once

#include <cstddef>
#include <cstdint>

namespace masked {

inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

enum class eval_status {
  ok,
  mask_out_of_range,
  mask_mismatch,
  source_too_small,
  output_too_small,
  compact_input_size_mismatch,
  empty_selection,
  no_selected_sequence,
};

struct eval_result {
  eval_status status{eval_status::ok};
  std::size_t selected_size{0};
  std::size_t required_source_size{0};
  std::size_t required_output_size{0};

  [[nodiscard]] constexpr auto operator==(eval_status expected) const noexcept
      -> bool {
    return status == expected;
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return status == eval_status::ok;
  }
};

template <class Tag, std::size_t Size, class UInt = std::uint64_t>
struct index_domain;

template <class Domain> class typed_index;
template <class Domain> class subset;

template <class Domain, typename Domain::mask_type Mask> struct subset_c;

template <class T, class Domain> class subset_view;

template <class T, class Domain, typename Domain::mask_type Mask>
class static_subset_view;

template <class T> class scalar_expr;
template <class Lhs, class Rhs, class Op> class binary_expr;
template <class Expr, class Op> class unary_expr;
template <class Expr> class restricted_expr;

template <class Expr, typename Expr::mask_type Mask>
class static_restricted_expr;

} // namespace masked
