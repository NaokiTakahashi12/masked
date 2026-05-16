#pragma once

#include <masked/domain.hpp>

namespace masked::detail {

template <class Lhs, class Rhs> struct common_domain {
  static_assert(expression<Lhs> && expression<Rhs>);

  using type = std::conditional_t<
      Lhs::has_sequence, typename Lhs::domain_type,
      std::conditional_t<Rhs::has_sequence, typename Rhs::domain_type, void>>;

  static_assert(
      !(Lhs::has_sequence && Rhs::has_sequence) ||
          std::same_as<typename Lhs::domain_type, typename Rhs::domain_type>,
      "masked expressions cannot combine different index domains");
};

template <class Lhs, class Rhs>
using common_domain_t = typename common_domain<Lhs, Rhs>::type;

template <class Lhs, class Rhs> consteval auto static_masks_match() -> bool {
  if constexpr (Lhs::has_sequence && Rhs::has_sequence &&
                Lhs::has_static_mask && Rhs::has_static_mask) {
    return Lhs::static_mask == Rhs::static_mask;
  } else {
    return false;
  }
}

template <class Lhs, class Rhs>
inline constexpr bool combined_has_static_mask_v =
    (Lhs::has_sequence && !Rhs::has_sequence && Lhs::has_static_mask) ||
    (Rhs::has_sequence && !Lhs::has_sequence && Rhs::has_static_mask) ||
    static_masks_match<Lhs, Rhs>();

template <class Lhs, class Rhs>
consteval auto combined_static_extent() -> std::size_t {
  if constexpr (Lhs::has_sequence && Rhs::has_sequence &&
                Lhs::has_static_mask && Rhs::has_static_mask) {
    static_assert(Lhs::static_mask == Rhs::static_mask,
                  "masked expressions use by-index semantics; compile-time "
                  "sequence masks must be equal");
    return Lhs::static_extent;
  } else if constexpr (Lhs::has_sequence && !Rhs::has_sequence &&
                       Lhs::has_static_mask) {
    return Lhs::static_extent;
  } else if constexpr (Rhs::has_sequence && !Lhs::has_sequence &&
                       Rhs::has_static_mask) {
    return Rhs::static_extent;
  } else {
    return masked::dynamic_extent;
  }
}

template <class Lhs, class Rhs>
consteval auto combined_static_mask()
    -> mask_type_or_t<common_domain_t<Lhs, Rhs>> {
  using mask_type = mask_type_or_t<common_domain_t<Lhs, Rhs>>;
  if constexpr (Lhs::has_sequence && Lhs::has_static_mask) {
    return static_cast<mask_type>(Lhs::static_mask);
  } else if constexpr (Rhs::has_sequence && Rhs::has_static_mask) {
    return static_cast<mask_type>(Rhs::static_mask);
  } else {
    return mask_type{0};
  }
}

struct plus_op {
  template <class Lhs, class Rhs>
  [[nodiscard]] constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      noexcept(noexcept(std::forward<Lhs>(lhs) + std::forward<Rhs>(rhs)))
          -> decltype(std::forward<Lhs>(lhs) + std::forward<Rhs>(rhs)) {
    return std::forward<Lhs>(lhs) + std::forward<Rhs>(rhs);
  }
};

struct minus_op {
  template <class Lhs, class Rhs>
  [[nodiscard]] constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      noexcept(noexcept(std::forward<Lhs>(lhs) - std::forward<Rhs>(rhs)))
          -> decltype(std::forward<Lhs>(lhs) - std::forward<Rhs>(rhs)) {
    return std::forward<Lhs>(lhs) - std::forward<Rhs>(rhs);
  }
};

struct multiply_op {
  template <class Lhs, class Rhs>
  [[nodiscard]] constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      noexcept(noexcept(std::forward<Lhs>(lhs) * std::forward<Rhs>(rhs)))
          -> decltype(std::forward<Lhs>(lhs) * std::forward<Rhs>(rhs)) {
    return std::forward<Lhs>(lhs) * std::forward<Rhs>(rhs);
  }
};

struct divide_op {
  template <class Lhs, class Rhs>
  [[nodiscard]] constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      noexcept(noexcept(std::forward<Lhs>(lhs) / std::forward<Rhs>(rhs)))
          -> decltype(std::forward<Lhs>(lhs) / std::forward<Rhs>(rhs)) {
    return std::forward<Lhs>(lhs) / std::forward<Rhs>(rhs);
  }
};

struct negate_op {
  template <class Value>
  [[nodiscard]] constexpr auto operator()(Value&& value) const
      noexcept(noexcept(-std::forward<Value>(value)))
          -> decltype(-std::forward<Value>(value)) {
    return -std::forward<Value>(value);
  }
};

} // namespace masked::detail
