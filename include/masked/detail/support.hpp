#pragma once

#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

#include <masked/fwd.hpp>

namespace masked::detail {

template <class T>
concept mask_integer =
    std::unsigned_integral<T> && !std::same_as<std::remove_cv_t<T>, bool>;

template <class T>
concept lvalue_contiguous_range = requires(T& values) {
  std::data(values);
  std::size(values);
};

template <class T>
concept readable_contiguous_range = requires(const T& values) {
  std::data(values);
  std::size(values);
};

template <class T>
struct static_extent_of
    : std::integral_constant<std::size_t, masked::dynamic_extent> {};

template <class T, std::size_t Extent>
struct static_extent_of<T[Extent]>
    : std::integral_constant<std::size_t, Extent> {};

template <class T, std::size_t Extent>
struct static_extent_of<std::array<T, Extent>>
    : std::integral_constant<std::size_t, Extent> {};

template <class T, std::size_t Extent>
struct static_extent_of<std::span<T, Extent>>
    : std::integral_constant<std::size_t, Extent> {};

template <class T>
inline constexpr std::size_t static_extent_of_v =
    static_extent_of<std::remove_cv_t<std::remove_reference_t<T>>>::value;

template <mask_integer UInt>
[[nodiscard]] constexpr auto popcount(UInt mask) noexcept -> std::size_t {
  return static_cast<std::size_t>(std::popcount(mask));
}

template <mask_integer UInt>
[[nodiscard]] constexpr auto representable_full_mask(std::size_t size) noexcept
    -> UInt {
  constexpr auto digits =
      static_cast<std::size_t>(std::numeric_limits<UInt>::digits);
  return size >= digits ? static_cast<UInt>(~UInt{0})
                        : static_cast<UInt>((UInt{1} << size) - UInt{1});
}

template <mask_integer UInt>
[[nodiscard]] constexpr auto has_out_of_range_bits(UInt mask,
                                                   std::size_t size) noexcept
    -> bool {
  constexpr auto digits =
      static_cast<std::size_t>(std::numeric_limits<UInt>::digits);
  if (size >= digits) {
    return false;
  }
  if (size == 0) {
    return mask != UInt{0};
  }
  const auto allowed = representable_full_mask<UInt>(size);
  return (mask & static_cast<UInt>(~allowed)) != UInt{0};
}

template <class T> struct is_domain : std::false_type {};

template <class T>
concept domain = is_domain<std::remove_cvref_t<T>>::value;

template <class T> struct is_expression : std::false_type {};

template <class T>
concept expression = is_expression<std::remove_cvref_t<T>>::value;

template <mask_integer UInt> class mask_cursor {
public:
  constexpr explicit mask_cursor(UInt mask) noexcept : remaining_(mask) {}

  [[nodiscard]] constexpr auto has_next() const noexcept -> bool {
    return remaining_ != UInt{0};
  }

  [[nodiscard]] constexpr auto next() noexcept -> std::size_t {
    assert(has_next());
    const auto bit = static_cast<std::size_t>(std::countr_zero(remaining_));
    remaining_ =
        static_cast<UInt>(remaining_ & static_cast<UInt>(remaining_ - UInt{1}));
    return bit;
  }

private:
  UInt remaining_;
};

template <class R>
using range_element_t =
    std::remove_pointer_t<decltype(std::data(std::declval<R&>()))>;

template <class T>
using output_value_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class Requested, class Inferred>
using requested_or_inferred_t =
    std::conditional_t<std::is_void_v<Requested>, Inferred, Requested>;

template <class Domain, class Fallback = std::uint64_t> struct mask_type_or {
  using type = typename Domain::mask_type;
};

template <class Fallback> struct mask_type_or<void, Fallback> {
  using type = Fallback;
};

template <class Domain, class Fallback = std::uint64_t>
using mask_type_or_t = typename mask_type_or<Domain, Fallback>::type;

} // namespace masked::detail
