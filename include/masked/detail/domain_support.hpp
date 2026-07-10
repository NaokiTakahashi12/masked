#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <masked/detail/support.hpp>

namespace masked::detail {

template <class Tag, std::size_t Size, mask_integer UInt>
struct is_domain<index_domain<Tag, Size, UInt>> : std::true_type {};

template <domain Domain, typename Domain::mask_type Mask, std::size_t Index,
          class Fn>
constexpr void for_each_static_index_impl(Fn&& fn) {
  if constexpr (Index < Domain::size) {
    constexpr auto bit = static_cast<typename Domain::mask_type>(
        typename Domain::mask_type{1} << Index);
    if constexpr ((Mask & bit) != typename Domain::mask_type{0}) {
      std::forward<Fn>(fn)(std::integral_constant<std::size_t, Index>{});
    }
    for_each_static_index_impl<Domain, Mask, Index + 1>(std::forward<Fn>(fn));
  }
}

template <domain Domain, typename Domain::mask_type Mask, class Fn>
constexpr void for_each_static_index(Fn&& fn) {
  for_each_static_index_impl<Domain, Mask, 0>(std::forward<Fn>(fn));
}

} // namespace masked::detail
