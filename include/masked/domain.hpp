#pragma once

#include <array>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>

#include <masked/detail/domain_support.hpp>

namespace masked {

template <class Tag, std::size_t Size, class UInt> struct index_domain {
  static_assert(detail::mask_integer<UInt>,
                "masked::index_domain mask type must be an unsigned integer");

  using tag_type = Tag;
  using mask_type = UInt;
  static constexpr std::size_t size = Size;
  static constexpr std::size_t digits =
      static_cast<std::size_t>(std::numeric_limits<mask_type>::digits);

  static_assert(size <= digits,
                "masked::index_domain size must fit into the mask type");

  [[nodiscard]] static constexpr auto full_mask() noexcept -> mask_type {
    return detail::representable_full_mask<mask_type>(size);
  }

  [[nodiscard]] static constexpr auto valid_mask(mask_type mask) noexcept
      -> bool {
    return !detail::has_out_of_range_bits(mask, size);
  }

  [[nodiscard]] static constexpr auto valid_index(std::size_t index) noexcept
      -> bool {
    return index < size;
  }
};

template <detail::domain Domain, class T>
using domain_array = std::array<T, Domain::size>;

template <detail::domain Domain, class T>
using domain_span = std::span<T, Domain::size>;

template <class Domain> class typed_index {
public:
  static_assert(detail::domain<Domain>,
                "masked::typed_index requires a masked index domain");

  using domain_type = Domain;

  [[nodiscard]] static constexpr auto from_index(std::size_t value) noexcept
      -> std::optional<typed_index> {
    if (!Domain::valid_index(value)) {
      return std::nullopt;
    }
    return typed_index(value, unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto
  from_index_asserted(std::size_t value) noexcept -> typed_index {
    assert(Domain::valid_index(value));
    return typed_index(value, unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto unchecked(std::size_t value) noexcept
      -> typed_index {
    return typed_index(value, unchecked_tag{});
  }

  [[nodiscard]] constexpr auto value() const noexcept -> std::size_t {
    return value_;
  }

  [[nodiscard]] friend constexpr auto operator==(typed_index lhs,
                                                 typed_index rhs) noexcept
      -> bool {
    return lhs.value_ == rhs.value_;
  }

  [[nodiscard]] friend constexpr auto operator!=(typed_index lhs,
                                                 typed_index rhs) noexcept
      -> bool {
    return !(lhs == rhs);
  }

private:
  struct unchecked_tag {};

  constexpr typed_index(std::size_t value, unchecked_tag) noexcept
      : value_(value) {}

  std::size_t value_{};
};

template <detail::domain Domain>
[[nodiscard]] constexpr auto make_index(std::size_t value) noexcept
    -> std::optional<typed_index<Domain>> {
  return typed_index<Domain>::from_index(value);
}

template <detail::domain Domain>
[[nodiscard]] constexpr auto make_index_asserted(std::size_t value) noexcept
    -> typed_index<Domain> {
  return typed_index<Domain>::from_index_asserted(value);
}

template <class Domain> class subset {
public:
  static_assert(detail::domain<Domain>,
                "masked::subset requires a masked index domain");

  using domain_type = Domain;
  using mask_type = typename Domain::mask_type;

  constexpr subset() noexcept = default;

  [[nodiscard]] static constexpr auto none() noexcept -> subset {
    return subset(mask_type{0}, unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto all() noexcept -> subset {
    return subset(Domain::full_mask(), unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto from_bits(mask_type mask) noexcept
      -> std::optional<subset> {
    if (!Domain::valid_mask(mask)) {
      return std::nullopt;
    }
    return subset(mask, unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto
  from_bits_asserted(mask_type mask) noexcept -> subset {
    assert(Domain::valid_mask(mask));
    return subset(mask, unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto unchecked(mask_type mask) noexcept
      -> subset {
    return subset(mask, unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto singleton(std::size_t index) noexcept
      -> subset {
    assert(Domain::valid_index(index));
    return subset(static_cast<mask_type>(mask_type{1} << index),
                  unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto
  singleton(typed_index<Domain> index) noexcept -> subset {
    return singleton(index.value());
  }

  [[nodiscard]] static constexpr auto
  from_indices(std::initializer_list<std::size_t> indices) noexcept -> subset {
    mask_type mask = 0;
    for (const auto index : indices) {
      assert(Domain::valid_index(index));
      mask = static_cast<mask_type>(
          mask | static_cast<mask_type>(mask_type{1} << index));
    }
    return subset(mask, unchecked_tag{});
  }

  [[nodiscard]] static constexpr auto
  from_indices(std::initializer_list<typed_index<Domain>> indices) noexcept
      -> subset {
    mask_type mask = 0;
    for (const auto index : indices) {
      mask = static_cast<mask_type>(
          mask | static_cast<mask_type>(mask_type{1} << index.value()));
    }
    return subset(mask, unchecked_tag{});
  }

  [[nodiscard]] constexpr auto raw() const noexcept -> mask_type {
    return mask_;
  }

  [[nodiscard]] constexpr auto in_range() const noexcept -> bool {
    return Domain::valid_mask(mask_);
  }

  [[nodiscard]] constexpr auto count() const noexcept -> std::size_t {
    return detail::popcount(mask_);
  }

  [[nodiscard]] constexpr auto empty() const noexcept -> bool {
    return mask_ == mask_type{0};
  }

  [[nodiscard]] constexpr auto contains(std::size_t index) const noexcept
      -> bool {
    assert(Domain::valid_index(index));
    return (mask_ & static_cast<mask_type>(mask_type{1} << index)) !=
           mask_type{0};
  }

  [[nodiscard]] constexpr auto
  contains(typed_index<Domain> index) const noexcept -> bool {
    return contains(index.value());
  }

  [[nodiscard]] constexpr auto is_full() const noexcept -> bool {
    return mask_ == Domain::full_mask();
  }

  [[nodiscard]] friend constexpr auto operator==(subset lhs,
                                                 subset rhs) noexcept -> bool {
    return lhs.mask_ == rhs.mask_;
  }

  [[nodiscard]] friend constexpr auto operator!=(subset lhs,
                                                 subset rhs) noexcept -> bool {
    return !(lhs == rhs);
  }

  [[nodiscard]] friend constexpr auto operator|(subset lhs, subset rhs) noexcept
      -> subset {
    return subset(static_cast<mask_type>(lhs.mask_ | rhs.mask_),
                  unchecked_tag{});
  }

  [[nodiscard]] friend constexpr auto operator&(subset lhs, subset rhs) noexcept
      -> subset {
    return subset(static_cast<mask_type>(lhs.mask_ & rhs.mask_),
                  unchecked_tag{});
  }

  [[nodiscard]] friend constexpr auto operator^(subset lhs, subset rhs) noexcept
      -> subset {
    return subset(static_cast<mask_type>(lhs.mask_ ^ rhs.mask_),
                  unchecked_tag{});
  }

  [[nodiscard]] friend constexpr auto difference(subset lhs,
                                                 subset rhs) noexcept
      -> subset {
    return subset(
        static_cast<mask_type>(lhs.mask_ & static_cast<mask_type>(~rhs.mask_)),
        unchecked_tag{});
  }

  [[nodiscard]] friend constexpr auto intersects(subset lhs,
                                                 subset rhs) noexcept -> bool {
    return (lhs.mask_ & rhs.mask_) != mask_type{0};
  }

  [[nodiscard]] friend constexpr auto includes(subset superset,
                                               subset subset_value) noexcept
      -> bool {
    return (superset.mask_ & subset_value.mask_) == subset_value.mask_;
  }

  [[nodiscard]] constexpr auto complement() const noexcept -> subset {
    return subset(static_cast<mask_type>(Domain::full_mask() &
                                         static_cast<mask_type>(~mask_)),
                  unchecked_tag{});
  }

private:
  struct unchecked_tag {};

  constexpr explicit subset(mask_type mask, unchecked_tag) noexcept
      : mask_(mask) {}

  mask_type mask_{0};
};

template <class Domain, typename Domain::mask_type Mask> struct subset_c {
  static_assert(detail::domain<Domain>,
                "masked::subset_c requires a masked index domain");

  using domain_type = Domain;
  using mask_type = typename Domain::mask_type;

  static_assert(Domain::valid_mask(Mask),
                "masked::subset_c selects an index outside its domain");

  static constexpr mask_type value = Mask;
  static constexpr std::size_t static_extent = detail::popcount(value);
  static constexpr bool is_empty = value == mask_type{0};
  static constexpr bool is_full = value == Domain::full_mask();

  [[nodiscard]] static constexpr auto as_subset() noexcept -> subset<Domain> {
    return subset<Domain>::unchecked(value);
  }

  [[nodiscard]] constexpr operator subset<Domain>() const noexcept {
    return as_subset();
  }
};

template <detail::domain Domain>
[[nodiscard]] constexpr auto none() noexcept -> subset<Domain> {
  return subset<Domain>::none();
}

template <detail::domain Domain>
[[nodiscard]] constexpr auto all() noexcept -> subset<Domain> {
  return subset<Domain>::all();
}

template <detail::domain Domain>
[[nodiscard]] constexpr auto singleton(std::size_t index) noexcept
    -> subset<Domain> {
  return subset<Domain>::singleton(index);
}

template <detail::domain Domain, typename Domain::mask_type Mask>
[[nodiscard]] constexpr auto subset_constant() noexcept
    -> subset_c<Domain, Mask> {
  return {};
}

template <detail::domain Domain, typename Domain::mask_type Lhs,
          typename Domain::mask_type Rhs>
[[nodiscard]] constexpr auto operator|(subset_c<Domain, Lhs>,
                                       subset_c<Domain, Rhs>) noexcept
    -> subset_c<Domain, static_cast<typename Domain::mask_type>(Lhs | Rhs)> {
  return {};
}

template <detail::domain Domain, typename Domain::mask_type Lhs,
          typename Domain::mask_type Rhs>
[[nodiscard]] constexpr auto operator&(subset_c<Domain, Lhs>,
                                       subset_c<Domain, Rhs>) noexcept
    -> subset_c<Domain, static_cast<typename Domain::mask_type>(Lhs& Rhs)> {
  return {};
}

template <detail::domain Domain, typename Domain::mask_type Lhs,
          typename Domain::mask_type Rhs>
[[nodiscard]] constexpr auto operator^(subset_c<Domain, Lhs>,
                                       subset_c<Domain, Rhs>) noexcept
    -> subset_c<Domain, static_cast<typename Domain::mask_type>(Lhs ^ Rhs)> {
  return {};
}

template <detail::domain Domain, typename Domain::mask_type Lhs,
          typename Domain::mask_type Rhs>
[[nodiscard]] constexpr auto difference(subset_c<Domain, Lhs>,
                                        subset_c<Domain, Rhs>) noexcept
    -> subset_c<Domain,
                static_cast<typename Domain::mask_type>(
                    Lhs& static_cast<typename Domain::mask_type>(~Rhs))> {
  return {};
}

template <detail::domain Domain, typename Domain::mask_type Mask>
[[nodiscard]] constexpr auto complement(subset_c<Domain, Mask>) noexcept
    -> subset_c<Domain, static_cast<typename Domain::mask_type>(
                            Domain::full_mask() &
                            static_cast<typename Domain::mask_type>(~Mask))> {
  return {};
}

template <detail::domain Domain, typename Domain::mask_type Lhs,
          typename Domain::mask_type Rhs>
[[nodiscard]] constexpr auto intersects(subset_c<Domain, Lhs>,
                                        subset_c<Domain, Rhs>) noexcept
    -> bool {
  return (Lhs & Rhs) != typename Domain::mask_type{0};
}

template <detail::domain Domain, typename Domain::mask_type Superset,
          typename Domain::mask_type SubsetValue>
[[nodiscard]] constexpr auto includes(subset_c<Domain, Superset>,
                                      subset_c<Domain, SubsetValue>) noexcept
    -> bool {
  return (Superset & SubsetValue) == SubsetValue;
}

template <detail::domain Domain, class Fn>
constexpr void for_each_index(subset<Domain> selected, Fn&& fn) {
  assert(selected.in_range());
  if (selected.is_full()) {
    for (std::size_t index = 0; index < Domain::size; ++index) {
      std::forward<Fn>(fn)(typed_index<Domain>::unchecked(index));
    }
    return;
  }

  auto cursor = detail::mask_cursor<typename Domain::mask_type>(selected.raw());
  while (cursor.has_next()) {
    std::forward<Fn>(fn)(typed_index<Domain>::unchecked(cursor.next()));
  }
}

template <detail::domain Domain, typename Domain::mask_type Mask, class Fn>
constexpr void for_each_index(subset_c<Domain, Mask>, Fn&& fn) {
  detail::for_each_static_index<Domain, Mask>(std::forward<Fn>(fn));
}

template <detail::domain Domain, class Pred>
[[nodiscard]] constexpr auto make_subset_if(Pred&& pred) -> subset<Domain> {
  typename Domain::mask_type mask = 0;
  for (std::size_t index = 0; index < Domain::size; ++index) {
    const auto typed = typed_index<Domain>::unchecked(index);
    if (std::forward<Pred>(pred)(typed)) {
      mask = static_cast<typename Domain::mask_type>(
          mask | static_cast<typename Domain::mask_type>(
                     typename Domain::mask_type{1} << index));
    }
  }
  return subset<Domain>::unchecked(mask);
}

} // namespace masked
