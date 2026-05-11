#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace masked {

inline constexpr std::size_t dynamic_extent = std::dynamic_extent;

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

template <class T> struct materialize_result {
  eval_result result{};
  std::vector<T> values{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return static_cast<bool>(result);
  }
};

template <class T, std::size_t Extent> struct array_result {
  eval_result result{};
  std::array<T, Extent> values{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return static_cast<bool>(result);
  }
};

template <class T> struct reduce_result {
  eval_result result{};
  T value{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return static_cast<bool>(result);
  }
};

namespace detail {

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
struct static_extent_of : std::integral_constant<std::size_t, dynamic_extent> {
};

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

} // namespace detail

template <class Tag, std::size_t Size,
          detail::mask_integer UInt = std::uint64_t>
struct index_domain {
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

namespace detail {

template <class Tag, std::size_t Size, mask_integer UInt>
struct is_domain<index_domain<Tag, Size, UInt>> : std::true_type {};

} // namespace detail

template <detail::domain Domain, class T>
using domain_array = std::array<T, Domain::size>;

template <detail::domain Domain, class T>
using domain_span = std::span<T, Domain::size>;

template <detail::domain Domain> class typed_index {
public:
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

template <detail::domain Domain> class subset {
public:
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

template <detail::domain Domain, typename Domain::mask_type Mask>
struct subset_c {
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

namespace detail {

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

} // namespace detail

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

template <class T, detail::domain Domain> class subset_view {
public:
  using element_type = T;
  using domain_type = Domain;
  using mask_type = typename Domain::mask_type;
  using eval_type = T&;
  using value_type = std::remove_cv_t<T>;
  static constexpr bool has_sequence = true;
  static constexpr bool has_static_mask = false;
  static constexpr std::size_t static_extent = dynamic_extent;

  struct state_type {
    detail::mask_cursor<mask_type> cursor;
    std::size_t index{0};
  };

  constexpr subset_view(std::span<T> values, subset<Domain> selected) noexcept
      : values_(values), selected_(selected) {}

  [[nodiscard]] constexpr auto selected() const noexcept -> subset<Domain> {
    return selected_;
  }

  [[nodiscard]] constexpr auto mask_value() const noexcept -> mask_type {
    return selected_.raw();
  }

  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return selected_.count();
  }

  [[nodiscard]] constexpr auto validate() const noexcept -> eval_result {
    if (!selected_.in_range()) {
      return {eval_status::mask_out_of_range, size(), 0, 0};
    }
    if (values_.size() < Domain::size) {
      return {eval_status::source_too_small, size(), Domain::size, 0};
    }
    return {eval_status::ok, size(), 0, 0};
  }

  [[nodiscard]] constexpr auto make_state() const noexcept -> state_type {
    auto state = state_type{detail::mask_cursor<mask_type>(selected_.raw()), 0};
    if (state.cursor.has_next()) {
      state.index = state.cursor.next();
    }
    return state;
  }

  [[nodiscard]] constexpr auto value(state_type& state) const noexcept -> T& {
    return values_[state.index];
  }

  [[nodiscard]] constexpr auto
  unchecked_value_at(std::size_t index) const noexcept -> T& {
    assert(selected_.contains(index));
    return values_[index];
  }

  constexpr void advance(state_type& state) const noexcept {
    state.index = state.cursor.next();
  }

private:
  std::span<T> values_;
  subset<Domain> selected_;
};

template <class T, detail::domain Domain, typename Domain::mask_type Mask>
class static_subset_view {
public:
  using element_type = T;
  using domain_type = Domain;
  using mask_type = typename Domain::mask_type;
  using eval_type = T&;
  using value_type = std::remove_cv_t<T>;
  static constexpr bool has_sequence = true;
  static constexpr bool has_static_mask = true;
  static constexpr mask_type static_mask = Mask;
  static constexpr std::size_t static_extent = detail::popcount(static_mask);

  static_assert(Domain::valid_mask(static_mask),
                "masked::select<Domain, Mask> selects an index outside its "
                "domain");

  struct state_type {
    detail::mask_cursor<mask_type> cursor;
    std::size_t index{0};
  };

  constexpr explicit static_subset_view(std::span<T> values) noexcept
      : values_(values) {}

  [[nodiscard]] static constexpr auto selected() noexcept -> subset<Domain> {
    return subset<Domain>::unchecked(static_mask);
  }

  [[nodiscard]] static constexpr auto mask_value() noexcept -> mask_type {
    return static_mask;
  }

  [[nodiscard]] static constexpr auto size() noexcept -> std::size_t {
    return static_extent;
  }

  [[nodiscard]] constexpr auto validate() const noexcept -> eval_result {
    if (values_.size() < Domain::size) {
      return {eval_status::source_too_small, static_extent, Domain::size, 0};
    }
    return {eval_status::ok, static_extent, 0, 0};
  }

  [[nodiscard]] constexpr auto make_state() const noexcept -> state_type {
    auto state = state_type{detail::mask_cursor<mask_type>(static_mask), 0};
    if (state.cursor.has_next()) {
      state.index = state.cursor.next();
    }
    return state;
  }

  [[nodiscard]] constexpr auto value(state_type& state) const noexcept -> T& {
    return values_[state.index];
  }

  [[nodiscard]] constexpr auto
  unchecked_value_at(std::size_t index) const noexcept -> T& {
    assert(Domain::valid_index(index));
    assert((static_mask & static_cast<mask_type>(mask_type{1} << index)) !=
           mask_type{0});
    return values_[index];
  }

  constexpr void advance(state_type& state) const noexcept {
    state.index = state.cursor.next();
  }

private:
  std::span<T> values_;
};

template <class T> class scalar_expr {
public:
  using domain_type = void;
  using value_type = std::remove_cv_t<T>;
  using eval_type = const value_type&;
  static constexpr bool has_sequence = false;
  static constexpr bool has_static_mask = false;
  static constexpr std::size_t static_extent = 0;

  struct state_type {};

  constexpr explicit scalar_expr(T value) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : value_(std::move(value)) {}

  [[nodiscard]] static constexpr auto size() noexcept -> std::size_t {
    return 0;
  }

  [[nodiscard]] static constexpr auto validate() noexcept -> eval_result {
    return {eval_status::ok, 0, 0, 0};
  }

  [[nodiscard]] static constexpr auto make_state() noexcept -> state_type {
    return {};
  }

  [[nodiscard]] constexpr auto value(state_type&) const noexcept
      -> const value_type& {
    return value_;
  }

  [[nodiscard]] constexpr auto unchecked_value_at(std::size_t) const noexcept
      -> const value_type& {
    return value_;
  }

  static constexpr void advance(state_type&) noexcept {}

private:
  value_type value_;
};

namespace detail {

template <class T, domain Domain>
struct is_expression<subset_view<T, Domain>> : std::true_type {};

template <class T, domain Domain, typename Domain::mask_type Mask>
struct is_expression<static_subset_view<T, Domain, Mask>> : std::true_type {};

template <class T> struct is_expression<scalar_expr<T>> : std::true_type {};

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
    return dynamic_extent;
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

template <class T> [[nodiscard]] constexpr auto as_expr(T&& value) {
  if constexpr (expression<T>) {
    return std::forward<T>(value);
  } else {
    return scalar_expr<std::decay_t<T>>(std::forward<T>(value));
  }
}

} // namespace detail

template <detail::expression Lhs, detail::expression Rhs, class Op>
class binary_expr {
public:
  using lhs_type = Lhs;
  using rhs_type = Rhs;
  using op_type = Op;
  using domain_type = detail::common_domain_t<Lhs, Rhs>;
  using eval_type = decltype(std::declval<const Op&>()(
      std::declval<typename Lhs::eval_type>(),
      std::declval<typename Rhs::eval_type>()));
  using value_type = detail::output_value_t<eval_type>;
  using mask_type = detail::mask_type_or_t<domain_type>;

  static constexpr bool has_sequence = Lhs::has_sequence || Rhs::has_sequence;
  static constexpr bool has_static_mask =
      detail::combined_has_static_mask_v<Lhs, Rhs>;
  static constexpr std::size_t static_extent =
      detail::combined_static_extent<Lhs, Rhs>();
  static constexpr mask_type static_mask = [] {
    if constexpr (has_static_mask) {
      return static_cast<mask_type>(detail::combined_static_mask<Lhs, Rhs>());
    } else {
      return mask_type{0};
    }
  }();

  struct state_type {
    typename Lhs::state_type lhs;
    typename Rhs::state_type rhs;
  };

  constexpr binary_expr(Lhs lhs, Rhs rhs) noexcept(
      std::is_nothrow_move_constructible_v<Lhs>&&
          std::is_nothrow_move_constructible_v<Rhs>)
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    if constexpr (Lhs::has_sequence) {
      return lhs_.size();
    } else if constexpr (Rhs::has_sequence) {
      return rhs_.size();
    } else {
      return 0;
    }
  }

  [[nodiscard]] constexpr auto mask_value() const noexcept -> mask_type {
    static_assert(has_sequence,
                  "masked::mask_value requires an expression containing "
                  "select()");
    if constexpr (Lhs::has_sequence) {
      return lhs_.mask_value();
    } else {
      return rhs_.mask_value();
    }
  }

  [[nodiscard]] constexpr auto validate() const noexcept -> eval_result {
    if (const auto lhs_status = lhs_.validate(); !lhs_status) {
      return lhs_status;
    }
    if (const auto rhs_status = rhs_.validate(); !rhs_status) {
      return rhs_status;
    }
    if constexpr (Lhs::has_sequence && Rhs::has_sequence) {
      if (lhs_.mask_value() != rhs_.mask_value()) {
        return {eval_status::mask_mismatch, size(), 0, 0};
      }
    }
    if constexpr (!has_sequence) {
      return {eval_status::no_selected_sequence, 0, 0, 0};
    } else {
      return {eval_status::ok, size(), 0, 0};
    }
  }

  [[nodiscard]] constexpr auto make_state() const noexcept -> state_type {
    return {lhs_.make_state(), rhs_.make_state()};
  }

  [[nodiscard]] constexpr auto value(state_type& state) const
      noexcept(noexcept(std::declval<const Op&>()(lhs_.value(state.lhs),
                                                  rhs_.value(state.rhs))))
          -> eval_type {
    return Op{}(lhs_.value(state.lhs), rhs_.value(state.rhs));
  }

  [[nodiscard]] constexpr auto unchecked_value_at(std::size_t index) const
      noexcept(noexcept(std::declval<const Op&>()(
          lhs_.unchecked_value_at(index), rhs_.unchecked_value_at(index))))
          -> eval_type {
    return Op{}(lhs_.unchecked_value_at(index), rhs_.unchecked_value_at(index));
  }

  constexpr void advance(state_type& state) const noexcept {
    if constexpr (Lhs::has_sequence) {
      lhs_.advance(state.lhs);
    }
    if constexpr (Rhs::has_sequence) {
      rhs_.advance(state.rhs);
    }
  }

private:
  Lhs lhs_;
  Rhs rhs_;
};

template <detail::expression Expr, class Op> class unary_expr {
public:
  using expr_type = Expr;
  using op_type = Op;
  using domain_type = typename Expr::domain_type;
  using eval_type = decltype(std::declval<const Op&>()(
      std::declval<typename Expr::eval_type>()));
  using value_type = detail::output_value_t<eval_type>;
  using mask_type = detail::mask_type_or_t<domain_type>;

  static constexpr bool has_sequence = Expr::has_sequence;
  static constexpr bool has_static_mask = Expr::has_static_mask;
  static constexpr std::size_t static_extent = Expr::static_extent;
  static constexpr mask_type static_mask = [] {
    if constexpr (has_static_mask) {
      return static_cast<mask_type>(Expr::static_mask);
    } else {
      return mask_type{0};
    }
  }();

  struct state_type {
    typename Expr::state_type expr;
  };

  constexpr explicit unary_expr(Expr expr) noexcept(
      std::is_nothrow_move_constructible_v<Expr>)
      : expr_(std::move(expr)) {}

  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return expr_.size();
  }

  [[nodiscard]] constexpr auto mask_value() const noexcept -> mask_type {
    static_assert(has_sequence,
                  "masked::mask_value requires an expression containing "
                  "select()");
    return expr_.mask_value();
  }

  [[nodiscard]] constexpr auto validate() const noexcept -> eval_result {
    return expr_.validate();
  }

  [[nodiscard]] constexpr auto make_state() const noexcept -> state_type {
    return {expr_.make_state()};
  }

  [[nodiscard]] constexpr auto value(state_type& state) const
      noexcept(noexcept(std::declval<const Op&>()(expr_.value(state.expr))))
          -> eval_type {
    return Op{}(expr_.value(state.expr));
  }

  [[nodiscard]] constexpr auto unchecked_value_at(std::size_t index) const
      noexcept(
          noexcept(std::declval<const Op&>()(expr_.unchecked_value_at(index))))
          -> eval_type {
    return Op{}(expr_.unchecked_value_at(index));
  }

  constexpr void advance(state_type& state) const noexcept {
    if constexpr (Expr::has_sequence) {
      expr_.advance(state.expr);
    }
  }

private:
  Expr expr_;
};

template <detail::expression Expr> class restricted_expr {
public:
  using expr_type = Expr;
  using domain_type = typename Expr::domain_type;
  using mask_type = typename Expr::mask_type;
  using eval_type = typename Expr::eval_type;
  using value_type = typename Expr::value_type;
  static constexpr bool has_sequence = true;
  static constexpr bool has_static_mask = false;
  static constexpr std::size_t static_extent = dynamic_extent;

  struct state_type {
    detail::mask_cursor<mask_type> cursor;
    std::size_t index{0};
  };

  constexpr restricted_expr(Expr expr, subset<domain_type> selected) noexcept(
      std::is_nothrow_move_constructible_v<Expr>)
      : expr_(std::move(expr)), selected_(selected) {}

  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return selected_.count();
  }

  [[nodiscard]] constexpr auto mask_value() const noexcept -> mask_type {
    return selected_.raw();
  }

  [[nodiscard]] constexpr auto validate() const noexcept -> eval_result {
    if (!selected_.in_range()) {
      return {eval_status::mask_out_of_range, size(), 0, 0};
    }
    if (const auto status = expr_.validate(); !status) {
      return status;
    }
    if ((expr_.mask_value() & selected_.raw()) != selected_.raw()) {
      return {eval_status::mask_mismatch, size(), 0, 0};
    }
    return {eval_status::ok, size(), 0, 0};
  }

  [[nodiscard]] constexpr auto make_state() const noexcept -> state_type {
    auto state = state_type{detail::mask_cursor<mask_type>(selected_.raw()), 0};
    if (state.cursor.has_next()) {
      state.index = state.cursor.next();
    }
    return state;
  }

  [[nodiscard]] constexpr auto value(state_type& state) const noexcept
      -> eval_type {
    return expr_.unchecked_value_at(state.index);
  }

  [[nodiscard]] constexpr auto
  unchecked_value_at(std::size_t index) const noexcept -> eval_type {
    assert(selected_.contains(index));
    return expr_.unchecked_value_at(index);
  }

  constexpr void advance(state_type& state) const noexcept {
    state.index = state.cursor.next();
  }

private:
  Expr expr_;
  subset<domain_type> selected_;
};

template <detail::expression Expr, typename Expr::mask_type Mask>
class static_restricted_expr {
public:
  using expr_type = Expr;
  using domain_type = typename Expr::domain_type;
  using mask_type = typename Expr::mask_type;
  using eval_type = typename Expr::eval_type;
  using value_type = typename Expr::value_type;
  static constexpr bool has_sequence = true;
  static constexpr bool has_static_mask = true;
  static constexpr mask_type static_mask = Mask;
  static constexpr std::size_t static_extent = detail::popcount(static_mask);

  static_assert(domain_type::valid_mask(static_mask),
                "masked::static_restricted_expr mask is outside its domain");

  struct state_type {
    detail::mask_cursor<mask_type> cursor;
    std::size_t index{0};
  };

  constexpr explicit static_restricted_expr(Expr expr) noexcept(
      std::is_nothrow_move_constructible_v<Expr>)
      : expr_(std::move(expr)) {}

  [[nodiscard]] static constexpr auto size() noexcept -> std::size_t {
    return static_extent;
  }

  [[nodiscard]] static constexpr auto mask_value() noexcept -> mask_type {
    return static_mask;
  }

  [[nodiscard]] constexpr auto validate() const noexcept -> eval_result {
    if (const auto status = expr_.validate(); !status) {
      return status;
    }
    if ((expr_.mask_value() & static_mask) != static_mask) {
      return {eval_status::mask_mismatch, static_extent, 0, 0};
    }
    return {eval_status::ok, static_extent, 0, 0};
  }

  [[nodiscard]] constexpr auto make_state() const noexcept -> state_type {
    auto state = state_type{detail::mask_cursor<mask_type>(static_mask), 0};
    if (state.cursor.has_next()) {
      state.index = state.cursor.next();
    }
    return state;
  }

  [[nodiscard]] constexpr auto value(state_type& state) const noexcept
      -> eval_type {
    return expr_.unchecked_value_at(state.index);
  }

  [[nodiscard]] constexpr auto
  unchecked_value_at(std::size_t index) const noexcept -> eval_type {
    assert(domain_type::valid_index(index));
    assert((static_mask & static_cast<mask_type>(mask_type{1} << index)) !=
           mask_type{0});
    return expr_.unchecked_value_at(index);
  }

  constexpr void advance(state_type& state) const noexcept {
    state.index = state.cursor.next();
  }

private:
  Expr expr_;
};

namespace detail {

template <expression Lhs, expression Rhs, class Op>
struct is_expression<binary_expr<Lhs, Rhs, Op>> : std::true_type {};

template <expression Expr, class Op>
struct is_expression<unary_expr<Expr, Op>> : std::true_type {};

template <expression Expr>
struct is_expression<restricted_expr<Expr>> : std::true_type {};

template <expression Expr, typename Expr::mask_type Mask>
struct is_expression<static_restricted_expr<Expr, Mask>> : std::true_type {};

} // namespace detail

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

namespace detail {

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

} // namespace detail

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
