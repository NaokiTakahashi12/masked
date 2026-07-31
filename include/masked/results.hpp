#pragma once

#include <array>
#include <vector>

#include <masked/fwd.hpp>

namespace masked {

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

} // namespace masked
