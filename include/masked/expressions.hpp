#pragma once

#include <masked/detail/expression_meta.hpp>

namespace masked {

template <class T, class Domain> class subset_view {
public:
  static_assert(detail::domain<Domain>,
                "masked::subset_view requires a masked index domain");

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

template <class T, class Domain, typename Domain::mask_type Mask>
class static_subset_view {
public:
  static_assert(detail::domain<Domain>,
                "masked::static_subset_view requires a masked index domain");

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

template <class Lhs, class Rhs, class Op> class binary_expr {
public:
  static_assert(detail::expression<Lhs> && detail::expression<Rhs>,
                "masked::binary_expr operands must be masked expressions");

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

template <class Expr, class Op> class unary_expr {
public:
  static_assert(detail::expression<Expr>,
                "masked::unary_expr operand must be a masked expression");

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

template <class Expr> class restricted_expr {
public:
  static_assert(detail::expression<Expr>,
                "masked::restricted_expr requires a masked expression");

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

template <class Expr, typename Expr::mask_type Mask>
class static_restricted_expr {
public:
  static_assert(detail::expression<Expr>,
                "masked::static_restricted_expr requires a masked expression");

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

} // namespace masked
