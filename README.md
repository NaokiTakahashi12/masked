# masked

`masked` is a small header-only C++20 library for lazy subset operations over contiguous
arrays over a typed finite index domain. A subset is represented by a domain-
specific bitmask, and sequence expressions are combined by original index, not
by rank.

```cpp
#include <masked/masked.hpp>

#include <array>

struct joint_tag {};
using joint_domain = masked::index_domain<joint_tag, 5, std::uint8_t>;

std::array<double, joint_domain::size> position{0.0, 10.0, 20.0, 30.0, 40.0};
std::array<double, joint_domain::size> velocity{1.0, 2.0, 3.0, 4.0, 5.0};
std::array<double, 3> out{};
const auto active = masked::subset<joint_domain>::from_bits_asserted(0b10110);

auto expr = masked::select(position, active) *
            masked::select(velocity, active) +
            2.0;

auto result = masked::checked_eval_to(out, expr);
```

The example evaluates selected indices `{1, 2, 4}` as:

```text
position[1] * velocity[1] + 2
position[2] * velocity[2] + 2
position[4] * velocity[4] + 2
```

## API

- `index_domain<Tag, Size, Mask>` defines a typed finite index space.
- `subset<Domain>` and `subset_c<Domain, Mask>` represent runtime and compile-
  time subsets of that domain.
- `subset<Domain>::from_bits(mask)` returns `std::optional` for checked
  runtime construction.
- `select(values, subset)` creates a non-owning runtime subset view.
- `select<Domain, Mask>(values)` creates a compile-time subset view and
  statically checks fixed-size arrays where possible.
- `+`, `-`, `*`, `/`, and unary `-` build lazy expressions.
- Scalars are broadcast across the selected domain indices.
- `unchecked_eval_to(out, expr)` is the fast unchecked evaluation path.
- `checked_eval_to(out, expr)` detects out-of-range masks, mask mismatches,
  undersized sources, and insufficient output capacity. Its `eval_result`
  reports `selected_size` and, when relevant, `required_source_size`.
- `checked_materialize(expr)` evaluates into an owning `std::vector` and
  returns an error status without evaluating invalid runtime masks.
- `unchecked_materialize(expr)` skips validation when the caller has already
  established the expression preconditions.
- `checked_eval_array(expr)` and `unchecked_eval_array(expr)` evaluate
  compile-time-length expressions into `std::array`.
- `checked_scatter_to(out, expr)` writes selected values back to their original
  indices in a domain-sized output range.
- `checked_add_to`, `checked_subtract_from`, `checked_domain_array`,
  `checked_gather_compact_from`, `checked_scatter_compact_to`, and
  `checked_transform_selected` cover common checked update and gather/scatter
  paths.
- `checked_sum`, `checked_dot`, `checked_min`, `checked_max`,
  `checked_arg_min`, `checked_arg_max`, and `checked_find_if` provide checked
  reductions and searches.
- `empty_selection` reports checked operations that require at least one
  selected value.

Checked and unchecked evaluation into an aliased output range is not guaranteed
to behave correctly. Use `unchecked_materialize` or `checked_materialize` to
make an explicit evaluation boundary in that case.

`select_all<Domain>(values)` requires `values` to cover the full domain.

## Demos

Demos are built by default and are not installed:

```sh
cmake -S . -B build
cmake --build build
./build/bin/masked_demo
```

Disable them with:

```sh
cmake -S . -B build -DMASKED_BUILD_DEMOS=OFF
```

## Integration

Installed package:

```cmake
find_package(masked CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE masked::masked)
```

`FetchContent` or `add_subdirectory`:

```cmake
include(FetchContent)

FetchContent_Declare(
  masked
  SOURCE_DIR /path/to/masked
)
FetchContent_MakeAvailable(masked)

target_link_libraries(my_target PRIVATE masked::masked)
```

When used as a subproject, demos, tests, benchmarks, and install rules are off
by default.
