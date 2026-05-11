# masked

`masked` is a small C++20 library for lazy subset operations over contiguous
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
const auto active = masked::subset<joint_domain>::from_bits(0b10110);

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
- `select(values, subset)` creates a non-owning runtime subset view.
- `select<Domain, Mask>(values)` creates a compile-time subset view and
  statically checks fixed-size arrays where possible.
- `+`, `-`, `*`, `/`, and unary `-` build lazy expressions.
- Scalars are broadcast across the selected domain indices.
- `unchecked_eval_to(out, expr)` is the fast unchecked evaluation path.
- `checked_eval_to(out, expr)` detects out-of-range masks, mask mismatches,
  undersized sources, and insufficient output capacity. Its `eval_result`
  reports `selected_size` and, when relevant, `required_source_size`.
- `materialize(expr)` evaluates into an owning `std::vector`.
- `checked_materialize(expr)` evaluates into an owning `std::vector` and
  returns an error status without evaluating invalid runtime masks.
- `unchecked_materialize(expr)` skips validation when the caller has already
  established the expression preconditions.
- `eval_array(expr)` evaluates compile-time-length expressions into
  `std::array`.
- `checked_scatter_to(out, expr)` writes selected values back to their original
  indices in a domain-sized output range.
- `unchecked_value_at(index)` is the explicit by-index unchecked access primitive
  used by the evaluator internals.
- `sparse_evaluator` is the default evaluator used by `eval_to`; `dense_evaluator`
  and `full_mask_evaluator` are reserved extension points for future strategies.

`eval_to` does not guarantee correct behavior when the output aliases any input
array used by the expression. Use `materialize` to make an explicit evaluation
boundary in that case.

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
