#pragma once

#include <utility>

namespace demo {

template <class T>
struct Box {
    T value;
};

template <class T, class F>
auto map_box(Box<T> b, F f) -> Box<decltype(f(std::move(b.value)))> {
    using U = decltype(f(std::move(b.value)));
    return Box<U>{f(std::move(b.value))};
}

}  // namespace demo
