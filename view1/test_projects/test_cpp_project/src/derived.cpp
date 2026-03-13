#include "derived.h"

namespace demo {

Derived::Derived(int bias) : bias_(bias) {}

std::string Derived::name() const {
    return "Derived";
}

int Derived::compute(int x) const {
    return x + bias_;
}

}  // namespace demo
