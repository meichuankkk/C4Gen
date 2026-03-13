#include "base.h"

namespace demo {

std::string Base::name() const {
    return "Base";
}

int call_compute(const Base& b, int x) {
    return b.compute(x);
}

}  // namespace demo
