#pragma once

#include <vector>

namespace demo {

class Base;

enum class Mode {
    Sum = 0,
    Max = 1,
};

int aggregate(const Base& b, const std::vector<int>& xs, Mode mode);

}  // namespace demo
