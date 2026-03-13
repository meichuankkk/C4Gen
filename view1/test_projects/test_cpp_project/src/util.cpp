#include "util.h"

#include <algorithm>
#include "base.h"

namespace demo {

int aggregate(const Base& b, const std::vector<int>& xs, Mode mode) {
    int acc = 0;
    int cur_max = 0;
    for (int x : xs) {
        int v = call_compute(b, x);
        acc += v;
        cur_max = std::max(cur_max, v);
    }
    if (mode == Mode::Max) {
        return cur_max;
    }
    return acc;
}

}  // namespace demo
