#include <iostream>
#include <vector>

#include "config.h"
#include "derived.h"
#include "templ.h"
#include "util.h"

namespace demo {

struct Runner {
    int run() const {
        Derived d(DEMO_FACTOR);
        std::vector<int> xs{1, 2, 3};
        int a = aggregate(d, xs, Mode::Sum);
        int b = aggregate(d, xs, Mode::Max);

        Box<int> bi{a + b};
        auto bj = map_box(bi, [](int v) { return v + 1; });

        return bj.value;
    }
};

}  // namespace demo

int main() {
    demo::Runner r;
    std::cout << r.run() << "\n";
    return 0;
}
