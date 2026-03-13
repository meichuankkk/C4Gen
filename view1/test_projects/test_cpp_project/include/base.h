#pragma once

#include <string>

namespace demo {

class Base {
public:
    virtual ~Base() = default;
    virtual std::string name() const;
    virtual int compute(int x) const = 0;
};

int call_compute(const Base& b, int x);

}  // namespace demo
