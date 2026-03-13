#pragma once

#include "base.h"

namespace demo {

class Derived : public Base {
public:
    explicit Derived(int bias);
    std::string name() const override;
    int compute(int x) const override;

private:
    int bias_;
};

}  // namespace demo
