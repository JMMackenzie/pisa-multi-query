#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>

namespace pisa {

using term_scorer_t = std::function<float(uint32_t, uint32_t)>;

template <typename Wand>
struct index_scorer {
   protected:
    const Wand &m_wdata;
   public:
    explicit index_scorer(const Wand &wdata) : m_wdata(wdata) {}
    virtual ~index_scorer() = default;

    virtual term_scorer_t term_scorer(uint64_t term_id) const = 0;
};

} // namespace pisa
