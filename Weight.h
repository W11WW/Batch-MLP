//
// Created by clash on 12/06/2023.
//

#ifndef AILEARNING_WEIGHT_H
#define AILEARNING_WEIGHT_H

#include <random>
#include <cmath>
#include <algorithm>
#include "Math.h"

class Weight {
public:

    Weight() = default;

    void setSize(const std::size_t size)
    {
        m_values.setSize(size);
        m_gradients.setSize(size);

        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator

        auto range = static_cast<float>(std::sqrt(1.0f / (float) size));
        std::uniform_real_distribution distr(-(range), (range)); // define the range

        // Perhaps give this a separate function

        for(float& m_value : m_values)
        {
            m_value = distr(gen);
        }
    }

    auto& getValues() noexcept { return m_values; }
    auto& getGradientValues() noexcept { return m_gradients; }

private:

    Vector<float> m_values;
    Vector<float> m_gradients;

};


#endif //AILEARNING_WEIGHT_H
