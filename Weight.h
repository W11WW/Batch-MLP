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
        /*
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator

        auto range = static_cast<float>(std::sqrt(1.0f / (float) previousLayerSize));
        std::normal_distribution<float> distr(0, 1); // define the range

        // Perhaps give this a separate function
         */
        for(float& m_value : m_values)
        {
            m_value = 0.2f;
        }
    }

    auto& getValues() noexcept { return m_values; }
    auto& getGradientValues() noexcept { return m_gradients; }

private:

    Wuu::Vector<float> m_values;
    Wuu::Vector<float> m_gradients;

};


#endif //AILEARNING_WEIGHT_H
