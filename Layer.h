//
// Created by clash on 12/06/2023.
//

#ifndef AILEARNING_LAYER_H
#define AILEARNING_LAYER_H
#include "Math.h"
#include "Weight.h"

class Layer {
public:

    Layer(const std::size_t& layerSize, const std::size_t& weightSize, const float& bias) : m_layerSize(layerSize), m_weightSize(weightSize), m_bias(bias)
    {
        m_weights.resize(layerSize);
        m_outputs.setSize(layerSize);
        m_convertedOutputs.setSize(layerSize);

        for(auto& weight : m_weights)
        {
            weight.setSize(weightSize);
        }
    }

    [[nodiscard]] auto getLayerSize() const noexcept { return m_layerSize; }
    [[nodiscard]] auto getWeightSize() const noexcept { return m_weightSize; }

    [[nodiscard]] auto& getWeights() noexcept { return m_weights; }
    [[nodiscard]] const auto& getWeights() const noexcept { return m_weights; }

    [[nodiscard]] auto& getOutputs() noexcept { return m_outputs; }
    [[nodiscard]] const auto& getOutputs() const noexcept { return m_outputs; }

    [[nodiscard]] auto& getConvertedOutputs() noexcept { return m_convertedOutputs; }
    [[nodiscard]] const auto& getConvertedOutputs() const noexcept { return m_convertedOutputs; }

    [[nodiscard]] const auto& getBias() const noexcept { return m_bias; }

private:

    std::size_t m_layerSize = 0;
    std::size_t m_weightSize = 0;

    // default bias value
    float m_bias = 0.0f;

    Vector<float> m_outputs;
    Vector<float> m_convertedOutputs;

    std::vector<Weight> m_weights;
};


#endif //AILEARNING_LAYER_H
