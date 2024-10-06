//
// Created by clash on 12/06/2023.
//

#include "BatchMLP.h"

void BatchMLP::addLayer(std::size_t layerSize, float bias)
{
    (m_layers.empty()) ? m_layers.emplace_back(layerSize, m_inputSize, bias) :
                         m_layers.emplace_back(layerSize, m_layers[m_layers.size() - 1].getLayerSize(), bias);
}

Vector<float> BatchMLP::getPredictedValues() const noexcept
{
    return m_layers[m_layers.size() - 1].getConvertedOutputs();
}
