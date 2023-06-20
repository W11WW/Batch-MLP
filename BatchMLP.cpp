//
// Created by clash on 12/06/2023.
//

#include "BatchMLP.h"

BatchMLP::BatchMLP()
{
    // default activation for both as mentioned in the header is the sigmoid function
    auto defaultActivation = [](Layer& layer) {

        for(auto& pair : layer.getWeights())
        {
            pair.second.getConvertedOutput() = 1.0f / (1.0f + std::exp(-pair.second.getOutput()));
        }
    };

    m_hiddenActivation = defaultActivation;
    m_outputActivation = defaultActivation;

    // default activation derivative for both as mentioned in the header is the sigmoid function
    auto defaultActivationDerivative = [](Layer& layer) -> std::vector<float> {
        std::vector<float> values;
        for(auto& pair : layer.getWeights())
        {
            values.push_back(pair.second.getConvertedOutput() * (1.0f - pair.second.getConvertedOutput()));
        }
        return values;
    };

    m_hiddenActivationDerivative = defaultActivationDerivative;
    m_outputActivationDerivative = defaultActivationDerivative;
}

void BatchMLP::forward(const std::vector<float> &inputVector)
{
    // eventually replace gathered outputs and implement it properly
    Vector<float> input { inputVector };

    for(auto& layer : m_layers)
    {
        for(auto& pair : layer.getWeights())
        {
            pair.second.getOutput() = dot(input, pair.first.getValues()) + layer.getBias();
        }
        m_hiddenActivation(layer);
        input = layer.retrieveConvertedOutputs();
    }
}

void BatchMLP::backward(const std::vector<float> &inputVector, const std::vector<float> &trainingLabel)
{
    // Calculate last layer gradients
    /*
    const std::size_t lastLayerIndex = m_layers.size() - 1;
    for(auto& pair : m_layers[lastLayerIndex].getWeights())
    {
        pair.first.getGradientValues() = m_layers
    }
     */
}

void BatchMLP::gradientDescent() {

}

void BatchMLP::train(const std::vector<float> &inputVector, const std::vector<float> &trainingLabel)
{

}

void BatchMLP::predict(const std::vector<float> &inputVector, const std::vector<float> &trainingLabel)
{

}

void BatchMLP::addLayer(std::size_t layerSize, float bias)
{
    (m_layers.empty()) ? m_layers.emplace_back(layerSize, m_inputSize, bias) : m_layers.emplace_back(layerSize, m_layers[m_layers.size() - 1].getLayerSize(), bias);
}
