//
// Created by clash on 12/06/2023.
//

#ifndef AILEARNING_BATCHMLP_H
#define AILEARNING_BATCHMLP_H
#include "Layer.h"

class BatchMLP {
public:

    BatchMLP() = default;

    template<typename ActivationFunction>
    void forward(const Vector<float>& inputVector, const ActivationFunction& activation)
    {
        for(int k = 0; k < m_layers.size(); k++)
        {
            for(int i = 0 ; i < m_layers[k].getLayerSize(); i++)
            {
                (k == 0) ? m_layers[k].getOutputs()[i] = dot(inputVector, m_layers[k].getWeights()[i].getValues()) :
                           m_layers[k].getOutputs()[i] = dot(m_layers[k - 1].getConvertedOutputs(), m_layers[k].getWeights()[i].getValues());
            }
            m_layers[k].getConvertedOutputs() = activation(m_layers[k].getOutputs());
        }
    }
    // FDActivation, FDError = first derivative of that function
    template<typename FDError, typename FDActivation>
    void backward(const Vector<float>& inputVector, const Vector<float>& trainingLabel,
                  const FDError& fdError, const FDActivation& fdActivation)
    {
        const int LastLayer = m_layers.size() - 1;

        auto firstDerivError = fdError(m_layers[LastLayer].getConvertedOutputs(), trainingLabel);
        auto firstDerivActivation = fdActivation(m_layers[LastLayer].getOutputs());

        Vector<float> errors {};
        errors.setSize(m_layers[LastLayer].getLayerSize());

        for(int i = 0; i < errors.getSize(); i++)
        {
            errors[i] = firstDerivError[i] * firstDerivActivation[i];
        }

        for(int i = 0; i < m_layers[LastLayer].getLayerSize(); i++)
        {
            for(int j  = 0; j < m_layers[LastLayer].getWeightSize(); j++)
            {
                (LastLayer == 0) ? m_layers[LastLayer].getWeights()[i].getGradientValues()[j] += errors[i] * inputVector[j] :
                                   m_layers[LastLayer].getWeights()[i].getGradientValues()[j] += errors[i] * m_layers[LastLayer - 1].getConvertedOutputs()[j];
            }
        }
        if(LastLayer > 0)
        {
            for(int i = LastLayer - 1; i > -1; i--)
            {
                float sum = 0.0f;
                for(int j = LastLayer; j < i; j-- )
                {
                    auto fda = fdActivation(m_layers[j - 1].getOutputs());

                    for(int k = 0; k < m_layers[j].getLayerSize(); k++)
                    {
                        sum += dot(m_layers[j].getWeights()[k].getValues(), fda);
                    }
                }

                auto fda = fdActivation(m_layers[i].getOutputs());

                for(int j = 0; j < m_layers[i].getLayerSize(); j++)
                {
                    for(int k = 0; k < m_layers[i].getWeightSize(); k++)
                    {
                        (i == 0) ? m_layers[i].getWeights()[j].getGradientValues()[k] += errors[j] * sum * fda[j] * inputVector[k] :
                                m_layers[i].getWeights()[j].getGradientValues()[k] += errors[j] * sum * fda[j] * m_layers[i - 1].getConvertedOutputs()[k];
                    }
                }
            }
        }
    }

    void gradientDescent()
    {
        for(auto & m_layer : m_layers)
        {
            for(int j = 0; j < m_layer.getLayerSize(); j++)
            {
                for(int k = 0; k < m_layer.getWeightSize(); k++)
                {

                    m_layer.getWeights()[j].getValues()[k] += ((-m_learningRate) * (m_layer.getWeights()[j].getGradientValues()[k]) / (float)m_batchSize);
                    m_layer.getWeights()[j].getGradientValues()[k] = 0.0f;
                }
            }
        }
    }

    void addLayer(std::size_t layerSize, float bias);

    void setBatchSize(std::size_t batchSize) noexcept { m_batchSize = batchSize; }
    [[nodiscard]] auto getBatchSize() const noexcept { return m_batchSize; }

    void setLearningRate(const float learningRate) noexcept { m_learningRate = learningRate; }
    [[nodiscard]] auto getLearningRate() const noexcept { return m_learningRate; }

    void setInputSize(std::size_t inputSize) noexcept { m_inputSize = inputSize; }
    [[nodiscard]] auto getInputSize() const noexcept { return m_inputSize; }

    [[nodiscard]] Vector<float> getPredictedValues() const noexcept;

private:

    // default value
    std::size_t m_batchSize = 50;
    // default value
    float m_learningRate = 0.5;
    // default value
    std::size_t m_inputSize = 0;

    std::vector<Layer> m_layers;
};
#endif //AILEARNING_BATCHMLP_H
