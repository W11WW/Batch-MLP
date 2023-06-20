//
// Created by clash on 12/06/2023.
//

#ifndef AILEARNING_BATCHMLP_H
#define AILEARNING_BATCHMLP_H
#include "Network.h"

class BatchMLP : Network {
public:

    BatchMLP();

    void forward(const std::vector<float>& inputVector) final;
    void backward(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel) final;

    void train(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel) final;
    void predict(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel) final;
    void gradientDescent();

    void addLayer(std::size_t layerSize,float bias);

    void setBatchSize(std::size_t batchSize) noexcept { m_batchSize = batchSize; }
    [[nodiscard]] auto getBatchSize() const noexcept { return m_batchSize; }

    void setLearningRate(const float learningRate) noexcept { m_learningRate = learningRate; }
    [[nodiscard]] auto getLearningRate() const noexcept { return m_learningRate; }

    void setInputSize(std::size_t inputSize) noexcept { m_inputSize = inputSize; }
    [[nodiscard]] auto getInputSize() const noexcept { return m_inputSize; }

private:

    // default value
    std::size_t m_batchSize = 100;
    // default value
    float m_learningRate = 0.5;
    // default value
    std::size_t m_inputSize = 0;
    // activation functions with sigmoid as the default
    std::function<void(Layer&)> m_hiddenActivation;
    std::function<void(Layer&)> m_outputActivation;
    // activation functions with sigmoid as the default
    std::function<std::vector<float>(Layer&)> m_hiddenActivationDerivative;
    std::function<std::vector<float>(Layer&)> m_outputActivationDerivative;

};


#endif //AILEARNING_BATCHMLP_H
