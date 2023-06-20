//
// Created by clash on 12/06/2023.
//

#ifndef AILEARNING_NETWORK_H
#define AILEARNING_NETWORK_H
#include "Layer.h"

class Network {
public:

    Network() = default;

    virtual void forward(const std::vector<float>& inputVector) = 0;
    virtual void backward(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel) = 0;

    virtual void train(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel) = 0;
    virtual void predict(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel) = 0;

protected:

    std::vector<Layer> m_layers;

};


#endif //AILEARNING_NETWORK_H
