#include "Math.h"

#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

#include "stb_image.h"
#include "BatchMLP.h"
/*
 *
const int mnistImageSizeX = 28;
const int mnistImageSizeY = 28;

const int inputSize = mnistImageSizeX * mnistImageSizeY; // = 784, without bias neuron

const int epochs = 50;
const float learning_rate = 0.5f;

const float bias = 0.0f;

const int batchSize = 100;

// this sigmoid function is h(a) = tanh(a) = e^a-e^-a / e^a+e^-a
float sigmoidFunction(float numb)
{
    //return (1 / (1 + std::exp(-numb)));
    return 1.0f / (1.0f + std::exp(-numb));
}

float firstDerivativeSigmoidFunction(float numb)
{
    return (sigmoidFunction(numb) * (1.0f - sigmoidFunction(numb)) );
}

// Mean squared error function
float firstDerivativeErrorFunction(float trainingLabel, float probability)
{
    return (trainingLabel - probability);
}
// Create custom split() function.
void customSplit(std::string str, char separator, std::vector<std::string>& strings) {
    int startIndex = 0, endIndex = 0;
    for (int i = 0; i <= str.size(); i++) {

        // If we reached the end of the word or the end of the input.
        if (str[i] == separator || i == str.size()) {
            endIndex = i;
            std::string temp;
            temp.append(str, startIndex, endIndex - startIndex);
            strings.push_back(temp);
            startIndex = endIndex + 1;
        }
    }
}

// 28 width 28 height and 1 channel
void createDataSets(std::vector<std::pair<std::vector<float>, std::vector<float>>>& datasets)
{
    // Create a text string, which is used to output the text file
    std::string myText;

    // Read from the text file
    std::ifstream MyReadFile("D:/ImageDataset/numbers - Only Mnist.csv");

    if(!MyReadFile)
    {
        throw std::system_error(errno, std::system_category(), "failed to open D:/ImageDataset/numbers - Only Mnist.csv");
    }

    std::vector<std::pair<std::vector<float>, std::vector<float>>> totalDataset;

    int numb = 0;
    int numb2 = 0;
    // Use a while loop together with the getline() function to read the file line by line
    while (getline (MyReadFile, myText)) {
        // Output the text from the file
        std::vector<std::string> splitString {};

        customSplit(myText, ',', splitString);

        std::string label = splitString[2];
        std::string filePath = "D:/ImageDataset/numbers/" + splitString[3];

        std::pair<std::vector<float>, std::vector<float>> dataPoint;

        int width, height, channels;
        unsigned char *img = stbi_load(filePath.c_str(), &width, &height, &channels, 1);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                dataPoint.first.push_back((((float) img[x + width * y] - 0.0f) / 255.0f - 0.0f));
            }
        }
        switch(std::stoi(label))
        {
            case 0:
                dataPoint.second = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                break;
            case 1:
                dataPoint.second = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
                break;
            case 2:
                dataPoint.second = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
                break;
            case 3:
                dataPoint.second = { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
                break;
            case 4:
                dataPoint.second = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
                break;
            case 5:
                dataPoint.second = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
                break;
            case 6:
                dataPoint.second = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
                break;
            case 7:
                dataPoint.second = { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
                break;
            case 8:
                dataPoint.second = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
                break;
            case 9:
                dataPoint.second = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
                break;
        };

        stbi_image_free(img);

        totalDataset.push_back(dataPoint);
        numb++;

        if(numb == 10000)
        {
            numb2++;
            std::cout << "10000 images have been scanned" << std::endl;
            std::cout << numb2 << "/6" << std::endl;
            numb = 0;
        }
    }
    // Close the file
    MyReadFile.close();

    std::shuffle(totalDataset.begin(), totalDataset.end(), std::mt19937(std::random_device()()));

    std::cout << "Completed reading all images with a total size of: " << totalDataset.size() << std::endl;

    datasets = totalDataset;
}

int largestValue(std::vector<float> arr)
{
    int i;

    // Initialize maximum element
    float max = arr[0];
    int indexNumber = 0;

    // Traverse array elements
    // from second and compare
    // every element with current max
    for (i = 1; i < arr.size(); i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
            indexNumber = i;
        }
    }

    return indexNumber;
}

// Currently using gradient descent
class Network {
public:

    void setTopology(const std::vector<int>& topology, const std::size_t& inputSize)
    {
        // Save topology
        m_topology = topology;

        // First layer given size of input vector
        Layer initialLayer {topology[0], inputSize, inputSize };
        m_layers.push_back(initialLayer);

        if(topology.size() > 1)
            for(int i = 1; i < topology.size(); i++)
            {
                Layer layer {topology[i], topology[i - 1],  m_layers[i - 1].getLayerSize()};

                m_layers.push_back(layer);
            }

        // Create all output and converted output vectors for corresponding layer
        for(const auto& layer : m_layers)
        {
            Outputs outputs {};
            outputs.setSize(layer.getLayerSize());

            m_outputs.push_back(outputs);
            m_convertedOutputs.push_back(outputs);
        }

    }

    void printTopology() const noexcept
    {

        for(int i = 0; i < m_layers.size(); i++)
        {
            std::cout << "Layer " << i + 1 << " has " << m_layers[i].getLayerSize()
            << " weights with each weight having a size " << m_layers[i].getWeights()[0].getWeightSize() << std::endl;
        }
    }

    void forwardPropagate(const std::vector<float>& inputVector)
    {
        for(int k = 0; k < m_layers.size(); k++)
        {
            for(int i = 0; i < m_layers[k].getLayerSize(); i++)
            {
                // Check to see if it is the first layer to apply the input vector instead of output vector
                // But it checks every time which can be costly see if u can find a better way
                    for(int j = 0; j < m_layers[k].getWeights()[0].getWeightSize(); j++)
                    {
                        if(k == 0)
                        {
                            m_outputs[k].getValues()[i] += m_layers[k].getWeights()[i].getValues()[j] * inputVector[j];
                        } else
                        {
                            m_outputs[k].getValues()[i] += m_layers[k].getWeights()[i].getValues()[j] * m_convertedOutputs[k - 1].getValues()[j];
                        }
                    }
                // Add bias
                m_outputs[k].getValues()[i] += bias;

                // Convert outputs with an activation function
                m_convertedOutputs[k].getValues()[i] = sigmoidFunction(m_outputs[k].getValues()[i]);
            }
        }
    }

    // error for last layer
    float errorLL(const int lastLayer, const int index, const std::vector<float>& trainingLabel)
    {
        return firstDerivativeErrorFunction(trainingLabel[index], m_convertedOutputs[lastLayer].getValues()[index]) *
        firstDerivativeSigmoidFunction(m_outputs[lastLayer].getValues()[index]);
    }
    // error for hidden layer
    float errorAL(const int lastLayer, const int index, const int currentLayer, const int outputSize,const std::vector<float>& trainingLabel,
                  const std::vector<float>& errorLLVector, std::vector<Outputs>& sigmoidDerivatives)
    {
        if(currentLayer == lastLayer)
        {
            std::cout << std::endl;
            std::cout << index << std::endl;
            std::cout << currentLayer << std::endl;
            std::cout << std::endl;
            return errorLLVector[index];
        }
        float sum = 0.0f;
        for(int i = 0; i < m_layers[currentLayer + 1].getLayerSize(); i++)
        {
            sum += errorAL(lastLayer, i, currentLayer + 1, outputSize, trainingLabel, errorLLVector, sigmoidDerivatives) * m_layers[currentLayer].getWeights()[i].getValues()[index];
        }
        return sigmoidDerivatives[currentLayer].getValues()[index] * sum;
    }

    void backwardPropagate(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel)
    {
        const int lastLayer = static_cast<int>(m_layers.size()) - 1;

        std::vector<float> firstError {};
        std::vector<Outputs> sigmoidDerivatives {};
        sigmoidDerivatives.resize(m_layers.size());
        // To save on computation we first compute all the sigmoid derivatives
        for(int i = 0; i < m_layers.size(); i++)
        {
            for(int k = 0; k < m_layers[i].getLayerSize(); k++)
            {
                sigmoidDerivatives[i].getValues().push_back(firstDerivativeSigmoidFunction(m_outputs[i].getValues()[k]));
            }
        }
        if(lastLayer != 0)
        {
            // Calculate gradient for weights in last Layer
            for(int j = 0; j < m_layers[lastLayer].getLayerSize(); j++)
            {
                for(int k = 0; k < m_layers[lastLayer].getWeights()[0].getWeightSize(); k++)
                {
                    m_layers[lastLayer].getWeights()[j].getGradientValues()[k] += errorLL(lastLayer, j, trainingLabel) * m_convertedOutputs[lastLayer - 1].getValues()[k];
                }
                firstError.push_back(errorLL(lastLayer, j, trainingLabel));
            }
        } else
        {
            // Calculate gradient for weights in last Layer
            for(int j = 0; j < m_layers[lastLayer].getLayerSize(); j++)
            {
                for(int k = 0; k < m_layers[lastLayer].getWeights()[0].getWeightSize(); k++)
                {
                    m_layers[lastLayer].getWeights()[j].getGradientValues()[k] += errorLL(lastLayer, j, trainingLabel) * inputVector[k];
                }
                firstError.push_back(errorLL(lastLayer, j, trainingLabel));
            }
        }
        if(lastLayer != 0)
        {
            // Calculate gradient for weights in hidden layers
            for(int l = 0; l < lastLayer; l++)
            {
                for(int j = 0; j < m_layers[l].getLayerSize(); j++)
                {
                    for(int k = 0; k < m_layers[l].getWeights()[0].getWeightSize(); k++)
                    {
                        (l == 0) ? m_layers[l].getWeights()[j].getGradientValues()[k] += errorAL(lastLayer, j, l + 1, m_layers[lastLayer].getLayerSize(), trainingLabel, firstError, sigmoidDerivatives) * inputVector[k]
                                : m_layers[l].getWeights()[j].getGradientValues()[k] += errorAL(lastLayer, j, l + 1, m_layers[lastLayer].getLayerSize(), trainingLabel, firstError, sigmoidDerivatives) * m_convertedOutputs[l].getValues()[k];
                    }
                }
            }
        }
    }

    void sequential_gradient_descent()
    {
        for(int i = 0; i < m_layers.size(); i++)
        {
            for(int j = 0; j < m_layers[i].getLayerSize(); j++)
            {
                for(int k = 0; k < m_layers[i].getWeights()[0].getWeightSize(); k++)
                {

                    m_layers[i].getWeights()[j].getValues()[k] += ((-learning_rate) * (m_layers[i].getWeights()[j].getGradientValues()[k]) / batchSize);
                    m_layers[i].getWeights()[j].getGradientValues()[k] = 0.0f;
                }
            }
        }
    }

    void train(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel, const int batchIndex)
    {
        forwardPropagate(inputVector);
        backwardPropagate(inputVector, trainingLabel);
        if(batchIndex == batchSize)
        {
            sequential_gradient_descent();
        }
    }

    bool predict(const std::vector<float>& inputVector, const std::vector<float>& trainingLabel)
    {
        forwardPropagate(inputVector);
        int indexLabel = 0;

        int indexPredict = largestValue(m_convertedOutputs[m_convertedOutputs.size() - 1].getValues());

        for(int i = 0; i < trainingLabel.size(); i++)
        {
            if(trainingLabel[i] == 1) { indexLabel = i; }
        }

        if(indexLabel == indexPredict){ return true; }
        return false;
    }

private:

    std::vector<int> m_topology;

    std::vector<Layer> m_layers;

    std::vector<Outputs> m_outputs;
    std::vector<Outputs> m_convertedOutputs;


};

 */

int main() {

    /*
    std::vector<int> topology { 10 };

    Network network {};
    network.setTopology(topology, inputSize);
    network.printTopology();

    std::vector<std::pair<std::vector<float>, std::vector<float>>> dataset;
    createDataSets(dataset);

    std::vector<std::pair<std::vector<float>, std::vector<float>>> trainingSet {};
    std::vector<std::pair<std::vector<float>, std::vector<float>>> validationSet {};
    std::vector<std::pair<std::vector<float>, std::vector<float>>> testSet {};

    for(int i = 0; i < 10000; i++)
    {
        trainingSet.push_back(dataset[i]);
    }

    for(int i = 9999; i < 11000; i++)
    {
        validationSet.push_back(dataset[i]);
    }

    for(int i = 54999; i < 59999; i++)
    {
        trainingSet.push_back(dataset[i]);
    }

    for(int i = 0; i < epochs; i++)
    {
        std::shuffle(trainingSet.begin(), trainingSet.end(), std::mt19937(std::random_device()()));
        int numbTrained = 0;
        for(int j = 0; j < trainingSet.size(); j++)
        {
            numbTrained++;
            network.train(trainingSet[j].first, trainingSet[j].second, numbTrained);
            if(numbTrained == batchSize) numbTrained = 0;
        }
        int numberWrong = 0;
        for(int j = 0; j < validationSet.size(); j++)
        {
            if (!network.predict(validationSet[j].first, validationSet[j].second)) numberWrong++;
        }
        std::cout << numberWrong << std::endl;
        float percentage = ( (float)(1000 - numberWrong) / 1000) * 100;
        std::cout << "Epoch " << i + 1 << " completed with a validation score of " << percentage << "%" << std::endl;
    }

    int numberWrong = 0;
    for(int j = 0; j < testSet.size(); j++)
    {
        if (!network.predict(trainingSet[j].first, trainingSet[j].second)) numberWrong++;
    }
    std::cout << "Test set completed with a score of " << ((5000 - numberWrong) / 5000) * 100 << "%" << std::endl;
     */

    Vector<float> one  = { 1.0f, 1.0f, 1.0f};
    Vector<float> two = { 1.0f, 1.0f, 1.0f};

    add(one, two);

    Vector<float> three = addR(one, two);

    std::cout << three << std::endl;

    BatchMLP mlp {};
    mlp.setInputSize(4);
    mlp.addLayer(4, 0.0f);
    mlp.addLayer(3, 0.0f);
    mlp.addLayer(2, 0.0f);

    mlp.forward({ 0.2f, 0.2f, 0.2f, 0.2f});

    return 0;
}
