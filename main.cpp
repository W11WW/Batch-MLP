#include "Math.h"

#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

#include "stb_image.h"
#include "BatchMLP.h"

const int epochs = 50;

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
void createDataSet(std::vector<std::pair<std::vector<float>, std::vector<float>>>& datasets, const std::string filePath)
{
    // Create a text string, which is used to output the text file
    std::string myText;

    // Read from the text file
    std::ifstream MyReadFile(filePath);

    if(!MyReadFile)
    {
        throw std::system_error(errno, std::system_category(), "failed to open" + filePath);
    }

    std::vector<std::pair<std::vector<float>, std::vector<float>>> totalDataset;
    // Use a while loop together with the getline() function to read the file line by line
    while (getline (MyReadFile, myText)) {
        // Output the text from the file
        std::vector<std::string> splitString {};

        customSplit(myText, ',', splitString);

        std::string label = splitString[0];

        std::pair<std::vector<float>, std::vector<float>> dataPoint;

        for(int i = 1; i < splitString.size(); i++)
        {
            dataPoint.first.push_back(((static_cast<float>(std::stoi(splitString[i])) - 0.0f) / 255.0f - 0.0f));
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

        totalDataset.push_back(dataPoint);
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

int main()
{
    std::function sigmoidFunction = [](const Vector<float>& outputs) -> Vector<float> {

        Vector<float> converted;
        converted.setSize(outputs.getSize());

        for(int i = 0; i < converted.getSize(); i++)
        {
            converted[i] = 1.0f / (1.0f + std::exp(-outputs[i]));
        }

        return converted;
    };

    std::function sigmoidFirstDerivative = [](const Vector<float> outputs) -> Vector<float> {

        Vector<float> converted;
        converted.setSize(outputs.getSize());

        for(int i = 0; i < converted.getSize(); i++)
        {
            converted[i] = outputs[i] * (1.0f - outputs[i]);
        }

        return converted;
    };

    std::function errorFirstDerivative = [](const Vector<float> outputs, const Vector<float> target) -> Vector<float> {

        Vector<float> converted;
        converted.setSize(outputs.getSize());
        for(int i = 0; i < converted.getSize(); i++)
        {
            converted[i] = (2 * (target[i] - outputs[i])) / (float) outputs.getSize();
        }

        return converted;
    };

    BatchMLP mlp {};

    mlp.setInputSize(784);
    mlp.setLearningRate(0.01f);
    mlp.addLayer(16, 0.0f);
    //mlp.addLayer(16, 0.0f);
    //mlp.addLayer(10, 0.0f);


    std::vector<std::pair<std::vector<float>, std::vector<float>>> trainingSet {};
    createDataSet(trainingSet, "/Users/michaelferents/Downloads/archive/mnist_train.csv");

    std::vector<std::pair<std::vector<float>, std::vector<float>>> validationSet {};
    createDataSet(validationSet, "/Users/michaelferents/Downloads/archive/mnist_test.csv");

    for(int i = 0; i < epochs; i++)
    {
        std::shuffle(trainingSet.begin(), trainingSet.end(), std::mt19937(std::random_device()()));
        int numbTrained = 0;
        for(auto & j : trainingSet)
        {
            numbTrained++;
            mlp.forward(j.first, sigmoidFunction);
            mlp.backward(j.first, j.second, errorFirstDerivative, sigmoidFirstDerivative);
            if(numbTrained == mlp.getBatchSize())
            {
                numbTrained = 0;
                mlp.gradientDescent();
            }
        }
        int numberRight = 0;
        for(auto & j : validationSet)
        {
            mlp.forward(j.first, sigmoidFunction);
            int indexForPredictedValue = largestValue(mlp.getPredictedValues().getValues());
            std::cout << mlp.getPredictedValues() << std::endl;
            for(int k = 0; k < j.second.size(); k++)
            {
                if(j.second[k] == 1 && (k == indexForPredictedValue)) numberRight++;
            }
        }
        int numberWrong = validationSet.size() - numberRight;
        std::cout << numberWrong << std::endl;
        float percentage = ( (float)(validationSet.size() - numberWrong) / (float)validationSet.size()) * 100;
        std::cout << "Epoch " << i + 1 << " completed with a validation score of " << percentage << "%" << std::endl;
    }

    /*
    int numberWrong = 0;
    for(int j = 0; j < testSet.size(); j++)
    {
        if (!network.predict(trainingSet[j].first, trainingSet[j].second)) numberWrong++;
    }
    std::cout << "Test set completed with a score of " << ((5000 - numberWrong) / 5000) * 100 << "%" << std::endl;
     */

    return 0;
}
