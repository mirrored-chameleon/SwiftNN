//
//  BasicModel.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 22/08/2026.
//

import Foundation

public struct BasicModel<TalentType: Talent>: Network {
    public var talent: TalentType
    public var weights: [[Matrix<Double>]]
    public var bias: [Matrix<Double>]

    var layerOutputs: [[Double]] = []

    var learningRate: Double = 0.001

    // MARK: - Prediction

    public mutating func predict(input inputMatrix: Matrix<Double>) -> Matrix<Double> {
        var currentLayerInputs = flatten(inputMatrix)

        layerOutputs.removeAll()

        for layerIndex in 0 ..< weights.count {
            let isFinalLayer = layerIndex == weights.count - 1

            var currentLayerOutputs: [Double] = []

            for neuronIndex in 0 ..< weights[layerIndex].count {
                var weightedSum = 0.0

                for inputIndex in 0 ..< currentLayerInputs.count {
                    let weightValue =
                        weights[layerIndex][neuronIndex][0, inputIndex]

                    let inputValue =
                        currentLayerInputs[inputIndex]

                    weightedSum += weightValue * inputValue
                }

                weightedSum += bias[layerIndex][neuronIndex, 0]

                if !isFinalLayer {
                    weightedSum = relu(weightedSum)
                }

                currentLayerOutputs.append(weightedSum)
            }

            if isFinalLayer {
                currentLayerOutputs = softmax(currentLayerOutputs)
            }

            layerOutputs.append(currentLayerOutputs)

            currentLayerInputs = currentLayerOutputs
        }

        return Matrix(
            rows: currentLayerInputs.count,
            columns: 1,
            grid: currentLayerInputs,
        )
    }

    // MARK: - Training

    public mutating func train(
        input inputMatrix: Matrix<Double>,
        target targetMatrix: Matrix<Double>,
    ) {
        let predictedMatrix = predict(input: inputMatrix)

        let predictedValues = flatten(predictedMatrix)
        let targetValues = flatten(targetMatrix)

        var layerErrors: [[Double]] =
            Array(repeating: [], count: weights.count)

        let outputLayerIndex = weights.count - 1

        var outputLayerErrors: [Double] = []

        for valueIndex in 0 ..< predictedValues.count {
            let prediction = predictedValues[valueIndex]
            let target = targetValues[valueIndex]

            outputLayerErrors.append(prediction - target)
        }

        layerErrors[outputLayerIndex] = outputLayerErrors

        if weights.count > 1 {
            for layerIndex in stride(
                from: weights.count - 2,
                through: 0,
                by: -1,
            ) {
                let neuronCount = weights[layerIndex].count

                var currentLayerErrors =
                    Array(repeating: 0.0, count: neuronCount)

                for neuronIndex in 0 ..< neuronCount {
                    var propagatedError = 0.0

                    for nextNeuronIndex in 0 ..< weights[layerIndex + 1].count {
                        let weightToNextNeuron =
                            weights[layerIndex + 1][nextNeuronIndex][0, neuronIndex]

                        let nextLayerError =
                            layerErrors[layerIndex + 1][nextNeuronIndex]

                        propagatedError +=
                            weightToNextNeuron * nextLayerError
                    }

                    let neuronOutput =
                        layerOutputs[layerIndex][neuronIndex]

                    currentLayerErrors[neuronIndex] =
                        propagatedError * reluDerivative(neuronOutput)
                }

                layerErrors[layerIndex] = currentLayerErrors
            }
        }

        // MARK: - Update Weights

        for layerIndex in 0 ..< weights.count {
            let inputsToLayer: [Double] = if layerIndex == 0 {
                flatten(inputMatrix)
            } else {
                layerOutputs[layerIndex - 1]
            }

            for neuronIndex in 0 ..< weights[layerIndex].count {
                for inputIndex in 0 ..< inputsToLayer.count {
                    var gradient =
                        layerErrors[layerIndex][neuronIndex]
                            * inputsToLayer[inputIndex]

                    gradient =
                        max(-1.0, min(1.0, gradient))

                    weights[layerIndex][neuronIndex][0, inputIndex] -=
                        learningRate * gradient
                }

                bias[layerIndex][neuronIndex, 0] -=
                    learningRate * layerErrors[layerIndex][neuronIndex]
            }
        }
    }

    // MARK: - Action Prediction

    mutating func predictAction(input inputMatrix: Matrix<Double>) -> Int {
        let predictionMatrix = predict(input: inputMatrix)

        let predictionValues = flatten(predictionMatrix)

        return argmax(predictionValues)
    }

    // MARK: - Initializer

    public init(layers: [Int], talent: TalentType) {
        self.talent = talent
        weights = []
        bias = []

        for layerIndex in 0 ..< (layers.count - 1) {
            let inputSize = layers[layerIndex]
            let outputSize = layers[layerIndex + 1]

            let initializationScale =
                sqrt(2.0 / Double(inputSize))

            var layerWeights: [Matrix<Double>] = []

            for _ in 0 ..< outputSize {
                let weightMatrix = Matrix(
                    rows: 1,
                    columns: inputSize,
                    grid: (0 ..< inputSize).map { _ in
                        Double.random(in: -1 ... 1)
                            * initializationScale
                    },
                )

                layerWeights.append(weightMatrix)
            }

            weights.append(layerWeights)

            let layerBias = Matrix(
                rows: outputSize,
                columns: 1,
                grid: Array(repeating: 0.0, count: outputSize),
            )

            bias.append(layerBias)
        }
    }
}
