//
//  SwiftNNTests.swift
//  SwiftNNTests
//
//  Created by Davyn Monagle on 1/3/2026.
//

import Foundation
internal import Surge

func relu(_ value: Double) -> Double {
    return max(0.0, value)
}

func reluDerivative(_ value: Double) -> Double {
    return value > 0 ? 1.0 : 0.0
}

// MARK:- Talent Protocol

/// A protocl that has 2 functions of `Encode` and `Decode` so that the model can take in different inputs and output different outputs.
///
/// There are 2 generics of `Input` and `Output` so that the model can take in different types you teach it, so it could be a `String` or an `Int` for instance.

protocol Talent {
    /// Tokens or the possible classification for a type could be
    ///
    /// Tokens are used for the possible outcomes of a prediction. Usually models return an array of predictions, and the highest value is the most likely output according to the model. We can then find out what value it thinks is the most likely outcome by looking up it's index in the tokens variable.
    
    var tokens: [Output] { get set }
    
    /// Input type for the model
    ///
    /// Generic type for `Talent` so that the `Network` can take in any input that you want it to.
    
    associatedtype Input
    
    /// Output type for the `Network`
    ///
    /// Generic type for `Talent` so that the `Network` can output any output you want it to.
    
    associatedtype Output
    
    /// `encode` function inside of `Talent`
    ///
    /// The function is used so that the `Network` is able to encode any type you want it to.
    
    func encode(_ input: Input) -> Matrix<Double>
    
    /// `decode` function inside of `Talent`
    ///
    /// The function is used so that the `Network` is able to turn the output matrix back into something usable.
    
    func decode(_ output: Matrix<Double>) -> Output
}


/// A protocol describing the basic requirements of a neural network.
///
/// Designed to be minimal, this protocol defines the core components
/// shared by neural networks: weights, biases, and training behavior.
///
/// The weights and biases are represented as `Matrix<Double>` values.
/// Implementations are expected to perform forward propagation for
/// prediction and backpropagation during training.

protocol Network {
    var talent: any Talent { get }
    var weights: [[Matrix<Double>]] { get set }
    var bias: [Matrix<Double>] { get set }
    
    /// `train()` function
    ///
    /// Takes in a `Matrix<Double>` as input, and a target input of the same type.
    
    mutating func train(input: Matrix<Double>, target: Matrix<Double>)
    
    /// `predict()` function
    ///
    /// Takes in a `Matrix<Double>` type and returns a `Matrix<Double>`.
    
    mutating func predict(input: Matrix<Double>) -> Matrix<Double>
}

// This is so that the Network can also be called Model if that is more comfortable for users.
typealias Model = Network


struct BasicModel: Network {
    
    var talent: any Talent
    var weights: [[Matrix<Double>]]
    var bias: [Matrix<Double>]
    var layerOutputs: [[Double]]
    var learningRate = 0.01
    
    mutating func train(input inputMatrix: Matrix<Double>, target targetMatrix: Matrix<Double>) {
        
        // 1️⃣ Forward pass
        let predictedMatrix = predict(input: inputMatrix)
        
        // 2️⃣ Flatten target and predicted for easy indexing
        var predictedValues: [Double] = []
        var targetValues: [Double] = []
        
        for row in 0..<predictedMatrix.rows {
            for col in 0..<predictedMatrix.columns {
                predictedValues.append(predictedMatrix[row, col])
            }
        }
        
        for row in 0..<targetMatrix.rows {
            for col in 0..<targetMatrix.columns {
                targetValues.append(targetMatrix[row, col])
            }
        }
        
        // 3️⃣ Compute output layer error
        var layerErrors: [[Double]] = Array(repeating: [], count: weights.count)
        
        let outputLayerIndex = weights.count - 1
        var outputLayerError: [Double] = []
        
        for i in 0..<predictedValues.count {
            // For output layer, derivative is 1 (linear output)
            let error = targetValues[i] - predictedValues[i]
            outputLayerError.append(error)
        }
        layerErrors[outputLayerIndex] = outputLayerError
        
        // 4️⃣ Backpropagate through hidden layers
        for layerIndex in (0..<weights.count-1).reversed() {
            var currentLayerError: [Double] = Array(repeating: 0.0, count: weights[layerIndex].count)
            
            for neuronIndex in 0..<weights[layerIndex].count {
                var errorSum: Double = 0.0
                // Each neuron error = sum of next layer neuron errors * weight * ReLU derivative
                for nextNeuronIndex in 0..<weights[layerIndex+1].count {
                    let nextError = layerErrors[layerIndex+1][nextNeuronIndex]
                    let weightToNextNeuron = weights[layerIndex+1][nextNeuronIndex][0, neuronIndex]
                    errorSum += nextError * weightToNextNeuron
                }
                // Multiply by ReLU derivative of neuron output
                let neuronOutput = layerOutputs[layerIndex][neuronIndex]
                currentLayerError[neuronIndex] = errorSum * reluDerivative(neuronOutput)
            }
            layerErrors[layerIndex] = currentLayerError
        }
        
        // 5️⃣ Update weights and biases
        for layerIndex in 0..<weights.count {
            let inputsToUse: [Double] = layerIndex == 0
                ? (0..<inputMatrix.rows).flatMap { row in
                    (0..<inputMatrix.columns).map { col in inputMatrix[row, col] }
                }
                : layerOutputs[layerIndex-1] // previous layer outputs
            
            for neuronIndex in 0..<weights[layerIndex].count {
                for inputIndex in 0..<inputsToUse.count {
                    let gradient = learningRate * layerErrors[layerIndex][neuronIndex] * inputsToUse[inputIndex]
                    weights[layerIndex][neuronIndex][0, inputIndex] += gradient
                }
                // Update bias
                let biasGradient = learningRate * layerErrors[layerIndex][neuronIndex]
                bias[layerIndex][neuronIndex, 0] += biasGradient
            }
        }
    }
    
    mutating func predict(input inputMatrix: Matrix<Double>) -> Matrix<Double> {
        
        // Convert input matrix into flat array
        var currentLayerInputValues: [Double] = []
        
        for rowIndex in 0..<inputMatrix.rows {
            for columnIndex in 0..<inputMatrix.columns {
                currentLayerInputValues.append(inputMatrix[rowIndex, columnIndex])
            }
        }
        
        // Clear previous outputs
        layerOutputs.removeAll()
        
        // Loop through each layer
        for layerIndex in 0..<weights.count {
            
            var currentLayerOutputValues: [Double] = []
            
            let isFinalLayer = (layerIndex == weights.count - 1)
            
            // Loop through each neuron in the layer
            for neuronIndex in 0..<weights[layerIndex].count {
                
                var weightedSum: Double = 0.0
                
                // Multiply each input by corresponding weight
                for inputIndex in 0..<currentLayerInputValues.count {
                    
                    let weightValue = weights[layerIndex][neuronIndex][0, inputIndex]
                    let inputValue = currentLayerInputValues[inputIndex]
                    
                    weightedSum += weightValue * inputValue
                }
                
                // Add bias
                let biasValue = bias[layerIndex][neuronIndex, 0]
                weightedSum += biasValue
                
                // Apply ReLU only if NOT final layer
                if !isFinalLayer {
                    weightedSum = relu(weightedSum)
                }
                
                currentLayerOutputValues.append(weightedSum)
            }
            
            // Store this layer’s outputs
            layerOutputs.append(currentLayerOutputValues)
            
            // Output becomes input to next layer
            currentLayerInputValues = currentLayerOutputValues
        }
        
        // Convert final output array back into Matrix
        return Matrix(
            rows: currentLayerInputValues.count,
            columns: 1,
            grid: currentLayerInputValues
        )
    }
    
    init(layers: [Int], talent: any Talent) {
        self.talent = talent
        self.learningRate = 0.01
        self.layerOutputs = []
        
        self.weights = []
        self.bias = []
        
        for i in 0..<(layers.count - 1) {
            let inputSize = layers[i]
            let outputSize = layers[i + 1]
            
            // Create weight matrix for each neuron in this layer
            var layerWeights: [Matrix<Double>] = []
            for _ in 0..<outputSize {
                // 1×inputSize random weight matrix
                let weightMatrix = Matrix(
                    rows: 1,
                    columns: inputSize,
                    grid: (0..<inputSize).map { _ in Double.random(in: -0.5...0.5) }
                )
                layerWeights.append(weightMatrix)
            }
            weights.append(layerWeights)
            
            // Bias for this layer
            let layerBias = Matrix(
                rows: outputSize,
                columns: 1,
                grid: Array(repeating: 0.0, count: outputSize)
            )
            bias.append(layerBias)
        }
    }
}
