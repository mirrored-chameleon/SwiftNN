//
//  SwiftNNTests.swift
//  SwiftNNTests
//
//  Created by Davyn Monagle on 1/3/2026.
//

import Foundation 


// MARK:- Talent Protocol

/// A protocl that has 2 functions of `Encode` and `Decode` so that the model can take in different inputs and output different outputs.
///
/// There are 2 generics of `Input` and `Output` so that the model can take in different types you teach it, so it could be a `String` or an `Int` for instance.

public protocol Talent {
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

public protocol Network {
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


public struct BasicModel: Network {

    public var talent: any Talent
    public var weights: [[Matrix<Double>]]
    public var bias: [Matrix<Double>]
    
    var layerOutputs: [[Double]] = []
    
    var learningRate: Double = 0.001
    
    // MARK: - Prediction
    
    mutating public func predict(input inputMatrix: Matrix<Double>) -> Matrix<Double> {
        
        var currentLayerInputs = flatten(inputMatrix)
        
        layerOutputs.removeAll()
        
        for layerIndex in 0..<weights.count {
            
            let isFinalLayer = layerIndex == weights.count - 1
            
            var currentLayerOutputs: [Double] = []
            
            for neuronIndex in 0..<weights[layerIndex].count {
                
                var weightedSum: Double = 0.0
                
                for inputIndex in 0..<currentLayerInputs.count {
                    
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
            grid: currentLayerInputs
        )
    }
    
    // MARK: - Training
    
    mutating public func train(
        input inputMatrix: Matrix<Double>,
        target targetMatrix: Matrix<Double>
    ) {
        
        let predictedMatrix = predict(input: inputMatrix)
        
        let predictedValues = flatten(predictedMatrix)
        let targetValues = flatten(targetMatrix)
        
        var layerErrors: [[Double]] =
            Array(repeating: [], count: weights.count)
        
        let outputLayerIndex = weights.count - 1
        
        var outputLayerErrors: [Double] = []
        
        for valueIndex in 0..<predictedValues.count {
            
            let prediction = predictedValues[valueIndex]
            let target = targetValues[valueIndex]
            
            outputLayerErrors.append(prediction - target)
        }
        
        layerErrors[outputLayerIndex] = outputLayerErrors
        
        if weights.count > 1 {
            
            for layerIndex in stride(
                from: weights.count - 2,
                through: 0,
                by: -1
            ) {
                
                let neuronCount = weights[layerIndex].count
                
                var currentLayerErrors =
                    Array(repeating: 0.0, count: neuronCount)
                
                for neuronIndex in 0..<neuronCount {
                    
                    var propagatedError: Double = 0.0
                    
                    for nextNeuronIndex in 0..<weights[layerIndex + 1].count {
                        
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
        
        for layerIndex in 0..<weights.count {
            
            let inputsToLayer: [Double]
            
            if layerIndex == 0 {
                inputsToLayer = flatten(inputMatrix)
            } else {
                inputsToLayer = layerOutputs[layerIndex - 1]
            }
            
            for neuronIndex in 0..<weights[layerIndex].count {
                
                for inputIndex in 0..<inputsToLayer.count {
                    
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
    
    public init(layers: [Int], talent: any Talent) {
        
        self.talent = talent
        self.weights = []
        self.bias = []
        
        for layerIndex in 0..<(layers.count - 1) {
            
            let inputSize = layers[layerIndex]
            let outputSize = layers[layerIndex + 1]
            
            let initializationScale =
                sqrt(2.0 / Double(inputSize))
            
            var layerWeights: [Matrix<Double>] = []
            
            for _ in 0..<outputSize {
                
                let weightMatrix = Matrix(
                    rows: 1,
                    columns: inputSize,
                    grid: (0..<inputSize).map { _ in
                        Double.random(in: -1...1)
                        * initializationScale
                    }
                )
                
                layerWeights.append(weightMatrix)
            }
            
            weights.append(layerWeights)
            
            let layerBias = Matrix(
                rows: outputSize,
                columns: 1,
                grid: Array(repeating: 0.0, count: outputSize)
            )
            
            bias.append(layerBias)
        }
    }
}
