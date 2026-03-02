import Foundation
internal import Surge

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
    var weights: Matrix<Double> { get set }
    var bias: Matrix<Double> { get set }
    
    /// `train()` function
    ///
    /// Takes in a `Matrix<Double>` as input, and a target input of the same type.
    
    mutating func train(input: Matrix<Double>, target: Matrix<Double>)
    
    /// `predict()` function
    ///
    /// Takes in a `Matrix<Double>` type and returns a `Matrix<Double>`.
    
    func predict(input: Matrix<Double>) -> Matrix<Double>
}

// This is so that the Network can also be called Model if that is more comfortable for users.
typealias Model = Network


struct NumberRecognitionTalent: Talent {
    var tokens: [Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    
    typealias Input = [Double]
    typealias Output = Int
    
    func encode(_ input: [Double]) -> Matrix<Double> {
        // Flattened 28x28 array -> 28x28 matrix
        return Matrix(rows: 28, columns: 28, grid: input)
    }
    
    func decode(_ output: Matrix<Double>) -> Int {
        var outputFlattened: [Double] = []
        
        for rowIndex in 0..<output.rows {
            for columnIndex in 0..<output.columns {
                outputFlattened.append(output[rowIndex, columnIndex])
            }
        }
        
        if let maxIndex = outputFlattened.firstIndex(of: outputFlattened.max() ?? 0.0) {
            return tokens[maxIndex]
        } else {
            return tokens.first ?? 0
        }
    }
}


struct BasicModel: Network {
    
    var talent: any Talent
    var weights: Matrix<Double>
    var bias: Matrix<Double>
    var learningRate = 0.01
    
    mutating func train(input inputMatrix: Matrix<Double>, target targetMatrix: Matrix<Double>) {
        // Forward pass
        let predictedMatrix = predict(input: inputMatrix)
        
        // Calculate the difference between target and prediction
        let errorMatrix = targetMatrix - predictedMatrix
        
        // Update weights manually
        for rowIndex in 0..<weights.rows {
            for columnIndex in 0..<weights.columns {
                let errorValue = errorMatrix[rowIndex, 0]
                let inputValue = inputMatrix[columnIndex, 0]
                let weightUpdate = learningRate * errorValue * inputValue
                weights[rowIndex, columnIndex] += weightUpdate
            }
        }
        
        // Update bias
        for biasRowIndex in 0..<bias.rows {
            let biasUpdate = learningRate * errorMatrix[biasRowIndex, 0]
            bias[biasRowIndex, 0] += biasUpdate
        }
    }
    
    func predict(input: Matrix<Double>) -> Matrix<Double> {
        // Simple forward pass: weights * input + bias
        return (weights * input) + bias
    }
}
