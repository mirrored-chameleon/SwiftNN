import Foundation
internal import Surge

func relu(_ value: Double) -> Double {
    return max(0.0, value)
}

func reluDerivative(_ value: Double) -> Double {
    return value > 0 ? 1.0 : 0.0
}

func softmax(_ values: [Double]) -> [Double] {
    let maxValue = values.max() ?? 0
    let exps = values.map { exp($0 - maxValue) }
    let sumExps = exps.reduce(0, +)
    return exps.map { $0 / sumExps }
}

func crossEntropyLoss(predicted: [Double], target: [Double]) -> Double {
    zip(predicted, target).reduce(0) { acc, pair in
        let (p, t) = pair
        return acc - t * log(p + 1e-12)
    }
}

func flattenMatrix(_ matrix: Matrix<Double>) -> [Double] {
    var result: [Double] = []
    for r in 0..<matrix.rows {
        for c in 0..<matrix.columns {
            result.append(matrix[r,c])
        }
    }
    return result
}

func matrixFromVector(_ values: [Double]) -> Matrix<Double> {
    Matrix(rows: values.count, columns: 1, grid: values)
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
    var weights: [[Matrix<Double>]]       // weights[layer][neuron]
    var bias: [Matrix<Double>]            // biases[layer]
    var layerOutputs: [[Double]]          // post-activation values
    var layerWeightedInputs: [[Double]]   // pre-activation values
    var learningRate: Double = 0.1

    init(layers: [Int], talent: any Talent) {
        self.talent = talent
        self.learningRate = 0.1
        self.layerOutputs = []
        self.layerWeightedInputs = []
        self.weights = []
        self.bias = []

        for i in 0..<(layers.count - 1) {
            let inputSize = layers[i]
            let outputSize = layers[i + 1]

            var layerWeights: [Matrix<Double>] = []
            for _ in 0..<outputSize {
                let weightMatrix = Matrix(
                    rows: 1,
                    columns: inputSize,
                    grid: (0..<inputSize).map { _ in Double.random(in: -0.5...0.5) }
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

    mutating func predict(input inputMatrix: Matrix<Double>) -> Matrix<Double> {
        var currentInputs = flattenMatrix(inputMatrix)
        layerOutputs.removeAll()
        layerWeightedInputs.removeAll()

        for layerIndex in 0..<weights.count {
            var weightedValues: [Double] = []
            var activatedValues: [Double] = []

            let isFinalLayer = layerIndex == weights.count - 1

            for neuronIndex in 0..<weights[layerIndex].count {
                var sum: Double = 0
                for inputIndex in 0..<currentInputs.count {
                    sum += weights[layerIndex][neuronIndex][0,inputIndex] * currentInputs[inputIndex]
                }
                sum += bias[layerIndex][neuronIndex,0]
                weightedValues.append(sum)

                if isFinalLayer {
                    activatedValues.append(sum) // softmax applied later
                } else {
                    activatedValues.append(relu(sum))
                }
            }

            if isFinalLayer {
                activatedValues = softmax(activatedValues)
            }

            layerWeightedInputs.append(weightedValues)
            layerOutputs.append(activatedValues)
            currentInputs = activatedValues
        }

        return matrixFromVector(currentInputs)
    }

    mutating func train(input inputMatrix: Matrix<Double>, target targetMatrix: Matrix<Double>) {
        let predictedMatrix = predict(input: inputMatrix)
        let predicted = flattenMatrix(predictedMatrix)
        let target = flattenMatrix(targetMatrix)

        var layerErrors: [[Double]] = Array(repeating: [], count: weights.count)

        // Softmax + cross-entropy derivative
        layerErrors[weights.count - 1] = zip(predicted, target).map { $0 - $1 }

        // Backpropagate through hidden layers
        if weights.count > 1 {
            for layerIndex in stride(from: weights.count - 2, through: 0, by: -1) {
                var currentErrors = Array(repeating: 0.0, count: weights[layerIndex].count)
                for neuronIndex in 0..<weights[layerIndex].count {
                    var sumError = 0.0
                    for nextNeuronIndex in 0..<weights[layerIndex+1].count {
                        sumError += layerErrors[layerIndex+1][nextNeuronIndex] *
                                    weights[layerIndex+1][nextNeuronIndex][0,neuronIndex]
                    }
                    currentErrors[neuronIndex] = sumError * reluDerivative(layerWeightedInputs[layerIndex][neuronIndex])
                }
                layerErrors[layerIndex] = currentErrors
            }
        }

        // Update weights and biases
        for layerIndex in 0..<weights.count {
            let inputs = layerIndex == 0 ? flattenMatrix(inputMatrix) : layerOutputs[layerIndex-1]
            for neuronIndex in 0..<weights[layerIndex].count {
                for inputIndex in 0..<inputs.count {
                    let gradient = learningRate * layerErrors[layerIndex][neuronIndex] * inputs[inputIndex]
                    weights[layerIndex][neuronIndex][0,inputIndex] += gradient
                }
                let biasGradient = learningRate * layerErrors[layerIndex][neuronIndex]
                bias[layerIndex][neuronIndex,0] += biasGradient
            }
        }
    }
}
