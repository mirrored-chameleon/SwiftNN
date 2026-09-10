//
//  Network.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 22/08/2026.
//

// A protocol describing the basic requirements of a neural network.
//
// Designed to be minimal, this protocol defines the core components
// shared by neural networks: weights, biases, and training behavior.
//
// The weights and biases are represented as `Matrix<Double>` values.
// Implementations are expected to perform forward propagation for
// prediction and backpropagation during training.

public protocol Network {
    associatedtype TalentType: Talent

    var talent: TalentType { get }
    var weights: [[Matrix<Double>]] { get set }
    var bias: [Matrix<Double>] { get set }

    // `train()` function
    //
    // Takes in a `Matrix<Double>` as input, and a target input of the same type.

    mutating func train(input: Matrix<Double>, target: Matrix<Double>)

    // `predict()` function
    //
    // Takes in a `Matrix<Double>` type and returns a `Matrix<Double>`.

    mutating func predict(input: Matrix<Double>) -> Matrix<Double>
}

/// This is so that the Network can also be called Model if that is more comfortable for users.
public typealias Model = Network
