//
//  SwiftNN.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 1/3/2026.
//

import Foundation
internal import Surge


// MARK: Protocols

/// A protocl that has 2 functions of `Encode` and `Decode` so that the model can take in different inputs and output different outputs.
///
/// There are 2 generics of `Input` and `Output` so that the model can take in different types you teach it, so it could be a `String` or an `Int` for instance.

protocol Talent {
    
    /// Input type for the model
    ///
    /// Generic type for `Talent` so that the `Network` can take in any input that you want it to.
    
    associatedtype Input
    
    /// Output type for the model
    ///
    /// Generic type for `Talent` so that the `Network` can output any output you want it to.
    
    associatedtype Output
    
    /// `encode` function inside of `Talent`
    ///
    /// The function is used so that the `Network` is able to encode any type you want it to.
    
    func encode(_ input: Input) -> Matrix<Double>
    
    /// `decode` function inside of `Talent`
    ///
    /// The function is used so that
    
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
    /// Weights for the `Network` to use
    ///
    /// A simple weights variable, for the `Network` to use to train and predict.
    ///
    /// It uses a `Matrix<Double>` type.
    
    var weights: Matrix<Double> { get set }
    /// Biases for the model to use
    ///
    /// A simple bias variable for the `Network` to use to train and predict.
    ///
    /// It uses a `Matrix<Double>` type.
    
    var bias: Matrix<Double> { get set }
    
    /// `train()` function
    ///
    /// Takes in a `Matrix<Double>` as input, and a target input of the same type.
    
    func train(input: Matrix<Double>, target: Matrix<Double>)
    
    /// `predict()` function
    ///
    /// Takes in a `Matrix<Double>` type and returns a `Matrix<Double>`.
    
    func predict(input: Matrix<Double>) -> Matrix<Double>
    
}


// This is so that the Network can also be called Model if that is more comfortable for users.
typealias Model = Network

