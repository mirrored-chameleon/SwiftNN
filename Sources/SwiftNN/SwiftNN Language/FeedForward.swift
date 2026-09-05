//
// FeedForward.swift
// SwiftNN Language
//
// Created by Davyn Monagle on 13/08/2026.
//

import Foundation

public protocol FeedForwardLayer {
    mutating func forward(_ input: Matrix<Double>) -> Matrix<Double>
}

public struct FeedForward: FeedForwardLayer, Codable {
    public var lastInput: Matrix<Double>?
    public var lastHiddenPreActivation: Matrix<Double>?
    public var lastHidden: Matrix<Double>?
    public var inputWeights: Matrix<Double>
    public var inputBias: Matrix<Double>
    public var outputWeights: Matrix<Double>
    public var outputBias: Matrix<Double>

    public enum CodingKeys: String, CodingKey {
        case inputWeights
        case inputBias
        case outputWeights
        case outputBias
    }

    public mutating func forward(
        _ input: Matrix<Double>) -> Matrix<Double>
    {
        lastInput = input
        let hiddenPreActivation =
            addBias(
                input * inputWeights,
                inputBias,
            )
        lastHiddenPreActivation =
            hiddenPreActivation
        let hidden =
            reluMatrix(hiddenPreActivation)
        lastHidden =
            hidden
        return addBias(
            hidden * outputWeights,
            outputBias,
        )
    }

    public func backwardOutput(
        _ gradient: Matrix<Double>,
    ) -> (
        weightGradient: Matrix<Double>,
        biasGradient: Matrix<Double>,
        hiddenGradient: Matrix<Double>,
    ) {
        guard let hidden = lastHidden else {
            fatalError("FeedForward backward called before forward.")
        }
        let weightGradient =
            hidden.transposed * gradient
        var biasGradient =
            Matrix<Double>(
                rows: 1,
                columns: gradient.columns,
                grid: Array(
                    repeating: 0.0,
                    count: gradient.columns,
                ),
            )
        for row in 0 ..< gradient.rows {
            for column in 0 ..< gradient.columns {
                biasGradient[0, column] += gradient[row, column]
            }
        }
        let hiddenGradient =
            gradient * outputWeights.transposed
        return (
            weightGradient,
            biasGradient,
            hiddenGradient,
        )
    }

    public func backwardInput(
        _ gradient: Matrix<Double>,
    ) -> (
        weightGradient: Matrix<Double>,
        biasGradient: Matrix<Double>,
        inputGradient: Matrix<Double>,
    ) {
        guard
            let input = lastInput,
            let preActivation = lastHiddenPreActivation
        else {
            fatalError("FeedForward backward called before forward.")
        }
        var reluGradient =
            Matrix<Double>(
                rows: gradient.rows,
                columns: gradient.columns,
                grid: Array(
                    repeating: 0.0,
                    count: gradient.rows * gradient.columns,
                ),
            )
        for row in 0 ..< gradient.rows {
            for column in 0 ..< gradient.columns {
                if preActivation[row, column] > 0.0 {
                    reluGradient[row, column] =
                        gradient[row, column]
                }
            }
        }
        let weightGradient =
            input.transposed * reluGradient
        var biasGradient =
            Matrix<Double>(
                rows: 1,
                columns: reluGradient.columns,
                grid: Array(
                    repeating: 0.0,
                    count: reluGradient.columns,
                ),
            )
        for row in 0 ..< reluGradient.rows {
            for column in 0 ..< reluGradient.columns {
                biasGradient[0, column] +=
                    reluGradient[row, column]
            }
        }
        let inputGradient =
            reluGradient * inputWeights.transposed
        return (
            weightGradient,
            biasGradient,
            inputGradient,
        )
    }

    public init(
        inputSize: Int,
        hiddenSize: Int,
    ) {
        inputWeights =
            randomWeights(
                rows: inputSize,
                columns: hiddenSize,
            )
        inputBias =
            Matrix<Double>(
                rows: 1,
                columns: hiddenSize,
                grid: Array(
                    repeating: 0.0,
                    count: hiddenSize,
                ),
            )
        outputWeights =
            randomWeights(
                rows: hiddenSize,
                columns: inputSize,
            )
        outputBias =
            Matrix<Double>(
                rows: 1,
                columns: inputSize,
                grid: Array(
                    repeating: 0.0,
                    count: inputSize,
                ),
            )
    }
}
