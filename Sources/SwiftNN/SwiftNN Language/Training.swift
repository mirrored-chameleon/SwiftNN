//
//  Training.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 13/08/2026.
//

import Foundation

// MARK: - Loss Result

// MARK: - Trainer

public struct TrainingExample {
    public let input: Matrix<Double>
    public let target: Matrix<Double>
}

public struct LossResult {
    public let loss: Double
    public let gradient: Matrix<Double>
}

public func subtractScaled(
    _ matrix: inout Matrix<Double>,
    gradient: Matrix<Double>,
    learningRate: Double,
) {
    precondition(
        matrix.rows == gradient.rows && matrix.columns == gradient.columns,
        "Matrix dimensions must match.",
    )

    for row in 0 ..< matrix.rows {
        for column in 0 ..< matrix.columns {
            matrix[row, column] -=
                learningRate * gradient[row, column]
        }
    }
}

public struct Trainer {
    public var learningRate: Double
    public var epochs: Int

    public init(
        learningRate: Double = 0.001,
        epochs: Int = 1000,
    ) {
        self.learningRate = learningRate
        self.epochs = epochs
    }

    // MARK: - Training

    public mutating func train(
        examples: [TrainingExample],
        trainingStep: (
            TrainingExample,
            Double,
        ) -> Double,
    ) {
        guard !examples.isEmpty else {
            return
        }

        for epoch in 0 ..< epochs {
            var totalLoss = 0.0

            for example in examples {
                totalLoss += trainingStep(
                    example,
                    learningRate,
                )
            }

            let averageLoss =
                totalLoss / Double(examples.count)

            if epoch % 1 == 0 {
                print(
                    "Epoch \(epoch) | Loss: \(averageLoss)",
                )
            }
        }
    }

    // MARK: - Softmax

    private func softmax(
        _ values: [Double],
    ) -> [Double] {
        guard !values.isEmpty else {
            return []
        }

        let maximum =
            values.max() ?? 0.0

        let exponentials =
            values.map {
                exp($0 - maximum)
            }

        let sum =
            exponentials.reduce(0.0, +)

        return exponentials.map {
            $0 / sum
        }
    }

    // MARK: - Cross Entropy

    private func crossEntropy(
        _ prediction: Matrix<Double>,
        _ target: Matrix<Double>,
    ) -> Double {
        precondition(
            prediction.grid.count == target.grid.count,
            "Prediction and target sizes must match.",
        )

        var loss = 0.0

        for row in 0 ..< prediction.rows {
            let rowValues = prediction[row]

            let maximum =
                rowValues.max() ?? 0.0

            var exponentials: [Double] = []

            for value in rowValues {
                exponentials.append(
                    exp(value - maximum),
                )
            }

            let sum =
                exponentials.reduce(0.0, +)

            for column in 0 ..< prediction.columns {
                let probability =
                    exponentials[column] / sum

                let targetValue =
                    target[row, column]

                if targetValue > 0.0 {
                    loss -=
                        targetValue
                        * log(
                            max(probability, 1e-12),
                        )
                }
            }
        }

        return loss
    }
}
