// MARK: - Training Example

import Foundation
//
//  Training.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 13/08/2026.
//

import SwiftNN

// MARK: - Loss Result

// MARK: - Trainer

struct TrainingExample {

    let input: Matrix<Double>
    let target: Matrix<Double>

}
struct LossResult {

    let loss: Double
    let gradient: Matrix<Double>

}
func subtractScaled(
    _ matrix: inout Matrix<Double>,
    gradient: Matrix<Double>,
    learningRate: Double
) {

    precondition(
        matrix.rows == gradient.rows && matrix.columns == gradient.columns,
        "Matrix dimensions must match."
    )

    for row in 0..<matrix.rows {
        for column in 0..<matrix.columns {

            matrix[row, column] -=
                learningRate * gradient[row, column]
        }
    }
}
struct Trainer {

    var learningRate: Double
    var epochs: Int

    init(
        learningRate: Double = 0.001,
        epochs: Int = 1000
    ) {
        self.learningRate = learningRate
        self.epochs = epochs
    }

    // MARK: - Training

    mutating func train(
        examples: [TrainingExample],
        trainingStep: (
            TrainingExample,
            Double
        ) -> Double
    ) {

        guard !examples.isEmpty else {
            return
        }

        for epoch in 0..<epochs {

            var totalLoss = 0.0

            for example in examples {

                totalLoss += trainingStep(
                    example,
                    learningRate
                )
            }

            let averageLoss =
                totalLoss / Double(examples.count)

            if epoch % 1 == 0 {
                print(
                    "Epoch \(epoch) | Loss: \(averageLoss)"
                )
            }
        }
    }

    // MARK: - Softmax

    private func softmax(
        _ values: [Double]
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
        _ target: Matrix<Double>
    ) -> Double {

        precondition(
            prediction.grid.count == target.grid.count,
            "Prediction and target sizes must match."
        )

        var loss = 0.0

        for row in 0..<prediction.rows {

            let rowValues = prediction[row]

            let maximum =
                rowValues.max() ?? 0.0

            var exponentials: [Double] = []

            for value in rowValues {
                exponentials.append(
                    exp(value - maximum)
                )
            }

            let sum =
                exponentials.reduce(0.0, +)

            for column in 0..<prediction.columns {

                let probability =
                    exponentials[column] / sum

                let targetValue =
                    target[row, column]

                if targetValue > 0.0 {
                    loss -=
                        targetValue
                        * log(
                            max(probability, 1e-12)
                        )
                }
            }
        }

        return loss
    }
}
