//
//  OutputProjection.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 13/08/2026.
//

import Foundation

struct OutputProjection: Codable {
    var weights: Matrix<Double>
    var bias: Matrix<Double>

    var lastInput: Matrix<Double>?
    
    enum CodingKeys: String, CodingKey {
        case weights
        case bias
    }

    mutating func forward(
        _ input: Matrix<Double>,
    ) -> Matrix<Double> {
        lastInput = input

        let output = input * weights

        var result = output

        for row in 0 ..< result.rows {
            for column in 0 ..< result.columns {
                result[row, column] += bias[0, column]
            }
        }

        return result
    }

    func backward(
        _ gradient: Matrix<Double>,
    ) -> (
        weightGradient: Matrix<Double>,
        biasGradient: Matrix<Double>,
        inputGradient: Matrix<Double>,
    ) {
        guard let input = lastInput else {
            fatalError("OutputProjection backward called before forward.")
        }

        let weightGradient =
            input.transposed * gradient

        var biasGradient =
            Matrix(
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

        let inputGradient =
            gradient * weights.transposed

        return (
            weightGradient,
            biasGradient,
            inputGradient,
        )
    }
}
