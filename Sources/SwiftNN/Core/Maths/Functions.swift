//
//  Functions.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 22/08/2026.
//

import Foundation

public func softmax(_ inputValues: [Double]) -> [Double] {

    let largestValue = inputValues.max() ?? 0.0

    let exponentValues = inputValues.map { value in
        exp(value - largestValue)
    }

    let exponentSum = exponentValues.reduce(0.0, +)

    return exponentValues.map { exponentValue in
        exponentValue / exponentSum
    }
}
public func argmax(_ values: [Double]) -> Int {

    var bestIndex = 0
    var bestValue = values[0]

    for index in 1..<values.count {

        let currentValue = values[index]

        if currentValue > bestValue {
            bestValue = currentValue
            bestIndex = index
        }
    }

    return bestIndex
}

public func clipped(_ value: Double, limit: Double) -> Double {
    return max(-limit, min(limit, value))
}
public func addBias(
    _ matrix: Matrix<Double>,
    _ bias: Matrix<Double>
) -> Matrix<Double> {

    precondition(
        bias.rows == 1 && bias.columns == matrix.columns,
        "Bias matrix must have one row and match the matrix column count."
    )

    var result = matrix

    for rowIndex in 0..<result.rows {
        for columnIndex in 0..<result.columns {
            result[rowIndex, columnIndex] += bias[0, columnIndex]
        }
    }

    return result
}
public func flatten(_ matrix: Matrix<Double>) -> [Double] {

    var flattenedValues: [Double] = []

    flattenedValues.reserveCapacity(matrix.rows * matrix.columns)

    for rowIndex in 0..<matrix.rows {

        for columnIndex in 0..<matrix.columns {

            flattenedValues.append(matrix[rowIndex, columnIndex])

        }
    }

    return flattenedValues
}
