//
//  Activations.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 22/08/2026.
//

import Foundation

public func relu(_ inputValue: Double) -> Double {
    max(0.0, inputValue)
}

public func reluMatrix(_ inputMatrix: Matrix<Double>) -> Matrix<Double> {
    Matrix(
        rows: inputMatrix.rows,
        columns: inputMatrix.columns,
        grid: inputMatrix.grid.map { value in
            relu(value)
        },
    )
}

public func reluDerivative(_ inputValue: Double) -> Double {
    inputValue > 0.0 ? 1.0 : 0.0
}

public func sigmoid(_ inputValue: Double) -> Double {
    1.0 / (1.0 + exp(-inputValue))
}

public func tanhDerivative(fromOutput outputValue: Double) -> Double {
    1.0 - (outputValue * outputValue)
}
