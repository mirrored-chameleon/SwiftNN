//
//  WeightInitialization.swift
//  SwiftNN Language
//

import Foundation
import SwiftNN

func randomWeights(
    rows: Int,
    columns: Int,
    scale: Double = 0.05
) -> Matrix<Double> {

    var values: [Double] = []

    for _ in 0..<(rows * columns) {
        values.append(
            Double.random(
                in: -scale...scale
            )
        )
    }

    return Matrix<Double>(
        rows: rows,
        columns: columns,
        grid: values
    )
}