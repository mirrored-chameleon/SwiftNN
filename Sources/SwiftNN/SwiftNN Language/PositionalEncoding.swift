//
//  Positional_Encoding.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 23/07/2026.
//

import SwiftNN
import Foundation

public protocol PositionalEncoding {

    func forward(_ embeddings: Matrix<Double>) -> Matrix<Double>

}

public struct SinusoidalPositionalEncoding: PositionalEncoding {

    public init() {}

    public func forward(_ embeddings: Matrix<Double>) -> Matrix<Double> {

        let sequenceLength = embeddings.rows
        let embeddingSize = embeddings.columns

        var matrix = Matrix(
            rows: sequenceLength,
            columns: embeddingSize,
            grid: Array(
                repeating: 0.0,
                count: sequenceLength * embeddingSize
            )
        )

        for position in 0..<sequenceLength {

            for dimension in 0..<embeddingSize {

                let i = dimension / 2

                let denominator = pow(
                    10000.0,
                    Double(2 * i) / Double(embeddingSize)
                )

                let angle = Double(position) / denominator

                if dimension.isMultiple(of: 2) {
                    matrix[position, dimension] = sin(angle)
                } else {
                    matrix[position, dimension] = cos(angle)
                }
            }
        }

        return embeddings + matrix
    }
}