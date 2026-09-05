//
//  SelfAttention.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 13/08/2026.
//

import Foundation

public protocol Attention {
    mutating func forward(
        _ input: Matrix<Double>,
    ) -> Matrix<Double>
}

public struct SelfAttention: Attention, Codable {
    public var queryWeights: Matrix<Double>
    public var keyWeights: Matrix<Double>
    public var valueWeights: Matrix<Double>

    public var lastInput: Matrix<Double>?
    public var lastQuery: Matrix<Double>?
    public var lastKey: Matrix<Double>?
    public var lastValue: Matrix<Double>?
    public var lastAttentionWeights: Matrix<Double>?

    // MARK: - Codable

    public enum CodingKeys: String, CodingKey {
        case queryWeights
        case keyWeights
        case valueWeights
    }

    // MARK: - Forward

    public mutating func forward(
        _ input: Matrix<Double>,
    ) -> Matrix<Double> {
        lastInput = input

        let query =
            input * queryWeights

        let key =
            input * keyWeights

        let value =
            input * valueWeights

        lastQuery = query
        lastKey = key
        lastValue = value

        // Q × Kᵀ
        let scores =
            query * key.transposed

        // Scaled dot-product attention
        let scale =
            sqrt(Double(key.columns))

        let scaledScores =
            scores / scale

        // Softmax each row
        let attentionWeights = Matrix(
            rows: scaledScores.rows,
            columns: scaledScores.columns,
            grid:
            (0 ..< scaledScores.rows).flatMap { row in
                softmax(
                    scaledScores[row],
                )
            },
        )

        lastAttentionWeights =
            attentionWeights

        // Attention × V
        return attentionWeights * value
    }

    // MARK: - Backward: Attention Output

    public func backwardOutput(
        _ gradient: Matrix<Double>,
    ) -> (
        attentionGradient: Matrix<Double>,
        valueGradient: Matrix<Double>,
    ) {
        guard
            let value = lastValue,
            let attentionWeights =
            lastAttentionWeights
        else {
            fatalError(
                "SelfAttention backward called before forward.",
            )
        }

        // dL/dA = dL/dOutput × Vᵀ
        let attentionGradient =
            gradient * value.transposed

        // dL/dV = Aᵀ × dL/dOutput
        let valueGradient =
            attentionWeights.transposed * gradient

        return (
            attentionGradient,
            valueGradient,
        )
    }

    // MARK: - Backward: Softmax

    public func softmaxBackward(
        _ gradient: Matrix<Double>,
    ) -> Matrix<Double> {
        guard
            let attentionWeights =
            lastAttentionWeights
        else {
            fatalError(
                "Softmax backward called before forward.",
            )
        }

        var result = Matrix(
            rows: gradient.rows,
            columns: gradient.columns,
            grid: Array(
                repeating: 0.0,
                count:
                gradient.rows *
                    gradient.columns,
            ),
        )

        for row in 0 ..< gradient.rows {
            let weights =
                attentionWeights[row]

            let incoming =
                gradient[row]

            var dotProduct = 0.0

            for column in 0 ..< gradient.columns {
                dotProduct +=
                    incoming[column] *
                    weights[column]
            }

            for column in 0 ..< gradient.columns {
                result[row, column] =
                    weights[column] *
                    (
                        incoming[column]
                            - dotProduct
                    )
            }
        }

        return result
    }

    // MARK: - Backward: Scores

    public func backwardScores(
        _ gradient: Matrix<Double>,
    ) -> (
        queryGradient: Matrix<Double>,
        keyGradient: Matrix<Double>,
    ) {
        guard
            let query = lastQuery,
            let key = lastKey
        else {
            fatalError(
                "Score backward called before forward.",
            )
        }

        let scale =
            sqrt(Double(key.columns))

        // dL/dScores
        let scoreGradient =
            gradient / scale

        // QKᵀ
        //
        // dQ = dScores × K
        let queryGradient =
            scoreGradient * key

        // dK = dScoresᵀ × Q
        let keyGradient =
            scoreGradient.transposed * query

        return (
            queryGradient,
            keyGradient,
        )
    }

    // MARK: - Backward: Weights

    public func backwardWeights(
        queryGradient: Matrix<Double>,
        keyGradient: Matrix<Double>,
        valueGradient: Matrix<Double>,
    ) -> (
        queryWeightGradient: Matrix<Double>,
        keyWeightGradient: Matrix<Double>,
        valueWeightGradient: Matrix<Double>,
        inputGradient: Matrix<Double>,
    ) {
        guard
            let input = lastInput
        else {
            fatalError(
                "Weight backward called before forward.",
            )
        }

        // X × W
        //
        // dW = Xᵀ × dOutput
        let queryWeightGradient =
            input.transposed *
            queryGradient

        let keyWeightGradient =
            input.transposed *
            keyGradient

        let valueWeightGradient =
            input.transposed *
            valueGradient

        // dX = dOutput × Wᵀ
        let queryInputGradient =
            queryGradient *
            queryWeights.transposed

        let keyInputGradient =
            keyGradient *
            keyWeights.transposed

        let valueInputGradient =
            valueGradient *
            valueWeights.transposed

        // All three branches originate from
        // the same input.
        let inputGradient =
            queryInputGradient
                + keyInputGradient
                + valueInputGradient

        return (
            queryWeightGradient,
            keyWeightGradient,
            valueWeightGradient,
            inputGradient,
        )
    }

    // MARK: - Init

    public init(dimension: Int) {
        queryWeights =
            randomWeights(
                rows: dimension,
                columns: dimension,
            )

        keyWeights =
            randomWeights(
                rows: dimension,
                columns: dimension,
            )

        valueWeights =
            randomWeights(
                rows: dimension,
                columns: dimension,
            )
    }
}
