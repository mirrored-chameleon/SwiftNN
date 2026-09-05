//
//  TransformerBlock.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 14/08/2026.
//

import Foundation

struct TransformerBlock: Codable {
    var attention: SelfAttention
    var feedForward: FeedForward

    init(
        dimension: Int,
        hiddenSize: Int
    ) {
        attention = SelfAttention(
            dimension: dimension
        )

        feedForward = FeedForward(
            inputSize: dimension,
            hiddenSize: hiddenSize
        )
    }

    enum CodingKeys: String, CodingKey {
        case attention
        case feedForward
    }

    // MARK: - Forward

    mutating func forward(
        _ input: Matrix<Double>
    ) -> Matrix<Double> {
        let attentionOutput = attention.forward(input)

        let attentionResidual =
            input + attentionOutput

        let feedForwardOutput =
            feedForward.forward(
                attentionResidual
            )

        return attentionResidual + feedForwardOutput
    }

    // MARK: - Backward

    mutating func backward(
        _ gradient: Matrix<Double>,
        learningRate: Double
    ) -> Matrix<Double> {
        let feedForwardOutput =
            feedForward.backwardOutput(
                gradient
            )

        let feedForwardInput =
            feedForward.backwardInput(
                feedForwardOutput.hiddenGradient
            )

        subtractScaled(
            &feedForward.outputWeights,
            gradient:
                feedForwardOutput.weightGradient,
            learningRate:
                learningRate
        )

        subtractScaled(
            &feedForward.outputBias,
            gradient:
                feedForwardOutput.biasGradient,
            learningRate:
                learningRate
        )

        subtractScaled(
            &feedForward.inputWeights,
            gradient:
                feedForwardInput.weightGradient,
            learningRate:
                learningRate
        )

        subtractScaled(
            &feedForward.inputBias,
            gradient:
                feedForwardInput.biasGradient,
            learningRate:
                learningRate
        )

        let attentionGradient =
            gradient + feedForwardInput.inputGradient

        let attentionOutputGradient =
            attention.backwardOutput(
                attentionGradient
            )

        let attentionScoresGradient =
            attention.softmaxBackward(
                attentionOutputGradient.attentionGradient
            )

        let attentionQueriesKeys =
            attention.backwardScores(
                attentionScoresGradient
            )

        let attentionWeights =
            attention.backwardWeights(
                queryGradient:
                    attentionQueriesKeys.queryGradient,
                keyGradient:
                    attentionQueriesKeys.keyGradient,
                valueGradient:
                    attentionOutputGradient.valueGradient
            )

        subtractScaled(
            &attention.queryWeights,
            gradient:
                attentionWeights.queryWeightGradient,
            learningRate:
                learningRate
        )

        subtractScaled(
            &attention.keyWeights,
            gradient:
                attentionWeights.keyWeightGradient,
            learningRate:
                learningRate
        )

        subtractScaled(
            &attention.valueWeights,
            gradient:
                attentionWeights.valueWeightGradient,
            learningRate:
                learningRate
        )

        return attentionGradient
            + attentionWeights.inputGradient
    }
}
