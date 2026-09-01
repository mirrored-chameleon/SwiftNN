//
//  TransformerBlock.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 14/08/2026.
//

import Foundation
import SwiftNN

struct TransformerBlock {

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

    // MARK: - Forward

    mutating func forward(
        _ input: Matrix<Double>
    ) -> Matrix<Double> {

        let attentionOutput =
            attention.forward(input)

        // Residual connection:
        //
        // input + attention(input)
        let attentionResidual =
            input + attentionOutput

        let feedForwardOutput =
            feedForward.forward(
                attentionResidual
            )

        // Second residual connection:
        //
        // attentionResidual + feedForward(attentionResidual)
        let output =
            attentionResidual + feedForwardOutput

        return output
    }

    // MARK: - Backward

    mutating func backward(
        _ gradient: Matrix<Double>,
        learningRate: Double
    ) -> Matrix<Double> {

        // --------------------------------------------------
        // Feed-forward backward
        // --------------------------------------------------

        let feedForwardOutput =
            feedForward.backwardOutput(
                gradient
            )

        let feedForwardInput =
            feedForward.backwardInput(
                feedForwardOutput.hiddenGradient
            )

        // Update feed-forward weights
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

        // --------------------------------------------------
        // First residual connection
        // --------------------------------------------------
        //
        // attentionResidual =
        //     input + attentionOutput
        //
        // Therefore the gradient entering the attention
        // branch is the gradient from the feed-forward
        // branch.

        let attentionGradient =
            gradient + feedForwardInput.inputGradient

        // --------------------------------------------------
        // Attention backward
        // --------------------------------------------------

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

        // Update attention weights
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

        // --------------------------------------------------
        // First residual connection
        // --------------------------------------------------
        //
        // attentionResidual =
        //     input + attentionOutput
        //
        // One gradient comes directly through the residual.
        // Another comes through attention.

        let inputGradient =
            attentionGradient
            + attentionWeights.inputGradient

        return inputGradient
    }
}