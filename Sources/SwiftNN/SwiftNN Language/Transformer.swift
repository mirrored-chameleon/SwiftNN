//
//  Transformer.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 14/08/2026.
//

import Foundation

public struct Transformer: Codable {
    public let modelDimension: Int

    public let vocabularySize: Int

    public var embeddings: EmbeddingLayer
    public var positionalEncoding: SinusoidalPositionalEncoding
    public var blocks: [TransformerBlock]
    public var outputProjection: OutputProjection
    
    public enum CodingKeys: String, CodingKey {
        case modelDimension
        case vocabularySize
        case embeddings
        case positionalEncoding
        case blocks
        case outputProjection
    }

    public init<Token>(
        vocabulary: Vocabulary<Token>,
        modelDimension: Int,
        hiddenSize: Int,
        numberOfBlocks: Int,
    ) {
        self.init(
            vocabularySize: vocabulary.count,
            modelDimension: modelDimension,
            hiddenSize: hiddenSize,
            numberOfBlocks: numberOfBlocks,
        )
    }

    public init(
        vocabularySize: Int,
        modelDimension: Int,
        hiddenSize: Int,
        numberOfBlocks: Int,
    ) {
        precondition(
            modelDimension > 0,
            "Model dimension must be greater than zero.",
        )

        precondition(
            vocabularySize > 0,
            "Vocabulary size must be greater than zero.",
        )

        self.modelDimension = modelDimension
        self.vocabularySize = vocabularySize

        embeddings = EmbeddingLayer(
            vocabularySize: vocabularySize,
            dims: modelDimension,
        )

        positionalEncoding =
            SinusoidalPositionalEncoding()

        var blocks: [TransformerBlock] = []

        for _ in 0 ..< numberOfBlocks {
            blocks.append(
                TransformerBlock(
                    dimension: modelDimension,
                    hiddenSize: hiddenSize,
                ),
            )
        }

        self.blocks = blocks

        outputProjection =
            OutputProjection(
                weights:
                Matrix<Double>.random(
                    rows: modelDimension,
                    columns:
                    vocabularySize,
                ),
                bias:
                Matrix<Double>(
                    rows: 1,
                    columns:
                    vocabularySize,
                    grid:
                    Array(
                        repeating: 0.0,
                        count:
                        vocabularySize,
                    ),
                ),
            )
    }

    // MARK: - Forward

    public mutating func forward(
        _ input: Matrix<Double>,
    ) -> Matrix<Double> {
        var embeddedValues: [Double] = []

        for value in input.grid {
            let id = Int(value)

            guard
                id >= 0,
                id < embeddings.embeddings.rows
            else {
                continue
            }

            embeddedValues.append(
                contentsOf: embeddings.embeddings[id],
            )
        }

        precondition(
            embeddedValues.count ==
                input.rows * modelDimension,
            "Embedding output has incorrect dimensions.",
        )

        var output =
            Matrix(
                rows: input.rows,
                columns: modelDimension,
                grid: embeddedValues,
            )

        output =
            positionalEncoding.forward(
                output,
            )

        for index in blocks.indices {
            output =
                blocks[index].forward(
                    output,
                )
        }

        return outputProjection.forward(
            output,
        )
    }

    // MARK: - Training

    public mutating func trainStep(
        input: Matrix<Double>,
        target: Matrix<Double>,
        learningRate: Double,
    ) -> Double {
        let prediction = forward(input)

        let lastRow =
            prediction.rows - 1

        let logits =
            prediction[lastRow]

        // --------------------------------------------------
        // Softmax
        // --------------------------------------------------

        let maximum =
            logits.max() ?? 0.0

        let exponentials =
            logits.map {
                exp($0 - maximum)
            }

        let total =
            exponentials.reduce(
                0.0,
                +,
            )

        let probabilities =
            exponentials.map {
                $0 / total
            }

        // --------------------------------------------------
        // Cross entropy
        // --------------------------------------------------

        var loss = 0.0

        for column in 0 ..< prediction.columns {
            let targetValue =
                target[0, column]

            if targetValue > 0.0 {
                loss -=
                    targetValue *
                    log(
                        max(
                            probabilities[column],
                            1e-12,
                        ),
                    )
            }
        }

        // --------------------------------------------------
        // Output gradient
        // --------------------------------------------------

        var fullGradient =
            Matrix<Double>(
                rows: prediction.rows,
                columns: prediction.columns,
                grid:
                Array(
                    repeating: 0.0,
                    count:
                    prediction.rows *
                        prediction.columns,
                ),
            )

        for column in 0 ..< prediction.columns {
            fullGradient[
                lastRow,
                column,
            ] =
                probabilities[column]
                    - target[0, column]
        }

        // --------------------------------------------------
        // Output projection
        // --------------------------------------------------

        let outputGradients =
            outputProjection.backward(
                fullGradient,
            )

        subtractScaled(
            &outputProjection.weights,
            gradient:
            outputGradients.weightGradient,
            learningRate:
            learningRate,
        )

        subtractScaled(
            &outputProjection.bias,
            gradient:
            outputGradients.biasGradient,
            learningRate:
            learningRate,
        )

        // --------------------------------------------------
        // Transformer blocks
        // --------------------------------------------------

        var blockGradient =
            outputGradients.inputGradient

        for index in blocks.indices.reversed() {
            blockGradient =
                blocks[index].backward(
                    blockGradient,
                    learningRate:
                    learningRate,
                )
        }

        // --------------------------------------------------
        // Embeddings
        // --------------------------------------------------

        updateEmbeddings(
            input: input,
            gradient: blockGradient,
            learningRate: learningRate,
        )

        return loss
    }

    // MARK: - Embedding Training

    private mutating func updateEmbeddings(
        input: Matrix<Double>,
        gradient: Matrix<Double>,
        learningRate: Double,
    ) {
        precondition(
            gradient.rows == input.rows,
            "Embedding gradient row count must match input.",
        )

        precondition(
            gradient.columns == modelDimension,
            "Embedding gradient dimension must match model dimension.",
        )

        for row in 0 ..< input.rows {
            let tokenID =
                Int(input[row, 0])

            guard
                tokenID >= 0,
                tokenID <
                embeddings.embeddings.rows
            else {
                continue
            }

            for column in 0 ..< modelDimension {
                embeddings
                    .embeddings[
                        tokenID,
                        column,
                    ] -=
                    learningRate *
                    gradient[
                        row,
                        column,
                    ]
            }
        }
    }
}
