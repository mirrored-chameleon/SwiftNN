import XCTest
@testable import SwiftNN

final class LanguageModelTrainingTests: XCTestCase {
    func testSequenceTrainingStartsAtFinalInputToken() {
        let vocabulary = TextVocabulary(
            tokens: ["<end>", "a", "b"],
            endToken: "<end>",
        )
        var transformer = Transformer(
            vocabulary: vocabulary,
            modelDimension: 1,
            hiddenSize: 1,
            numberOfBlocks: 0,
        )
        transformer.embeddings.embeddings = Matrix(
            rows: 3,
            columns: 1,
            grid: [0.0, 0.0, 0.0],
        )
        transformer.outputProjection.weights = Matrix(
            rows: 1,
            columns: 3,
            grid: [0.0, 1.0, -1.0],
        )
        transformer.outputProjection.bias = Matrix(
            rows: 1,
            columns: 3,
            grid: [0.0, 0.0, 0.0],
        )

        let loss = transformer.trainSequence(
            input: Matrix(
                rows: 3,
                columns: 1,
                grid: [1.0, 2.0, 1.0],
            ),
            targets: [1, 0],
            learningRate: 0.0,
        )

        let expectedLoss = (
            negativeLogProbability(at: 1, target: 1)
                + negativeLogProbability(at: 2, target: 0)
        ) / 2.0
        XCTAssertEqual(loss, expectedLoss, accuracy: 1e-12)
    }

    func testCharacterSequenceTrainingKeepsLossAndParametersFinite() throws {
        let characters = Array("abcdefghijklmnopqrstuvwxyz ").map(String.init)
        let vocabulary = TextVocabulary(
            tokens: ["<end>"] + characters,
            endToken: "<end>",
        )
        let prefix = "swift "
        let continuation = "neural networks learn"
        let prefixIDs = prefix.compactMap { vocabulary.id(for: String($0)) }
        let continuationIDs = continuation.compactMap {
            vocabulary.id(for: String($0))
        }
        let input = Matrix<Double>(
            rows: prefixIDs.count + continuationIDs.count,
            columns: 1,
            grid: (prefixIDs + continuationIDs).map(Double.init),
        )
        let targets = continuationIDs + [try XCTUnwrap(vocabulary.endTokenID)]
        var transformer = Transformer(
            vocabulary: vocabulary,
            modelDimension: 8,
            hiddenSize: 16,
            numberOfBlocks: 1,
        )

        for step in 0 ..< 250 {
            let loss = transformer.trainSequence(
                input: input,
                targets: targets,
                learningRate: 0.001,
            )

            XCTAssertTrue(loss.isFinite, "Loss was non-finite at step \(step).")
            XCTAssertTrue(
                hasFiniteParameters(transformer),
                "A parameter was non-finite at step \(step).",
            )
        }
    }

    func testCharacterLanguageModelGeneratesTrainedReply() throws {
        let characters = Array("abcdefghijklmnopqrstuvwxyz ").map(String.init)
        let vocabulary = TextVocabulary(
            tokens: ["<end>"] + characters,
            endToken: "<end>",
        )
        let conversations = [
            (input: "hello", reply: "hi"),
            (input: "how are you", reply: "i am well"),
            (input: "what is your name", reply: "swift nn"),
            (input: "goodbye", reply: "see you"),
        ]
        var transformer = Transformer(
            vocabulary: vocabulary,
            modelDimension: 16,
            hiddenSize: 32,
            numberOfBlocks: 1,
        )

        for _ in 0 ..< 400 {
            for conversation in conversations {
                let inputIDs = conversation.input.compactMap {
                    vocabulary.id(for: String($0))
                }
                let targetIDs = conversation.reply.compactMap {
                    vocabulary.id(for: String($0))
                }
                let input = Matrix<Double>(
                    rows: inputIDs.count + targetIDs.count,
                    columns: 1,
                    grid: (inputIDs + targetIDs).map(Double.init),
                )
                let targets = targetIDs + [try XCTUnwrap(vocabulary.endTokenID)]
                let loss = transformer.trainSequence(
                    input: input,
                    targets: targets,
                    learningRate: 0.001,
                )

                XCTAssertTrue(loss.isFinite)
                XCTAssertTrue(hasFiniteParameters(transformer))
            }
        }

        var model = LanguageModel(
            transformer: transformer,
            vocabulary: vocabulary,
            learningRate: 0.001,
        )
        let generated = try model.generate(from: "hello", maxTokens: 10)

        XCTAssertEqual(generated, "hellohi")
    }

    private func negativeLogProbability(at position: Int, target: Int) -> Double {
        let value = sin(Double(position))
        let logits = [0.0, value, -value]
        let normalizer = logits.map(exp).reduce(0.0, +)
        return -log(exp(logits[target]) / normalizer)
    }

    private func hasFiniteParameters(_ transformer: Transformer) -> Bool {
        hasFiniteValues(transformer.embeddings.embeddings)
            && hasFiniteValues(transformer.outputProjection.weights)
            && hasFiniteValues(transformer.outputProjection.bias)
            && transformer.blocks.allSatisfy { block in
                hasFiniteValues(block.attention.queryWeights)
                    && hasFiniteValues(block.attention.keyWeights)
                    && hasFiniteValues(block.attention.valueWeights)
                    && hasFiniteValues(block.feedForward.inputWeights)
                    && hasFiniteValues(block.feedForward.inputBias)
                    && hasFiniteValues(block.feedForward.outputWeights)
                    && hasFiniteValues(block.feedForward.outputBias)
            }
    }

    private func hasFiniteValues(_ matrix: Matrix<Double>) -> Bool {
        matrix.grid.allSatisfy { $0.isFinite }
    }
}
