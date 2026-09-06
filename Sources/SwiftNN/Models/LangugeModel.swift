//
//  LangugeModel.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 5/9/2026.
//

import Foundation

public enum LanguageErrors: Error {
    case unsupportedData
    case unknownToken(String)
}

public struct SequenceModel<TokenizerType: Tokenizer>: Codable {
    public var transformer: Transformer
    public var vocabulary: Vocabulary<TokenizerType.Token>
    public let tokenizer: TokenizerType
    public let learningRate: Double

    public init(
        transformer: Transformer,
        vocabulary: Vocabulary<TokenizerType.Token>,
        tokenizer: TokenizerType,
        learningRate: Double
    ) {
        self.transformer = transformer
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.learningRate = learningRate
    }

    public mutating func generate(
        from input: TokenizerType.Input,
        maxTokens: Int
    ) throws -> TokenizerType.Input {
        guard maxTokens > 0 else {
            return input
        }

        var tokens = tokenizer.tokenize(input)

        for _ in 0 ..< maxTokens {
            var inputIDs: [Double] = []

            for token in tokens {
                guard let id = vocabulary.id(for: token) else {
                    throw LanguageErrors.unknownToken(
                        String(describing: token)
                    )
                }

                inputIDs.append(Double(id))
            }

            guard !inputIDs.isEmpty else {
                break
            }

            let input = Matrix<Double>(
                rows: inputIDs.count,
                columns: 1,
                grid: inputIDs
            )

            let prediction = transformer.forward(input)
            let lastRow = prediction.rows - 1
            let logits = prediction[lastRow]

            guard let nextTokenID = logits.indices.max(by: {
                logits[$0] < logits[$1]
            }) else {
                break
            }

            if let endTokenID = vocabulary.endTokenID,
               nextTokenID == endTokenID
            {
                break
            }

            guard let nextToken = vocabulary.token(for: nextTokenID) else {
                break
            }

            tokens.append(nextToken)
        }

        return tokenizer.detokenize(tokens)
    }

    public mutating func train(
        on examples: [(input: TokenizerType.Input, target: TokenizerType.Input)],
        epochs: Int
    ) {
        guard epochs > 0 else {
            return
        }

        guard let endTokenID = vocabulary.endTokenID else {
            return
        }

        for epoch in 0 ..< epochs {
            var totalLoss = 0.0
            var tokenCount = 0

            for example in examples.shuffled() {
                let inputTokens = tokenizer.tokenize(example.input)
                let targetTokens = tokenizer.tokenize(example.target)

                var sequenceIDs: [Double] = []

                for token in inputTokens {
                    guard let id = vocabulary.id(for: token) else {
                        continue
                    }

                    sequenceIDs.append(Double(id))
                }

                guard !sequenceIDs.isEmpty else {
                    continue
                }

                var targetIDs: [Int] = []

                for token in targetTokens {
                    guard let id = vocabulary.id(for: token) else {
                        continue
                    }

                    targetIDs.append(id)
                }

                guard !targetIDs.isEmpty else {
                    continue
                }

                for targetID in targetIDs {
                    let input = Matrix<Double>(
                        rows: sequenceIDs.count,
                        columns: 1,
                        grid: sequenceIDs
                    )

                    let vocabularySize = transformer.vocabularySize

                    var target = Matrix<Double>(
                        rows: 1,
                        columns: vocabularySize,
                        grid: Array(
                            repeating: 0.0,
                            count: vocabularySize
                        )
                    )

                    target[0, targetID] = 1.0

                    let loss = transformer.trainStep(
                        input: input,
                        target: target,
                        learningRate: learningRate
                    )

                    totalLoss += loss
                    tokenCount += 1

                    sequenceIDs.append(Double(targetID))
                }

                let input = Matrix<Double>(
                    rows: sequenceIDs.count,
                    columns: 1,
                    grid: sequenceIDs
                )

                let vocabularySize = transformer.vocabularySize

                var endTarget = Matrix<Double>(
                    rows: 1,
                    columns: vocabularySize,
                    grid: Array(
                        repeating: 0.0,
                        count: vocabularySize
                    )
                )

                endTarget[0, endTokenID] = 1.0

                let endLoss = transformer.trainStep(
                    input: input,
                    target: endTarget,
                    learningRate: learningRate
                )

                totalLoss += endLoss
                tokenCount += 1
            }

            let averageLoss = tokenCount > 0
                ? totalLoss / Double(tokenCount)
                : 0.0

            print(
                "Epoch \(epoch) Loss: \(totalLoss) Average Loss: \(averageLoss)"
            )
        }
    }

    public func export() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let data = try encoder.encode(self)

        guard let json = String(data: data, encoding: .utf8) else {
            throw LanguageErrors.unsupportedData
        }

        return json
    }

    public static func `import`(from json: String) throws -> SequenceModel {
        guard let data = json.data(using: .utf8) else {
            throw LanguageErrors.unsupportedData
        }

        return try JSONDecoder().decode(
            SequenceModel.self,
            from: data
        )
    }
}

public typealias LanguageModel = SequenceModel<WhitespaceTokenizer>

public extension SequenceModel where TokenizerType == WhitespaceTokenizer {
    init(
        transformer: Transformer,
        vocabulary: TextVocabulary,
        learningRate: Double
    ) {
        self.init(
            transformer: transformer,
            vocabulary: vocabulary,
            tokenizer: WhitespaceTokenizer(),
            learningRate: learningRate
        )
    }
}
