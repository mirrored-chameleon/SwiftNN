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

    // MARK: - Generation

    public mutating func generate(
        from input: TokenizerType.Input,
        maxTokens: Int
    ) throws -> TokenizerType.Input {
        guard maxTokens > 0 else {
            return input
        }

        var tokens = tokenizer.tokenize(input)

          if let separator = tokenizer.conversationSeparator,
              tokens.last != separator,
              let separatorID = vocabulary.id(for: separator),
              let separatorToken = vocabulary.token(for: separatorID)
        {
                tokens.append(separatorToken)
        }

        for _ in 0 ..< maxTokens {
            var inputIDs: [Double] = []

            for token in tokens {
                let id: Int
                if let knownID = vocabulary.id(for: token) {
                    id = knownID
                } else if let unknownToken = tokenizer.unknownToken,
                          let unknownID = vocabulary.id(for: unknownToken)
                {
                    id = unknownID
                } else {
                    throw LanguageErrors.unknownToken(String(describing: token))
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
            var logits = prediction[lastRow]

            if tokens.count >= 3,
               tokens[tokens.count - 1] == tokens[tokens.count - 2],
               tokens[tokens.count - 2] == tokens[tokens.count - 3],
               let repeatedTokenID = vocabulary.id(
                   for: tokens[tokens.count - 1]
               )
            {
                logits[repeatedTokenID] = -Double.infinity
            }

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

    // MARK: - Training

    public mutating func train(
        on examples: [
            (
                input: TokenizerType.Input,
                target: TokenizerType.Input
            )
        ],
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
                    let id = vocabulary.id(for: token)
                        ?? tokenizer.unknownToken.flatMap(vocabulary.id)

                    guard let id else { continue }

                    sequenceIDs.append(Double(id))
                }

                     if let separator = tokenizer.conversationSeparator,
                         inputTokens.last != separator,
                         let separatorID = vocabulary.id(for: separator)
                {
                    sequenceIDs.append(Double(separatorID))
                }

                guard !sequenceIDs.isEmpty else {
                    continue
                }

                var targetIDs: [Int] = []

                for token in targetTokens {
                    let id = vocabulary.id(for: token)
                        ?? tokenizer.unknownToken.flatMap(vocabulary.id)

                    guard let id else { continue }

                    targetIDs.append(id)
                }

                guard !targetIDs.isEmpty else {
                    continue
                }

                // Include the target sequence in the model input so that
                // each target position can predict the following token.
                sequenceIDs.append(
                    contentsOf: targetIDs.map(Double.init)
                )

                // The model predicts each target token, followed by
                // the end token.
                var targets = targetIDs
                targets.append(endTokenID)

                let input = Matrix<Double>(
                    rows: sequenceIDs.count,
                    columns: 1,
                    grid: sequenceIDs
                )

                // Train the complete sequence in one forward/backward pass
                // instead of running a separate trainStep for every token.
                let loss = transformer.trainSequence(
                    input: input,
                    targets: targets,
                    learningRate: learningRate
                )

                totalLoss += loss * Double(targets.count)
                tokenCount += targets.count
            }

            let averageLoss = tokenCount > 0
                ? totalLoss / Double(tokenCount)
                : 0.0

            print(
                "Epoch \(epoch) Loss: \(totalLoss) " +
                "Average Loss: \(averageLoss)"
            )
        }
    }

    // MARK: - Export

    public func export() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [
            .prettyPrinted,
            .sortedKeys
        ]

        let data = try encoder.encode(self)

        guard let json = String(
            data: data,
            encoding: .utf8
        ) else {
            throw LanguageErrors.unsupportedData
        }

        return json
    }

    // MARK: - Import

    public static func `import`(
        from json: String
    ) throws -> SequenceModel {
        guard let data = json.data(
            using: .utf8
        ) else {
            throw LanguageErrors.unsupportedData
        }

        return try JSONDecoder().decode(
            SequenceModel.self,
            from: data
        )
    }
}

public typealias LanguageModel = SequenceModel<CharacterTokenizer>
public typealias WordLanguageModel = SequenceModel<WhitespaceTokenizer>

public extension SequenceModel where TokenizerType == CharacterTokenizer {
    init(
        transformer: Transformer,
        vocabulary: TextVocabulary,
        learningRate: Double
    ) {
        self.init(
            transformer: transformer,
            vocabulary: vocabulary,
            tokenizer: CharacterTokenizer(),
            learningRate: learningRate
        )
    }
}