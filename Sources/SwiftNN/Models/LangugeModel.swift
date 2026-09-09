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

public struct LanguageModel: Codable {
    public var transformer: Transformer
    public var vocabulary: Vocabulary
    public let learningRate: Double

    public init(
        transformer: Transformer,
        vocabulary: Vocabulary,
        learningRate: Double
    ) {
        self.transformer = transformer
        self.vocabulary = vocabulary
        self.learningRate = learningRate
    }

    // MARK: - Generation

    public mutating func generate(
        from input: String,
        maxTokens: Int
    ) throws -> String {
        guard maxTokens > 0 else {
            return input
        }

        var tokens = Array(input)

        for _ in 0 ..< maxTokens {
            var inputIDs: [Double] = []

            for token in tokens {
                let tokenString = String(token)

                guard let id = vocabulary.id(for: tokenString) else {
                    throw LanguageErrors.unknownToken(tokenString)
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

            guard let nextToken = vocabulary.token(
                for: nextTokenID
            ) else {
                break
            }

            guard let character = nextToken.first else {
                break
            }

            tokens.append(character)
        }

        return String(tokens)
    }

    // MARK: - Training

    public mutating func train(
        on examples: [
            (
                input: String,
                target: String
            )
        ],
        epochs: Int
    ) {
        guard epochs > 0 else {
            return
        }

        let endToken = "\u{0003}"

        guard let endTokenID = vocabulary.id(
            for: endToken
        ) else {
            return
        }

        for epoch in 0 ..< epochs {
            var totalLoss = 0.0
            var tokenCount = 0

            for example in examples.shuffled() {
                let inputCharacters = Array(example.input)
                let targetCharacters = Array(example.target)

                var inputIDs: [Double] = []

                for character in inputCharacters {
                    let token = String(character)

                    guard let id = vocabulary.id(
                        for: token
                    ) else {
                        continue
                    }

                    inputIDs.append(Double(id))
                }

                guard !inputIDs.isEmpty else {
                    continue
                }

                var targetIDs: [Int] = []

                for character in targetCharacters {
                    let token = String(character)

                    guard let id = vocabulary.id(
                        for: token
                    ) else {
                        continue
                    }

                    targetIDs.append(id)
                }

                guard !targetIDs.isEmpty else {
                    continue
                }

                var sequenceIDs = inputIDs

                sequenceIDs.append(
                    contentsOf: targetIDs.map(Double.init)
                )

                var targets = targetIDs
                targets.append(endTokenID)

                let input = Matrix<Double>(
                    rows: sequenceIDs.count,
                    columns: 1,
                    grid: sequenceIDs
                )

                let loss = transformer.trainSequence(
                    input: input,
                    targets: targets,
                    learningRate: learningRate
                )

                totalLoss += loss
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
    ) throws -> LanguageModel {
        guard let data = json.data(
            using: .utf8
        ) else {
            throw LanguageErrors.unsupportedData
        }

        return try JSONDecoder().decode(
            LanguageModel.self,
            from: data
        )
    }
}