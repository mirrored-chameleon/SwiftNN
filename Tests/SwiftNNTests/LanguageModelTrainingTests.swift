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
            tokens: ["<end>", "|"] + characters,
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
            modelDimension: 24,
            hiddenSize: 48,
            numberOfBlocks: 2,
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

    func testCharacterTransformerStressTrainingAndGeneration() throws {
        let conversations = [
            (input: "hello there", reply: "hello how can i help"),
            (input: "how are you", reply: "i am doing well"),
            (input: "what is your name", reply: "i am swiftnn"),
            (input: "what can you do", reply: "i can learn small text patterns"),
            (input: "tell me about swift", reply: "swift is a programming language"),
            (input: "what is a transformer", reply: "it predicts the next token"),
            (input: "good morning", reply: "good morning how are you"),
            (input: "goodbye", reply: "goodbye see you soon"),
            (input: "thank you", reply: "you are welcome"),
            (input: "can you help me", reply: "yes i will try to help"),
            (input: "what is two plus two", reply: "two plus two is four"),
            (input: "where are you", reply: "i run in a local program"),
        ]
        let characters = Array(
            Set(conversations.flatMap { $0.input + $0.reply })
        ).sorted().map(String.init)
        let vocabulary = TextVocabulary(
            tokens: ["<end>", "|"] + characters,
            endToken: "<end>",
        )
        let examples = try conversations.map { conversation in
            try encode(conversation, with: vocabulary)
        }
        var transformer = Transformer(
            vocabulary: vocabulary,
            modelDimension: 24,
            hiddenSize: 48,
            numberOfBlocks: 2,
        )
        let initialLoss = averageLoss(
            for: examples,
            transformer: &transformer,
        )
        let trainingStart = Date()

        for epoch in 0 ..< 20 {
            for example in examples.shuffled() {
                let loss = transformer.trainSequence(
                    input: example.input,
                    targets: example.targets,
                    learningRate: 0.002,
                )

                XCTAssertTrue(loss.isFinite, "Loss was non-finite in epoch \(epoch).")
            }

            XCTAssertTrue(
                hasFiniteParameters(transformer),
                "A parameter was non-finite after epoch \(epoch).",
            )
        }

        let trainingDuration = Date().timeIntervalSince(trainingStart)
        let finalLoss = averageLoss(
            for: examples,
            transformer: &transformer,
        )
        XCTAssertTrue(finalLoss.isFinite)
        XCTAssertLessThan(
            finalLoss,
            initialLoss,
            "Training loss did not improve under the character-token stress load.",
        )

        var model = LanguageModel(
            transformer: transformer,
            vocabulary: vocabulary,
            learningRate: 0.002,
        )
        let generationStart = Date()
        var generatedResponses: [String: String] = [:]

        for conversation in conversations {
            let prompt = conversation.input + "|"
            let generated = try model.generate(
                from: prompt,
                maxTokens: conversation.reply.count,
            )

            XCTAssertEqual(
                generated,
                prompt + conversation.reply,
                "Unexpected response for prompt: \(conversation.input)",
            )
            generatedResponses[conversation.input] = generated
        }

        let generationDuration = Date().timeIntervalSince(generationStart)

        let exported = try model.export()
        var restoredModel = try LanguageModel.import(from: exported)

        for conversation in conversations {
            let prompt = conversation.input + "|"
            let restoredGenerated = try restoredModel.generate(
                from: prompt,
                maxTokens: conversation.reply.count,
            )

            XCTAssertEqual(
                restoredGenerated,
                generatedResponses[conversation.input],
                "Imported model changed response for prompt: \(conversation.input)",
            )
        }

        print(
            "Character stress benchmark: \(examples.count) examples, " +
                "\(examples.reduce(0) { $0 + $1.targets.count }) target tokens per epoch, " +
                "loss \(initialLoss) -> \(finalLoss), " +
                "training \(trainingDuration)s, generation \(generationDuration)s, " +
                "responses: \(generatedResponses)",
        )
    }

    func testTinyModelLearnsAndRoundTrips() throws {
        let vocabulary = TextVocabulary(
            tokens: ["<end>", "a", "b"],
            endToken: "<end>",
        )
        let input = Matrix<Double>(
            rows: 2,
            columns: 1,
            grid: [1.0, 2.0],
        )
        let targets = [2, 0]
        var transformer = Transformer(
            vocabulary: vocabulary,
            modelDimension: 8,
            hiddenSize: 16,
            numberOfBlocks: 1,
        )

        for _ in 0 ..< 500 {
            _ = transformer.trainSequence(
                input: input,
                targets: targets,
                learningRate: 0.05,
            )
        }

        var model = LanguageModel(
            transformer: transformer,
            vocabulary: vocabulary,
            learningRate: 0.05,
        )
        let generated = try model.generate(from: "a", maxTokens: 1)
        let exported = try model.export()
        var restoredModel = try LanguageModel.import(from: exported)
        let restoredGenerated = try restoredModel.generate(from: "a", maxTokens: 1)

        print("Tiny model response: \(generated)")
        XCTAssertEqual(generated, "ab")
        XCTAssertEqual(restoredGenerated, generated)
    }

    func testTinyConversationalModelOnRealPhrases() throws {
        let conversations = [
            (input: "hello", reply: "hi"),
            (input: "how are you", reply: "good"),
        ]
        let characters = Array(
            Set(conversations.flatMap { $0.input + $0.reply })
        ).sorted().map(String.init)
        let vocabulary = TextVocabulary(
            tokens: ["<end>", "|"] + characters,
            endToken: "<end>",
        )
        let examples = try conversations.map { conversation in
            try encode(conversation, with: vocabulary)
        }
        var transformer = Transformer(
            vocabulary: vocabulary,
            modelDimension: 4,
            hiddenSize: 8,
            numberOfBlocks: 0,
        )

        for _ in 0 ..< 500 {
            for example in examples {
                _ = transformer.trainSequence(
                    input: example.input,
                    targets: example.targets,
                    learningRate: 0.05,
                )
            }
        }

        var model = LanguageModel(
            transformer: transformer,
            vocabulary: vocabulary,
            learningRate: 0.05,
        )
        var responses: [String: String] = [:]

        for conversation in conversations {
            let prompt = conversation.input + "|"
            let response = try model.generate(
                from: prompt,
                maxTokens: conversation.reply.count,
            )
            responses[conversation.input] = response
            XCTAssertEqual(
                response,
                prompt + conversation.reply,
                "Unexpected response for prompt: \(conversation.input)",
            )
        }

        let exported = try model.export()
        var restoredModel = try LanguageModel.import(from: exported)

        for conversation in conversations {
            let prompt = conversation.input + "|"
            let restoredResponse = try restoredModel.generate(
                from: prompt,
                maxTokens: conversation.reply.count,
            )
            XCTAssertEqual(restoredResponse, responses[conversation.input])
        }

        print("Conversational responses: \(responses)")
    }

    func testWordModelLearnsShortFactualSentences() throws {
        let facts = [
                (input: "what do bees make", target: "Bees make honey."),
                (input: "how many legs does a bee have", target: "A bee has six legs."),
        ]
        let tokens = Set(
            facts.flatMap { conversation in
                conversation.input.split(separator: " ").map(String.init)
                    + conversation.target.split(separator: " ").map(String.init)
            }
        )
        let vocabulary = Vocabulary<String>(
            tokens: ["<end>", "|"] + tokens.sorted(),
            endToken: "<end>",
        )
        var model = WordLanguageModel(
            transformer: Transformer(
                vocabulary: vocabulary,
                modelDimension: 8,
                hiddenSize: 16,
                numberOfBlocks: 1,
            ),
            vocabulary: vocabulary,
            tokenizer: WhitespaceTokenizer(),
            learningRate: 0.02,
        )

        model.train(on: facts, epochs: 200)

        for fact in facts {
            let response = try model.generate(
                from: fact.input,
                maxTokens: fact.target.split(separator: " ").count,
            )
            XCTAssertTrue(
                response.hasSuffix(fact.target),
                "Unexpected factual response: \(response)",
            )
        }

        let exported = try model.export()
        var restoredModel = try WordLanguageModel.import(from: exported)
        let restoredResponse = try restoredModel.generate(
            from: facts[0].input,
            maxTokens: 3,
        )
        XCTAssertEqual(restoredResponse, try model.generate(
            from: facts[0].input,
            maxTokens: 3,
        ))
        XCTAssertTrue(restoredResponse.contains("."))
    }

    func testWordModelGeneratesTwoSentenceFact() throws {
        let fact = (input: "where do bees live", target: "Bees live in hives. They work together.")
        let targetTokens = fact.target.split(separator: " ").map(String.init)
        let vocabulary = Vocabulary<String>(
            tokens: ["<end>", "|"] + fact.input.split(separator: " ").map(String.init)
                + targetTokens,
            endToken: "<end>",
        )
        var model = WordLanguageModel(
            transformer: Transformer(
                vocabulary: vocabulary,
                modelDimension: 8,
                hiddenSize: 16,
                numberOfBlocks: 1,
            ),
            vocabulary: vocabulary,
            tokenizer: WhitespaceTokenizer(),
            learningRate: 0.02,
        )

        model.train(on: [fact], epochs: 200)
        let response = try model.generate(
            from: fact.input,
            maxTokens: targetTokens.count,
        )

        print("Two-sentence response: \(response)")
        XCTAssertTrue(response.hasSuffix(fact.target))
    }

    func testUnknownTokensUseFallbackWhenAvailable() throws {
        let vocabulary = TextVocabulary(
            tokens: ["<end>", "<unk>", "|", "known"],
            endToken: "<end>",
        )
        var model = LanguageModel(
            transformer: Transformer(
                vocabulary: vocabulary,
                modelDimension: 4,
                hiddenSize: 8,
                numberOfBlocks: 0,
            ),
            vocabulary: vocabulary,
            learningRate: 0.01,
        )

        model.train(
            on: [(input: "mystery", target: "known")],
            epochs: 1,
        )

        let response = try model.generate(from: "mystery", maxTokens: 1)
        XCTAssertTrue(response.hasPrefix("mystery"))
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

    private func encode(
        _ conversation: (input: String, reply: String),
        with vocabulary: TextVocabulary,
    ) throws -> EncodedConversation {
        let inputIDs = try (conversation.input + "|").map {
            try XCTUnwrap(vocabulary.id(for: String($0)))
        }
        let targetIDs = try conversation.reply.map {
            try XCTUnwrap(vocabulary.id(for: String($0)))
        }
        let endTokenID = try XCTUnwrap(vocabulary.endTokenID)

        return EncodedConversation(
            input: Matrix(
                rows: inputIDs.count + targetIDs.count,
                columns: 1,
                grid: (inputIDs + targetIDs).map(Double.init),
            ),
            targets: targetIDs + [endTokenID],
        )
    }

    private func averageLoss(
        for examples: [EncodedConversation],
        transformer: inout Transformer,
    ) -> Double {
        var totalLoss = 0.0
        var tokenCount = 0

        for example in examples {
            let predictions = transformer.forward(example.input)
            let firstTargetRow = example.input.rows - example.targets.count

            for (offset, targetID) in example.targets.enumerated() {
                let logits = predictions[firstTargetRow + offset]
                let largestLogit = logits.max() ?? 0.0
                let logNormalizer = largestLogit + log(
                    logits.map { exp($0 - largestLogit) }.reduce(0.0, +),
                )
                totalLoss += logNormalizer - logits[targetID]
                tokenCount += 1
            }
        }

        return totalLoss / Double(tokenCount)
    }
}

private struct EncodedConversation {
    let input: Matrix<Double>
    let targets: [Int]
}
