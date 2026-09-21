import Foundation
import SwiftNN

private struct EncodedConversation {
    let input: Matrix<Double>
    let targets: [Int]
}

private func encode(
    _ conversation: (input: String, reply: String),
    vocabulary: TextVocabulary,
) -> EncodedConversation {
    let inputIDs = conversation.input.compactMap {
        vocabulary.id(for: String($0))
    }
    let targetIDs = conversation.reply.compactMap {
        vocabulary.id(for: String($0))
    }

    return EncodedConversation(
        input: Matrix(
            rows: inputIDs.count + targetIDs.count,
            columns: 1,
            grid: (inputIDs + targetIDs).map(Double.init),
        ),
        targets: targetIDs + [vocabulary.endTokenID!],
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

struct StressHarness {
    static func main() throws {
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
            Set(conversations.flatMap { $0.input + $0.reply }),
        ).sorted().map(String.init)
        let vocabulary = TextVocabulary(
            tokens: ["<end>"] + characters,
            endToken: "<end>",
        )
        let examples = conversations.map {
            encode($0, vocabulary: vocabulary)
        }
        var transformer = Transformer(
            vocabulary: vocabulary,
            modelDimension: 24,
            hiddenSize: 48,
            numberOfBlocks: 2,
        )
        let initialLoss = averageLoss(for: examples, transformer: &transformer)
        let trainingStart = Date()

        for _ in 0 ..< 20 {
            for example in examples.shuffled() {
                _ = transformer.trainSequence(
                    input: example.input,
                    targets: example.targets,
                    learningRate: 0.002,
                )
            }
        }

        let trainingDuration = Date().timeIntervalSince(trainingStart)
        let finalLoss = averageLoss(for: examples, transformer: &transformer)
        var model = LanguageModel(
            transformer: transformer,
            vocabulary: vocabulary,
            learningRate: 0.002,
        )
        let generationStart = Date()
        let response = try model.generate(from: "hello there", maxTokens: 32)
        let generationDuration = Date().timeIntervalSince(generationStart)

        print("initial loss: \(initialLoss)")
        print("final loss: \(finalLoss)")
        print("training seconds: \(trainingDuration)")
        print("generation seconds: \(generationDuration)")
        print("response: \(response)")
    }
}
