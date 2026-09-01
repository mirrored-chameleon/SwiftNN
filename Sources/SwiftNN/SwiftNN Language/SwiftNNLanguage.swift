import Foundation
import SwiftNN

@main
final class SwiftNNLanguageTest12 {

    static func main() {

        print("=== SwiftNN Language: Test 12 ===")
        print()
        print("REAL CHATBOT TEST")
        print()

        let tokens = [
            "<PAD>",
            "<UNK>",
            "<EOS>",

            "hello",
            "hi",
            "there",

            "how",
            "are",
            "you",
            "i",
            "am",
            "good",
            "great",
            "fine",

            "what",
            "is",
            "your",
            "name",
            "who",
            "call",
            "me",
            "swiftnn",

            "do",
            "you",
            "like",
            "coding",
            "yes",
            "no",

            "what",
            "do",
            "you",
            "like",
            "doing",

            "swift",
            "a",
            "programming",
            "language",

            "machine",
            "learning",
            "library",

            "can",
            "code",
            "help",
            "with",
            "coding",
            "yes",

            "are",
            "robot",
            "tiny",
            "model",

            "thank",
            "thanks",
            "welcome",

            "bye",
            "goodbye",
            "see",
            "later",

            "what",
            "are",
            "you",
            "building",
            "working",
            "on",

            "a",
            "neural",
            "network",
            "transformer",

            "cool",
            "that",
            "sounds",
            "interesting",

            "why",
            "was",
            "made",
            "to",
            "build",
            "models",

            "i",
            "like",
            "science",
            "and",
            "technology"
        ]

        let vocabulary = Vocabulary(
            tokens: tokens,
            dims: 128
        )

        var transformer = Transformer(
            vocabulary: vocabulary,
            modelDimension: 128,
            hiddenSize: 256,
            numberOfBlocks: 2
        )

        /*
         Each conversation is represented as a chain.

         For example:

         hello
         hello → hi
         hello hi → <EOS>

         This allows the current TrainingExample system
         to learn multi-token responses without changing
         the underlying trainer yet.
        */

        let conversations: [[String]] = [

            ["hello", "hi", "<EOS>"],
            ["hello", "there", "hi", "<EOS>"],
            ["hi", "hello", "<EOS>"],

            ["how", "are", "you", "i", "am", "good", "<EOS>"],
            ["how", "are", "you", "doing", "i", "am", "great", "<EOS>"],
            ["are", "you", "okay", "i", "am", "fine", "<EOS>"],

            ["what", "is", "your", "name", "i", "am", "swiftnn", "<EOS>"],
            ["who", "are", "you", "i", "am", "swiftnn", "<EOS>"],
            ["what", "should", "i", "call", "you", "you", "can", "call", "me", "swiftnn", "<EOS>"],

            ["what", "do", "you", "like", "i", "like", "coding", "<EOS>"],
            ["what", "do", "you", "like", "doing", "i", "like", "coding", "<EOS>"],
            ["do", "you", "like", "coding", "yes", "i", "like", "coding", "<EOS>"],

            ["what", "is", "swift", "swift", "is", "a", "programming", "language", "<EOS>"],
            ["what", "is", "swiftnn", "swiftnn", "is", "a", "machine", "learning", "library", "<EOS>"],

            ["can", "you", "code", "yes", "i", "can", "code", "<EOS>"],
            ["can", "you", "help", "with", "coding", "yes", "i", "can", "help", "<EOS>"],

            ["are", "you", "a", "robot", "i", "am", "a", "tiny", "model", "<EOS>"],
            ["what", "are", "you", "i", "am", "a", "tiny", "model", "<EOS>"],

            ["what", "are", "you", "building", "i", "am", "building", "a", "neural", "network", "<EOS>"],
            ["what", "are", "you", "working", "on", "i", "am", "working", "on", "a", "transformer", "<EOS>"],

            ["why", "was", "swiftnn", "made", "to", "build", "models", "<EOS>"],

            ["thank", "you", "welcome", "<EOS>"],
            ["thanks", "you", "are", "welcome", "<EOS>"],

            ["bye", "goodbye", "<EOS>"],
            ["goodbye", "see", "you", "later", "<EOS>"],

            ["that", "sounds", "cool", "yes", "it", "is", "interesting", "<EOS>"],
            ["that", "sounds", "interesting", "thank", "you", "<EOS>"],

            ["i", "like", "science", "and", "technology", "i", "like", "coding", "<EOS>"]
        ]

        print("Training conversations: \(conversations.count)")
        print()

        var trainingExamples: [TrainingExample] = []

        for conversation in conversations {

            guard conversation.count >= 2 else {
                continue
            }

            for index in 1..<conversation.count {

                let inputTokens = Array(conversation[0..<index])
                let targetToken = conversation[index]

                guard let targetID = vocabulary.id(for: targetToken) else {
                    continue
                }

                var inputIDs: [Double] = []

                for token in inputTokens {

                    if let id = vocabulary.id(for: token) {
                        inputIDs.append(Double(id))
                    }
                }

                guard !inputIDs.isEmpty else {
                    continue
                }

                let input = Matrix<Double>(
                    rows: inputIDs.count,
                    columns: 1,
                    grid: inputIDs
                )

                var targetGrid = Array(
                    repeating: 0.0,
                    count: tokens.count
                )

                targetGrid[targetID] = 1.0

                let target = Matrix<Double>(
                    rows: 1,
                    columns: tokens.count,
                    grid: targetGrid
                )

                trainingExamples.append(
                    TrainingExample(
                        input: input,
                        target: target
                    )
                )
            }
        }

        print("Training examples: \(trainingExamples.count)")
        print()
        print("Training...")
        print()

        var trainer = Trainer(
            learningRate: 0.001,
            epochs: 250
        )

        trainer.train(
            examples: trainingExamples
        ) { example, learningRate in

            transformer.trainStep(
                input: example.input,
                target: example.target,
                learningRate: learningRate
            )
        }

        print()
        print("=== Conversation Tests ===")
        print()

        let prompts = [
            "hello",
            "how are you",
            "what is your name",
            "who are you",
            "what do you like",
            "do you like coding",
            "what is swift",
            "what is swiftnn",
            "can you code",
            "are you a robot",
            "what are you",
            "what are you building",
            "what are you working on",
            "thank you",
            "bye"
        ]

        for prompt in prompts {

            let generated = generate(
                prompt: prompt,
                vocabulary: vocabulary,
                transformer: &transformer,
                maximumNewTokens: 10
            )

            print("Prompt:    \(prompt)")
            print("Generated: \(generated)")
            print()
        }

        print("=== Test 12 complete ===")
    }

    static func generate(
        prompt: String,
        vocabulary: Vocabulary,
        transformer: inout Transformer,
        maximumNewTokens: Int
    ) -> String {

        var words = prompt
            .split(separator: " ")
            .map(String.init)

        for _ in 0..<maximumNewTokens {

            var inputIDs: [Double] = []

            for word in words {

                guard let id = vocabulary.id(for: word) else {
                    continue
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

            guard let bestIndex = logits.indices.max(
                by: {
                    logits[$0] < logits[$1]
                }
            ) else {
                break
            }

            guard let nextToken = vocabulary.token(
                for: bestIndex
            ) else {
                break
            }

            if nextToken == "<EOS>" {
                break
            }

            if nextToken == "<PAD>" || nextToken == "<UNK>" {
                break
            }

            words.append(nextToken)
        }

        return words.joined(separator: " ")
    }
}