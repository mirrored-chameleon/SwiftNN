//
// BubbleCLI.swift
// SwiftNNApp
//
// Created by Davyn Monagle on 23/3/2026.
//


import Foundation
import Surge
import SwiftNN

// MARK: - GLOBAL STATE

var conversationMemory: [String] = []
var storedFacts: [String: String] = [:]

// MARK: - FACT STORAGE

func saveFactsToDisk() {
    if let data = try? JSONEncoder().encode(storedFacts) {
        try? data.write(to: URL(fileURLWithPath: "facts.json"))
    }
}

func loadFactsFromDisk() {
    if let data = try? Data(contentsOf: URL(fileURLWithPath: "facts.json")),
       let decoded = try? JSONDecoder().decode([String: String].self, from: data) {
        storedFacts = decoded
    }
}

// MARK: - MATRIX HELPERS


func softmaxValues(_ values: [Double]) -> [Double] {
    let maxValue = values.max() ?? 0
    let exponentials = values.map { exp($0 - maxValue) }
    let sum = exponentials.reduce(0, +)
    return exponentials.map { $0 / sum }
}

// MARK: - SAMPLING

func sampleIndex(from probabilities: [Double]) -> Int {

    let temperature: Double = 0.85

    let adjusted = probabilities.map { pow($0, 1.0 / temperature) }
    let total = adjusted.reduce(0, +)

    if total == 0 {
        return Int.random(in: 0..<probabilities.count)
    }

    let normalized = adjusted.map { $0 / total }

    let randomValue = Double.random(in: 0...1)
    var cumulative = 0.0

    for (index, value) in normalized.enumerated() {
        cumulative += value
        if randomValue < cumulative {
            return index
        }
    }

    return normalized.count - 1
}

// MARK: - TOKENIZER

struct Tokenizer {
    static func tokenize(_ text: String) -> [String] {
        let pattern = #"[a-zA-Z]+|[.!?]"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let nsText = text.lowercased() as NSString

        return regex.matches(in: nsText as String,
                             range: NSRange(location: 0, length: nsText.length))
            .map { nsText.substring(with: $0.range) }
    }
}

// MARK: - TALENT (FIXED)

struct TextTalent: Talent {

    // ✅ REQUIRED
    typealias Input = String
    typealias Output = String

    var tokens: [String]
    private var tokenIndexLookup: [String: Int]

    init(corpus: [String]) {
        let allTokens = corpus.flatMap { Tokenizer.tokenize($0) }
        tokens = Array(Set(allTokens)).sorted()

        tokenIndexLookup = Dictionary(
            uniqueKeysWithValues:
                tokens.enumerated().map { ($1, $0) }
        )
    }

    // ✅ REQUIRED BY PROTOCOL
    func encode(_ input: String) -> Matrix<Double> {
        return encodeTokens(Tokenizer.tokenize(input))
    }

    // ✅ REQUIRED BY PROTOCOL
    func decode(_ output: Matrix<Double>) -> String {
        let values = flattenMatrix(output)
        let index = sampleIndex(from: values)
        return tokens[index]
    }

    // CUSTOM (USED INTERNALLY)

    func encodeTokens(_ inputTokens: [String]) -> Matrix<Double> {

        let vocabularySize = tokens.count
        var grid: [Double] = []

        for token in inputTokens {
            var vector = Array(repeating: 0.0, count: vocabularySize)

            if let index = tokenIndexLookup[token] {
                vector[index] = 1.0
            }

            grid += vector
        }

        return Matrix(rows: vocabularySize,
                      columns: inputTokens.count,
                      grid: grid)
    }

    func decodeWithContext(_ output: Matrix<Double>,
                           context: [String]) -> String {

        var values = flattenMatrix(output)

        for token in context.suffix(5) {
            if let index = tokenIndexLookup[token] {
                values[index] *= 0.3
            }
        }

        let index = sampleIndex(from: values)
        return tokens[index]
    }

    func indexOfToken(_ token: String) -> Int {
        return tokenIndexLookup[token] ?? 0
    }
}

// MARK: - MODEL

struct BubbleModel: Network {

    var talent: Talent
    var weights: [[Matrix<Double>]] = []
    var bias: [Matrix<Double>] = []

    let vocabularySize: Int
    let hiddenSize: Int = 32
    let learningRate: Double = 0.002

    var embeddingMatrix: Matrix<Double>
    var outputWeightMatrix: Matrix<Double>

    init(talent: TextTalent) {

        self.talent = talent
        self.vocabularySize = talent.tokens.count

        func randomMatrix(rows: Int, columns: Int) -> Matrix<Double> {
            Matrix(rows: rows,
                   columns: columns,
                   grid: (0..<rows * columns).map {
                       _ in Double.random(in: -0.5...0.5)
                   })
        }

        embeddingMatrix = randomMatrix(rows: hiddenSize,
                                       columns: vocabularySize)

        outputWeightMatrix = randomMatrix(rows: vocabularySize,
                                          columns: hiddenSize)
    }

    mutating func predict(input: Matrix<Double>) -> Matrix<Double> {

        let embedded = embeddingMatrix * input
        let lastColumnIndex = max(0, input.columns - 1)

        let hiddenVector = Matrix(
            rows: hiddenSize,
            columns: 1,
            grid: (0..<hiddenSize).map {
                embedded[$0, lastColumnIndex]
            }
        )

        let logits = outputWeightMatrix * hiddenVector
        let probabilities = softmaxValues(flattenMatrix(logits))

        return Matrix(rows: vocabularySize,
                      columns: 1,
                      grid: probabilities)
    }

    mutating func train(input: Matrix<Double>, target: Matrix<Double>) {

        let embedded = embeddingMatrix * input
        let lastColumnIndex = max(0, input.columns - 1)

        let hiddenVector = Matrix(
            rows: hiddenSize,
            columns: 1,
            grid: (0..<hiddenSize).map {
                embedded[$0, lastColumnIndex]
            }
        )

        let logits = outputWeightMatrix * hiddenVector
        let probabilities = softmaxValues(flattenMatrix(logits))

        var error = probabilities
        let targetValues = flattenMatrix(target)

        for i in 0..<error.count {
            error[i] -= targetValues[i]
        }

        for vocabularyIndex in 0..<vocabularySize {
            for hiddenIndex in 0..<hiddenSize {
                outputWeightMatrix[vocabularyIndex, hiddenIndex] -=
                    learningRate * error[vocabularyIndex] * hiddenVector[hiddenIndex, 0]
            }
        }
    }
}

// MARK: - DATASET

func loadDatasetFromFile(path: String) -> [(String, String)] {

    guard let text = try? String(contentsOfFile: path) else {
        print("❌ Failed to load dataset")
        return []
    }

    var dataset: [(String, String)] = []

    for rawLine in text.components(separatedBy: "\n") {

        let line = rawLine.trimmingCharacters(in: .whitespaces)
        if line.isEmpty { continue }

        let parts = line.components(separatedBy: "->")

        if parts.count == 2 {
            let input = parts[0].trimmingCharacters(in: .whitespaces)
            let output = parts[1].trimmingCharacters(in: .whitespaces)

            dataset.append((input, output))
        }
    }

    print("Loaded dataset:", dataset.count)
    return dataset
}

// MARK: - GENERATION

func generateResponse(model: inout BubbleModel,
                      talent: TextTalent,
                      input: String) -> String {

    let lower = input.lowercased()

    // 🧠 SYSTEM PROMPT (lightweight)
    let systemPrompt = "bubble chaotic energetic silly weird creature"

    // 👤 MEMORY
    if lower.contains("my name is") {
        let words = Tokenizer.tokenize(input)
        if let name = words.last {
            storedFacts["name"] = name
            saveFactsToDisk()
            return "hehe hi \(name) !!"
        }
    }

    if lower.contains("what is my name") {
        if let name = storedFacts["name"] {
            return "your name is \(name) !! hehe"
        }
    }

    // 💬 INPUT-AWARE RESPONSES (VERY IMPORTANT)

    if lower.contains("hi") || lower.contains("hello") {
        return ["HELLO!!", "hi hi hi !!", "BOING hello !!", "hehe hello!!"]
            .randomElement()!
    }

    if lower.contains("okay") {
        return ["okay?? okay??", "that sounds suspicious...", "hmmmmm okay!!"]
            .randomElement()!
    }

    if lower.contains("?") {
        return ["I DO NOT KNOW!!", "maybe?? probably?? definitely not!!",
                "that is a VERY concerning question!!"]
            .randomElement()!
    }

    // 🧠 MODEL INPUT
    var tokens =
        Tokenizer.tokenize(systemPrompt) +
        Array(conversationMemory.suffix(6)) +
        Tokenizer.tokenize(input)

    var generatedTokens: [String] = []

    for _ in 0..<10 {

        let inputMatrix = talent.encodeTokens(tokens)
        let prediction = model.predict(input: inputMatrix)

        let nextToken = talent.decodeWithContext(prediction, context: tokens)

        if generatedTokens.suffix(2).allSatisfy({ $0 == nextToken }) {
            continue
        }

        generatedTokens.append(nextToken)
        tokens.append(nextToken)
    }

    let words = generatedTokens.filter {
        $0.count > 1 && ![".", "!", "?"].contains($0)
    }

    let fallbackOptions = [
        ["this", "is", "very", "strange"],
        ["something", "feels", "wrong"],
        ["i", "do", "not", "like", "this"],
        ["this", "is", "suspicious"]
    ]

    let coreWords: [String]

    if words.isEmpty {
        // ONLY fallback if completely empty
        coreWords = fallbackOptions.randomElement()!
    } else {
        // trust the model more
        coreWords = Array(words.prefix(6))
    }

    // 🧠 GRAMMAR + STRUCTURE
    let subject = ["i", "this", "that", "it"].randomElement()!

    let verb = subject == "i"
        ? ["am", "feel", "think"].randomElement()!
        : ["is", "feels", "seems"].randomElement()!

    var sentence = "\(subject) \(verb) " + coreWords.joined(separator: " ")

    // 🫧 BUBBLE CHAOS LAYER
    let noises = ["hehe", "BOING", "pop", "WHEEEE"]

    if Double.random(in: 0...1) < 0.5 {
        sentence += " " + noises.randomElement()!
    }

    // make it more expressive
    let endings = ["!", "!!", "!!!", "?!"]
    sentence += endings.randomElement()!

    // 💾 MEMORY UPDATE
    conversationMemory += Tokenizer.tokenize(input)
    conversationMemory += Tokenizer.tokenize(sentence)

    if conversationMemory.count > 50 {
        conversationMemory.removeFirst(25)
    }

    return sentence
}

// MARK: - MAIN

/* loadFactsFromDisk()

let datasetPath = "/Users/davyn/Library/Mobile Documents/com~apple~CloudDocs/Development/SwiftNN/SwiftNNApp/bubble_data.txt"

let dataset = loadDatasetFromFile(path: datasetPath)

let corpus = dataset.flatMap { [$0.0, $0.1] }

let talent = TextTalent(corpus: corpus)
var model = BubbleModel(talent: talent)
 print("Training...")

for epoch in 0..<100 {
    for (input, output) in dataset {
        let inputMatrix = talent.encode(input)
        let targetMatrix = talent.encode(output)
        model.train(input: inputMatrix, target: targetMatrix)
    }
    print("Epoch \(epoch) complete")
}

print("🫧 Bubble ready! (type quit to exit)\n")

while true {
    print("You:", terminator: " ")
    guard let input = readLine(), input != "quit" else { break }

    let response = generateResponse(model: &model,
                                    talent: talent,
                                    input: input)

    print("Bubble:", response)
} */
