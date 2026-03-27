//
// BubbleCLI.swift
// SwiftNNApp
//
// Created by Davyn Monagle on 23/3/2026.
//


import Foundation
import Surge
import SwiftNN

// MARK: - Global Memory

var memory: [String] = []
var facts: [String: String] = [:]

// MARK: - Save / Load Facts

func saveFacts() {
    if let data = try? JSONEncoder().encode(facts) {
        try? data.write(to: URL(fileURLWithPath: "facts.json"))
    }
}

func loadFacts() {
    if let data = try? Data(contentsOf: URL(fileURLWithPath: "facts.json")),
       let decoded = try? JSONDecoder().decode([String: String].self, from: data) {
        facts = decoded
    }
}

// MARK: - Math

func flatten(_ matrix: Matrix<Double>) -> [Double] {
    var result: [Double] = []
    for r in 0..<matrix.rows {
        for c in 0..<matrix.columns {
            result.append(matrix[r, c])
        }
    }
    return result
}

func softmax(_ values: [Double]) -> [Double] {
    let maxValue = values.max() ?? 0
    let exps = values.map { exp($0 - maxValue) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}

// MARK: - Sampling (FIXED + STABLE)

func sample(_ values: [Double]) -> Int {

    let temperature = Double.random(in: 0.7...0.95)

    let adjusted = values.map { pow($0, 1.0 / temperature) }
    let sum = adjusted.reduce(0, +)

    if sum == 0 { return Int.random(in: 0..<values.count) }

    let normalized = adjusted.map { $0 / sum }

    let random = Double.random(in: 0...1)
    var cumulative = 0.0

    for (i, v) in normalized.enumerated() {
        cumulative += v
        if random < cumulative {
            return i
        }
    }

    return normalized.count - 1
}

// MARK: - Tokenizer

struct Tokenizer {
    static func tokenize(_ text: String) -> [String] {
        let pattern = #"[a-zA-Z]+|[.!?]"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let ns = text.lowercased() as NSString

        return regex.matches(in: ns as String,
                             range: NSRange(location: 0, length: ns.length))
            .map { ns.substring(with: $0.range) }
    }
}

// MARK: - Talent

struct TextTalent: Talent {
    typealias Input = String
    
    typealias Output = String
    

    var tokens: [String]
    private var map: [String: Int]

    init(corpus: [String]) {
        let all = corpus.flatMap { Tokenizer.tokenize($0) }
        tokens = Array(Set(all)).sorted()
        map = Dictionary(uniqueKeysWithValues:
            tokens.enumerated().map { ($1, $0) })
    }

    func encodeTokens(_ tokensInput: [String]) -> Matrix<Double> {

        let vocab = tokens.count
        var grid: [Double] = []

        for token in tokensInput {
            var vector = Array(repeating: 0.0, count: vocab)

            if let index = map[token] {
                vector[index] = 1.0
            }

            grid += vector
        }

        return Matrix(rows: vocab,
                      columns: tokensInput.count,
                      grid: grid)
    }

    func decode(_ output: Matrix<Double>,
                context: [String]) -> String {

        var values = flatten(output)

        // Penalise repetition
        for token in context.suffix(5) {
            let index = map[token] ?? 0
            values[index] *= 0.3
        }

        let index = sample(values)
        return tokens[index]
    }

    func index(of token: String) -> Int {
        return map[token] ?? 0
    }
}

// MARK: - Model

struct BubbleModel: Network {

    var talent: Talent
    var weights: [[Matrix<Double>]] = []
    var bias: [Matrix<Double>] = []

    let vocabSize: Int
    let modelSize: Int = 64
    let learningRate: Double = 0.003

    var embedding: Matrix<Double>
    var outputWeights: Matrix<Double>

    init(talent: TextTalent) {

        self.talent = talent
        self.vocabSize = talent.tokens.count

        func randomMatrix(_ r: Int, _ c: Int) -> Matrix<Double> {
            Matrix(rows: r, columns: c,
                   grid: (0..<r*c).map { _ in Double.random(in: -0.5...0.5) })
        }

        embedding = randomMatrix(modelSize, vocabSize)
        outputWeights = randomMatrix(vocabSize, modelSize)
    }

    mutating func predict(input: Matrix<Double>) -> Matrix<Double> {

        if input.columns == 0 {
            return Matrix(rows: vocabSize, columns: 1,
                          grid: Array(repeating: 1.0 / Double(vocabSize),
                                      count: vocabSize))
        }

        let embedded = embedding * input

        let lastIndex = max(0, input.columns - 1)

        let last = Matrix(
            rows: modelSize,
            columns: 1,
            grid: (0..<modelSize).map {
                embedded[$0, lastIndex]
            }
        )

        let logits = outputWeights * last
        let probs = softmax(flatten(logits))

        return Matrix(rows: vocabSize, columns: 1, grid: probs)
    }

    mutating func train(input: Matrix<Double>, target: Matrix<Double>) {

        let prediction = predict(input: input)

        let p = flatten(prediction)
        let t = flatten(target)

        var error = p

        for i in 0..<error.count {
            error[i] -= t[i]
        }

        for v in 0..<vocabSize {
            for h in 0..<modelSize {
                outputWeights[v, h] -= learningRate * error[v]
            }
        }
    }
}

// MARK: - Dataset

func loadDataset(path: String) -> [(String, String)] {

    guard let text = try? String(contentsOfFile: path) else {
        print("❌ Could not load dataset")
        return []
    }

    var data: [(String, String)] = []

    for raw in text.components(separatedBy: "\n") {

        let line = raw.trimmingCharacters(in: .whitespaces)

        if line.isEmpty { continue }

        let parts = line.components(separatedBy: "->")

        if parts.count == 2 {
            data.append((
                parts[0].trimmingCharacters(in: .whitespaces),
                parts[1].trimmingCharacters(in: .whitespaces)
            ))
        }
    }

    print("Loaded dataset:", data.count)
    return data
}

// MARK: - Generate

func generate(model: inout BubbleModel,
              talent: TextTalent,
              input: String) -> String {

    let lower = input.lowercased()

    // FACT MEMORY
    if lower.contains("my name is") {
        let words = Tokenizer.tokenize(input)
        if let name = words.last {
            facts["name"] = name
            saveFacts()
            return "hehe hi \(name) !"
        }
    }

    if lower.contains("what is my name") {
        if let name = facts["name"] {
            return "your name is \(name) !"
        }
    }

    if lower.contains("i like") {
        let words = Tokenizer.tokenize(input)
        if let item = words.last {
            facts["likes"] = item
            saveFacts()
            return "ooh you like \(item) !"
        }
    }

    if lower.contains("what do i like") {
        if let like = facts["likes"] {
            return "you like \(like) !"
        }
    }

    var tokens = Tokenizer.tokenize(input)

    for _ in 0..<15 {

        let inputMatrix = talent.encodeTokens(tokens)
        let prediction = model.predict(input: inputMatrix)

        var next = talent.decode(prediction, context: tokens)

        if Double.random(in: 0...1) < 0.15 {
            next = ["hehe", "boing", "pop", "wheee"].randomElement()!
        }

        tokens.append(next)

        if [".", "!", "?"].contains(next) {
            break
        }
    }

    let reply = tokens.suffix(10).joined(separator: " ")

    memory += Tokenizer.tokenize(input)
    memory += Tokenizer.tokenize(reply)

    if memory.count > 50 {
        memory.removeFirst(25)
    }

    return reply
}

// MARK: - MAIN

loadFacts()

let dataset = loadDataset(path: "./bubble_data.txt")

if dataset.isEmpty {
    print("Dataset empty!")
    exit(1)
}

let corpus = dataset.flatMap { [$0.0, $0.1] }

let talent = TextTalent(corpus: corpus)
var model = BubbleModel(talent: talent)

print("Training...")

for epoch in 0..<150 {

    var totalLoss = 0.0
    var count = 0

    for (input, output) in dataset.shuffled() {

        let inputTokens = Tokenizer.tokenize(input)
        let outputTokens = Tokenizer.tokenize(output)

        for i in 0..<outputTokens.count {

            let context = inputTokens + Array(outputTokens.prefix(i))
            let x = talent.encodeTokens(context)

            var target = Array(repeating: 0.0, count: talent.tokens.count)
            let index = talent.index(of: outputTokens[i])
            target[index] = 1.0

            let y = Matrix(rows: talent.tokens.count,
                           columns: 1,
                           grid: target)

            let prediction = model.predict(input: x)
            let p = flatten(prediction)

            totalLoss += -log(max(p[index], 1e-9))
            count += 1

            model.train(input: x, target: y)
        }
    }

    print("Epoch \(epoch):", totalLoss / Double(count))
}

print("🫧 Bubble ready! (type quit)\n")

while true {
    print("You:", terminator: " ")
    guard let input = readLine(), input != "quit" else { break }

    let response = generate(model: &model,
                            talent: talent,
                            input: input)

    print("Bubble:", response)
}
