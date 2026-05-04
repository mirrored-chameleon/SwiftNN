// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import SwiftNN

final class TokenStore {
    
    var forwardTokens: [Int: String] = [
        0: "<UNK>",
        1: "<END>",
        2: "<START>"
    ]
    
    var backwardTokens: [String: Int] = [
        "<UNK>": 0,
        "<END>": 1,
        "<START>": 2
    ]
    
    var embeddings: [Int: Matrix<Double>] = [:]
    let embeddingWidth = 50
    
    var allTokens: [String] {
        forwardTokens
            .sorted { leftEntry, rightEntry in
                leftEntry.key < rightEntry.key
            }
            .map { entry in
                entry.value
            }
    }
    
    func tokenize(_ text: String) -> [Int] {
        
        let splitTokens = text
            .lowercased()
            .split(separator: " ")
            .map { substring in
                String(substring)
            }
        
        var tokenIdentifiers: [Int] = []
        
        for token in splitTokens {
            if let existingIdentifier = backwardTokens[token] {
                tokenIdentifiers.append(existingIdentifier)
            } else {
                let newIdentifier = forwardTokens.count
                forwardTokens[newIdentifier] = token
                backwardTokens[token] = newIdentifier
                tokenIdentifiers.append(newIdentifier)
            }
        }
        
        return tokenIdentifiers
    }
    
    func embedding(for tokenIdentifier: Int) -> Matrix<Double> {
        
        if let existingEmbedding = embeddings[tokenIdentifier] {
            return existingEmbedding
        }
        
        let newEmbedding = Matrix(
            rows: 1,
            columns: embeddingWidth,
            grid: (0..<embeddingWidth).map { _ in
                Double.random(in: -0.1...0.1)
            }
        )
        
        embeddings[tokenIdentifier] = newEmbedding
        
        return newEmbedding
    }
    
    func token(for tokenIdentifier: Int) -> String {
        forwardTokens[tokenIdentifier] ?? "<UNK>"
    }
}

final class EnglishTalent: Talent {
    
    typealias Input = String
    typealias Output = String
    
    var tokens: [String]
    
    private let tokenStore: TokenStore
    
    init(tokenStore: TokenStore = TokenStore()) {
        self.tokenStore = tokenStore
        self.tokens = tokenStore.allTokens
    }
    
    func encode(_ input: String) -> Matrix<Double> {
        
        let tokenIdentifiers = tokenStore.tokenize(input)
        
        tokens = tokenStore.allTokens
        
        let tokenEmbeddings = tokenIdentifiers.map { tokenIdentifier in
            tokenStore.embedding(for: tokenIdentifier)
        }
        
        return tokenEmbeddings.reduce(
            Matrix(rows: 1, columns: 50, grid: Array(repeating: 0.0, count: 50))
        ) { partialResult, currentEmbedding in
            partialResult + currentEmbedding
        }
    }
    
    func decode(_ output: Matrix<Double>) -> String {
        
        let outputValues = flatten(output)
        
        guard !outputValues.isEmpty else {
            return "<UNK>"
        }
        
        let predictedTokenIdentifier = argmax(outputValues)
        
        return tokenStore.token(for: predictedTokenIdentifier)
    }
}

struct SimpleNetwork: Network {
    var talent: any Talent

    var weights: [[Matrix<Double>]]
    var bias: [Matrix<Double>]

    var learningRate: Double = 0.01

    var tokensStore: TokenStore


    mutating func predict(input: Matrix<Double>) -> Matrix<Double> {
        var currentOutput = input

        for layerIndex in 0..<weights.count {
            for weightMatrix in weights[layerIndex] {
                currentOutput = currentOutput * weightMatrix + bias[layerIndex]
                
                if layerIndex < weights.count - 1 {
                    currentOutput = reluMatrix(currentOutput)
                }
            }
        }
        
        return Matrix<Double>(rows: currentOutput.rows, columns: currentOutput.columns, grid: softmax(currentOutput.grid))
    }

    mutating func train(input: Matrix<Double>, target: Matrix<Double>) {
        let prediction  = predict(input: input)
        let error = prediction - target

        for layer in 0..<weights.count {
            for weight in 0..<weights[layer].count {
                let gradientScale = learningRate * error[0, weight]
                let expandedInput = Matrix<Double>(
                    rows: weights[layer][weight].rows,
                    columns: weights[layer][weight].columns,
                    grid: Array(repeating: input.grid, count: weights[layer][weight].rows)
                        .flatMap { row in row }
                )
                weights[layer][weight] = weights[layer][weight] - gradientScale * expandedInput
            }
        }

        for biasIndex in 0..<bias.count {
            bias[biasIndex] = bias[biasIndex] - learningRate * error
        }
    }

    init(_ layers: [Int], talent: Talent, tokensStore: TokenStore) {
        self.weights = []
        self.bias = []

        var previousSize = 50
        for layer in layers {
            weights.append([Matrix<Double>(
                rows: previousSize,
                columns: layer,
                grid: Array(repeating: Double.random(in: -1...1), count: previousSize * layer)
            )])
            bias.append(Matrix<Double>(
                rows: 1,
                columns: layer,
                grid: Array(repeating: 0.0, count: layer)
            ))
            previousSize = layer
        }

        self.tokensStore = tokensStore
        self.talent = talent
    }
}

@main
struct SwiftNNTest {
    static func main() {
        
    }
}
