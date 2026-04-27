// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import SwiftNN

final class TokenStore {
    
    private var forwardTokens: [Int: String] = [
        0: "<UNK>",
        1: "<END>",
        2: "<START>"
    ]
    
    private var backwardTokens: [String: Int] = [
        "<UNK>": 0,
        "<END>": 1,
        "<START>": 2
    ]
    
    private var embeddings: [Int: Matrix<Double>] = [:]
    private let embeddingWidth = 50
    
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
            grid: Array(repeating: 0.0, count: embeddingWidth)
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



@main
struct SwiftNNTest {
    static func main() {
        
    }
}
