//
//  Vocabulary.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 21/07/2026.
//

// MARK: - Embeddings

// Handles storing the embedding matrix.
import Foundation

public struct EmbeddingLayer: Codable {
    public var embeddings: Matrix<Double>

    public init(vocabularySize: Int, dims: Int) {
        embeddings = randomWeights(
            rows: vocabularySize,
            columns: dims,
        )
    }
}

// MARK: - Vocabulary

/// Stores the vocabulary and handles encoding/decoding tokens.
public struct Vocabulary<Token: Hashable & Codable>: Codable {
    public private(set) var tokenToId: [Token: Int]
    public private(set) var idToToken: [Token]
    public private(set) var endTokenID: Int?

    public var count: Int {
        idToToken.count
    }

    public enum CodingKeys: String, CodingKey {
        case idToToken
        case endTokenID
    }

    public init(tokens: [Token], endToken: Token? = nil) {
        tokenToId = [:]
        idToToken = []

        for (id, token) in tokens.enumerated() {
            tokenToId[token] = id
            idToToken.append(token)
        }

        if let endToken {
            endTokenID = tokenToId[endToken]
        } else {
            endTokenID = nil
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(
            keyedBy: CodingKeys.self,
        )

        idToToken = try container.decode(
            [Token].self,
            forKey: .idToToken,
        )

        endTokenID = try container.decodeIfPresent(
            Int.self,
            forKey: .endTokenID,
        )

        tokenToId = [:]

        for (id, token) in idToToken.enumerated() {
            tokenToId[token] = id
        }
    }

    /// Returns the ID of a token.
    public func id(for token: Token) -> Int? {
        tokenToId[token]
    }

    /// Returns the token for an ID.
    public func token(for id: Int) -> Token? {
        guard id >= 0, id < idToToken.count else {
            return nil
        }

        return idToToken[id]
    }
}

public typealias TextVocabulary = Vocabulary<String>
