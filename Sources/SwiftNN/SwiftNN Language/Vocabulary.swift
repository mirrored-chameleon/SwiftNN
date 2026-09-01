//
//  Vocabulary.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 21/07/2026.
//

// MARK: - Embeddings

/// Handles storing the embedding matrix.
import SwiftNN

// MARK: - Vocabulary

/// Stores the vocabulary and handles encoding/decoding tokens.
struct EmbeddingLayer {

    var embeddings: Matrix<Double>

    init(vocabularySize: Int, dims: Int) {
        embeddings = randomWeights(
            rows: vocabularySize,
            columns: dims
        )
    }
}
public struct Vocabulary {

    var tokenToId: [String: Int]
    var idToToken: [String]
    var embeddings: EmbeddingLayer

    init(tokens: [String], dims: Int) {

        tokenToId = [:]
        idToToken = []

        embeddings = EmbeddingLayer(
            vocabularySize: tokens.count,
            dims: dims
        )

        for (id, token) in tokens.enumerated() {
            tokenToId[token] = id
            idToToken.append(token)
        }
    }

    /// Returns the ID of a token.
    func id(for token: String) -> Int? {
        tokenToId[token]
    }

    /// Returns the embedding vector for a token.
    func embedding(for token: String) -> [Double]? {

        guard let id = tokenToId[token] else {
            return nil
        }

        return embeddings.embeddings[id]
    }

    /// Returns the token for an ID.
    func token(for id: Int) -> String? {

        guard id >= 0 && id < idToToken.count else {
            return nil
        }

        return idToToken[id]
    }
}
