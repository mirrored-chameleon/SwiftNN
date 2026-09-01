//
//  Talents.swift
//  SwiftNN Language
//
//  Created by Davyn Monagle on 23/07/2026.
//

import SwiftNN

public struct EnglishTalent: Talent {

    public typealias Input = String
    public typealias Output = String

    public var tokens: [String] = []

    public var vocabulary: Vocabulary

    public init(vocabulary: Vocabulary) {
        self.vocabulary = vocabulary
        self.tokens = vocabulary.idToToken
    }

    public func encode(_ input: String) -> Matrix<Double> {

        let words = input.lowercased().split(separator: " ").map(String.init)

        var ids: [Double] = []

        for word in words {

            if let id = vocabulary.tokenToId[word] {
                ids.append(Double(id))
            }
        }

        return Matrix(
            rows: ids.count,
            columns: 1,
            grid: ids
        )
    }

    public func decode(_ output: Matrix<Double>) -> String {

        let ids = flatten(output).map { Int($0.rounded()) }

        var words: [String] = []

        for id in ids {

            if let token = vocabulary.token(for: id) {
                words.append(token)
            }

        }

        return words.joined(separator: " ")
    }
}