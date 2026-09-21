//
//  Tokenizers.swift
//  SwiftNN Language
//

import Foundation

public struct WhitespaceTokenizer: Tokenizer, Codable {
    public init() {}

    public var conversationSeparator: String? { "|" }
    public var unknownToken: String? { "<unk>" }

    public func tokenize(_ input: String) -> [String] {
        input
            .split(separator: " ")
            .map(String.init)
    }

    public func detokenize(_ tokens: [String]) -> String {
        tokens.joined(separator: " ")
    }
}

public struct CharacterTokenizer: Tokenizer, Codable {
    public init() {}

    public var conversationSeparator: String? { "|" }
    public var unknownToken: String? { "<unk>" }

    public func tokenize(_ input: String) -> [String] {
        input.map(String.init)
    }

    public func detokenize(_ tokens: [String]) -> String {
        tokens.joined()
    }
}