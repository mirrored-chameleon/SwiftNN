//
//  Tokenizers.swift
//  SwiftNN Language
//

import Foundation

public struct WhitespaceTokenizer: Tokenizer, Codable {
    public init() {}

    public func tokenize(_ input: String) -> [String] {
        input
            .split(separator: " ")
            .map(String.init)
    }

    public func detokenize(_ tokens: [String]) -> String {
        tokens.joined(separator: " ")
    }
}
