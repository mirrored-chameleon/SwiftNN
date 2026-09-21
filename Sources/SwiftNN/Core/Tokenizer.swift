//
//  Tokenizer.swift
//  SwiftNN
//

public protocol Tokenizer: Codable {
    associatedtype Input
    associatedtype Token: Hashable & Codable

    var conversationSeparator: Token? { get }
    var unknownToken: Token? { get }

    func tokenize(_ input: Input) -> [Token]

    func detokenize(_ tokens: [Token]) -> Input
}

public extension Tokenizer {
    var conversationSeparator: Token? {
        nil
    }

    var unknownToken: Token? {
        nil
    }
}
