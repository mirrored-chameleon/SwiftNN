//
//  Tokenizer.swift
//  SwiftNN
//

public protocol Tokenizer: Codable {
    associatedtype Input
    associatedtype Token: Hashable & Codable

    func tokenize(_ input: Input) -> [Token]

    func detokenize(_ tokens: [Token]) -> Input
}
