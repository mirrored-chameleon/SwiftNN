//
//  Talent.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 22/08/2026.
//

// MARK: - Talent Protocol

// A protocl that has 2 functions of `Encode` and `Decode` so that the model can take in different inputs and output different outputs.
//
// There are 2 generics of `Input` and `Output` so that the model can take in different types you teach it, so it could be a `String` or an `Int` for instance.

public protocol Talent {
    // Tokens or the possible classification for a type could be
    //
    // Tokens are used for the possible outcomes of a prediction. Usually models return an array of predictions, and the highest value is the most likely output according to the model. We can then find out what value it thinks is the most likely outcome by looking up it's index in the tokens variable.

    var tokens: [Output] { get set }

    // Input type for the model
    //
    // Generic type for `Talent` so that the `Network` can take in any input that you want it to.

    associatedtype Input

    // Output type for the `Network`
    //
    // Generic type for `Talent` so that the `Network` can output any output you want it to.

    associatedtype Output

    // `encode` function inside of `Talent`
    //
    // The function is used so that the `Network` is able to encode any type you want it to.

    func encode(_ input: Input) -> Matrix<Double>

    // `decode` function inside of `Talent`
    //
    // The function is used so that the `Network` is able to turn the output matrix back into something usable.

    func decode(_ output: Matrix<Double>) -> Output
}
