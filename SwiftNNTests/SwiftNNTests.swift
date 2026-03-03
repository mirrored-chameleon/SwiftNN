//
//  SwiftNNTests.swift
//  SwiftNNTests
//
//  Created by Davyn Monagle on 1/3/2026.
//

import Testing
import Surge
@testable import SwiftNN
import Foundation

struct EnglishVowelTalent: Talent {

    var tokens: [Int] = [0, 1] // 0 = consonant, 1 = vowel

    typealias Input = Character
    typealias Output = Int
    
    // Encode letter as one-hot vector
    func encode(_ input: Character) -> Matrix<Double> {
        let letters = Array("abcdefghijklmnopqrstuvwxyz")
        var vector = Array(repeating: 0.0, count: letters.count)
        if let index = letters.firstIndex(of: Character(input.lowercased())) {
            vector[index] = 1.0
        }
        return Matrix(rows: vector.count, columns: 1, grid: vector)
    }
    
    // Decode output: >0.5 = vowel, else consonant
    internal func decode(_ output: Matrix<Double>) -> Int {
        let value = output[0, 0]
        return value > 0.5 ? 1 : 0
    }
}

func testEnglishVowelModel() {
    
    let talent = EnglishVowelTalent()
    
    // Network: 26 input → 8 hidden → 1 output
    var model = BasicModel(layers: [26, 8, 1], talent: talent)
    
    // Training data
    let letters = Array("abcdefghijklmnopqrstuvwxyz")
    let targets: [Double] = letters.map { "aeiou".contains($0) ? 1.0 : 0.0 }
    
    // Train
    for _ in 0..<500 {
        for (letter, target) in zip(letters, targets) {
            model.train(
                input: talent.encode(letter),
                target: Matrix(rows: 1, columns: 1, grid: [target])
            )
        }
    }
    
    // Test
    for letter in letters {
        let predictionMatrix = model.predict(input: talent.encode(letter))
        let prediction = talent.decode(predictionMatrix)
        print("Letter: \(letter) → Prediction: \(prediction == 1 ? "Vowel" : "Consonant")")
    }
}

struct SwiftNNTests {

    @Test func example() async throws {
        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
        testEnglishVowelModel()
    }

}
