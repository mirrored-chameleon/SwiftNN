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

// MARK: - Lightweight Talent for testing

struct SimpleBinaryTalent: Talent {
    var tokens: [Int] = [0, 1] // possible outputs
    
    typealias Input = [Double]
    typealias Output = Int
    
    func encode(_ input: [Double]) -> Matrix<Double> {
        // Simple 1xN matrix
        return Matrix(rows: input.count, columns: 1, grid: input)
    }
    
    func decode(_ output: Matrix<Double>) -> Int {
        var totalSum: Double = 0.0
        
        for rowIndex in 0..<output.rows {
            for columnIndex in 0..<output.columns {
                totalSum += output[rowIndex, columnIndex]
            }
        }
        
        return totalSum > 0.5 ? 1 : 0
    }
}

// MARK: - Test Script

func runSimpleModelTest() {
    // Create talent
    let testTalent = SimpleBinaryTalent()
    
    // Define a tiny 2-input model: 1 output x 2 inputs
    var testModel = BasicModel(
        talent: testTalent,
        weights: Matrix(rows: 1, columns: 2, grid: [0.0, 0.0]),
        bias: Matrix(rows: 1, columns: 1, grid: [0.0]),
        learningRate: 0.1
    )
    
    // Training dataset: input -> target
    let trainingInputs: [[Double]] = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    let trainingTargets: [Double] = [
        0, // 0 OR 0 = 0
        1, // 0 OR 1 = 1
        1, // 1 OR 0 = 1
        1  // 1 OR 1 = 1
    ]
    
    // Convert to matrices
    let inputMatrices = trainingInputs.map { testTalent.encode($0) }
    let targetMatrices = trainingTargets.map { Matrix(rows: 1, columns: 1, grid: [$0]) }
    
    // Train model for 500 epochs
    for _ in 0..<500 {
        for (inputMatrix, targetMatrix) in zip(inputMatrices, targetMatrices) {
            testModel.train(input: inputMatrix, target: targetMatrix)
        }
    }
    
    // Test the model
    print("Testing trained model:")
    for (input, inputMatrix) in zip(trainingInputs, inputMatrices) {
        let predictedMatrix = testModel.predict(input: inputMatrix)
        let predictedValue = testTalent.decode(predictedMatrix)
        print("Input: \(input) -> Prediction: \(predictedValue)")
    }
}


struct SwiftNNTests {

    @Test func example() async throws {
        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
        runSimpleModelTest()
    }

}
