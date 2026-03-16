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

struct XORTalent: Talent {
    
    typealias Input = (Double, Double)
    typealias Output = Double
    
    var tokens: [Double] = [0, 1]
    
    func encode(_ input: Input) -> Matrix<Double> {
        
        return Matrix(
            rows: 2,
            columns: 1,
            grid: [input.0, input.1]
        )
    }
    
    func decode(_ output: Matrix<Double>) -> Double {
        
        let values = flatten(output)
        
        let index = argmax(values)
        
        return index == 1 ? 1.0 : 0.0
    }
}

func testBasicModelXOR() {
    
    print("Starting XOR Test")
    
    let talent = XORTalent()
    
    var model = BasicModel(
        layers: [2, 4, 2], // IMPORTANT: 2 outputs
        talent: talent
    )
    
    let trainingPairs: [(Matrix<Double>, Matrix<Double>)] = [
        
        (
            Matrix(rows: 2, columns: 1, grid: [0,0]),
            Matrix(rows: 2, columns: 1, grid: [1,0])
        ),
        
        (
            Matrix(rows: 2, columns: 1, grid: [0,1]),
            Matrix(rows: 2, columns: 1, grid: [0,1])
        ),
        
        (
            Matrix(rows: 2, columns: 1, grid: [1,0]),
            Matrix(rows: 2, columns: 1, grid: [0,1])
        ),
        
        (
            Matrix(rows: 2, columns: 1, grid: [1,1]),
            Matrix(rows: 2, columns: 1, grid: [1,0])
        )
        
    ]
    
    print("Predictions BEFORE training")
    
    for (inputMatrix, _) in trainingPairs {
        
        let prediction = model.predict(input: inputMatrix)
        
        print(
            inputMatrix[0,0],
            inputMatrix[1,0],
            "->",
            flatten(prediction)
        )
    }
    
    for epoch in 0..<10000 {
        
        for (inputMatrix, targetMatrix) in trainingPairs {
            
            model.train(
                input: inputMatrix,
                target: targetMatrix
            )
        }
        
        if epoch % 2000 == 0 {
            print("Epoch:", epoch)
        }
    }
    
    print("\nPredictions AFTER training")
    
    for (inputMatrix, _) in trainingPairs {
        
        let prediction = model.predict(input: inputMatrix)
        
        print(
            inputMatrix[0,0],
            inputMatrix[1,0],
            "->",
            flatten(prediction),
            "decoded:",
            talent.decode(prediction)
        )
    }
}

struct SwiftNNTests {

    @Test func example() async throws {
        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
        testBasicModelXOR()
    }

}
