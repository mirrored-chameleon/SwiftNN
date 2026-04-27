// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation

struct XORTalent: Talent {

    typealias Input = [Double]
    typealias Output = Int

    var tokens: [Int] = [0, 1]

    func encode(_ input: [Double]) -> Matrix<Double> {
        return Matrix(rows: input.count, columns: 1, grid: input)
    }

    func decode(_ output: Matrix<Double>) -> Int {
        return argmax(flatten(output))
    }
}

func runXORTest() {

    print("🧠 Starting XOR test...")

    var model = BasicModel(
        layers: [2, 4, 2], // slightly bigger hidden layer helps
        talent: XORTalent()
    )

    let trainingData: [([Double], [Double])] = [
        ([0, 0], [1, 0]),
        ([0, 1], [0, 1]),
        ([1, 0], [0, 1]),
        ([1, 1], [1, 0])
    ]

    // Train
    for _ in 0..<5000 {
        for sample in trainingData {

            let input = Matrix<Double>(
                rows: 2,
                columns: 1,
                grid: sample.0
            )

            let target = Matrix<Double>(
                rows: 2,
                columns: 1,
                grid: sample.1
            )

            model.train(input: input, target: target)
        }
    }

    // Test
    for sample in trainingData {

        let input = Matrix<Double>(
            rows: 2,
            columns: 1,
            grid: sample.0
        )

        let output = model.predict(input: input)
        let prediction = model.predictAction(input: input)

        print("Input: \(sample.0) → Output: \(flatten(output)) → Class: \(prediction)")
    }

    print("✅ XOR test finished")
}


@main
struct SwiftNN {
    static func main() {
        runXORTest()
    }
}
