import Foundation

public struct XORTalent: Talent {

    public typealias Input = [Double]
    public typealias Output = Int

    public var tokens: [Int] = [0, 1]

    public init() {}

    public func encode(_ input: [Double]) -> Matrix<Double> {
        Matrix(rows: input.count, columns: 1, grid: input)
    }

    public func decode(_ output: Matrix<Double>) -> Int {
        argmax(flatten(output))
    }
}

public func runXORTest() {

    print("Starting XOR test...")

    var model = BasicModel(
        layers: [2, 4, 2],
        talent: XORTalent()
    )

    let trainingData: [([Double], [Double])] = [
        ([0, 0], [1, 0]),
        ([0, 1], [0, 1]),
        ([1, 0], [0, 1]),
        ([1, 1], [1, 0])
    ]

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

    for sample in trainingData {
        let input = Matrix<Double>(
            rows: 2,
            columns: 1,
            grid: sample.0
        )

        let output = model.predict(input: input)
        let prediction = model.predictAction(input: input)

        print("Input: \(sample.0) -> Output: \(flatten(output)) -> Class: \(prediction)")
    }

    print("XOR test finished")
}
