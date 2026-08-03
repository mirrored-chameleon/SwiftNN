// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation

public func relu(_ inputValue: Double) -> Double {
    return max(0.0, inputValue)
}
public func reluMatrix(_ inputMatrix: Matrix<Double>) -> Matrix<Double> {

    Matrix(
        rows: inputMatrix.rows,
        columns: inputMatrix.columns,
        grid: inputMatrix.grid.map { value in
            relu(value)
        }
    )
}
public func reluDerivative(_ inputValue: Double) -> Double {
    return inputValue > 0.0 ? 1.0 : 0.0
}
public func sigmoid(_ inputValue: Double) -> Double {
    return 1.0 / (1.0 + exp(-inputValue))
}
public func tanhDerivative(fromOutput outputValue: Double) -> Double {
    return 1.0 - (outputValue * outputValue)
}
public func softmax(_ inputValues: [Double]) -> [Double] {

    let largestValue = inputValues.max() ?? 0.0

    let exponentValues = inputValues.map { value in
        exp(value - largestValue)
    }

    let exponentSum = exponentValues.reduce(0.0, +)

    return exponentValues.map { exponentValue in
        exponentValue / exponentSum
    }
}
public func argmax(_ values: [Double]) -> Int {

    var bestIndex = 0
    var bestValue = values[0]

    for index in 1..<values.count {

        let currentValue = values[index]

        if currentValue > bestValue {
            bestValue = currentValue
            bestIndex = index
        }
    }

    return bestIndex
}
public func sampleIndex(from probabilities: [Double], temperature: Double = 1.0) -> Int {

    precondition(!probabilities.isEmpty, "Cannot sample from an empty array.")

    let safeTemperature = max(temperature, 0.0001)

    let adjustedValues = probabilities.map { probability in
        pow(max(probability, 0.0), 1.0 / safeTemperature)
    }

    let totalValue = adjustedValues.reduce(0.0, +)

    if totalValue == 0.0 {
        return Int.random(in: 0..<probabilities.count)
    }

    let normalizedValues = adjustedValues.map { value in
        value / totalValue
    }

    let randomValue = Double.random(in: 0.0..<1.0)

    var cumulativeValue = 0.0

    for index in 0..<normalizedValues.count {
        cumulativeValue += normalizedValues[index]

        if randomValue < cumulativeValue {
            return index
        }
    }

    return normalizedValues.count - 1
}
public func clipped(_ value: Double, limit: Double) -> Double {
    return max(-limit, min(limit, value))
}
public func flatten(_ matrix: Matrix<Double>) -> [Double] {

    var flattenedValues: [Double] = []

    flattenedValues.reserveCapacity(matrix.rows * matrix.columns)

    for rowIndex in 0..<matrix.rows {

        for columnIndex in 0..<matrix.columns {

            flattenedValues.append(matrix[rowIndex, columnIndex])

        }
    }

    return flattenedValues
}
public struct Matrix<T: FloatingPoint> {
    public var rows: Int
    public var columns: Int
    public var grid: [T]

    public var transposed: Matrix<T> {

        transpose()

    }

    public init(rows: Int, columns: Int, grid: [T]) {
        self.rows = rows
        self.columns = columns
        self.grid = grid
    }

    private func index(row: Int, column: Int) -> Int {

        return row * columns + column

    }

    public subscript(row: Int, col: Int) -> T {
        get {
            return grid[index(row: row, column: col)]
        }
        set {
            grid[index(row: row, column: col)] = newValue
        }
    }

    public subscript(row: Int) -> [T] {
        get {
            precondition(row >= 0 && row < rows)

            let start = row * columns
            let end = start + columns

            return Array(grid[start..<end])
        }
        set {
            precondition(row >= 0 && row < rows)
            precondition(newValue.count == columns)

            let start = row * columns

            for i in 0..<columns {
                grid[start + i] = newValue[i]
            }
        }
    }

    static public func * (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        var result = rhs
        for i in 0..<result.grid.count {
            result.grid[i] = lhs * result.grid[i]
        }
        return result
    }

    static public func + (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        precondition(lhs.rows == rhs.rows && lhs.columns == rhs.columns)

        var result = lhs
        for i in 0..<lhs.grid.count {
            result.grid[i] = lhs.grid[i] + rhs.grid[i]
        }
        return result
    }

    static public func - (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        precondition(lhs.rows == rhs.rows && lhs.columns == rhs.columns)

        var result = lhs
        for i in 0..<lhs.grid.count {
            result.grid[i] = lhs.grid[i] - rhs.grid[i]
        }
        return result
    }

    static public func * (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        var result = lhs
        for i in 0..<lhs.grid.count {
            result.grid[i] = lhs.grid[i] * rhs
        }
        return result
    }

    public func transpose() -> Matrix<T> {

        var newGrid = Array(
            repeating: T.zero,
            count: rows * columns
        )

        for row in 0..<rows {

            for column in 0..<columns {

                newGrid[column * rows + row] =
                    self[row, column]

            }

        }

        return Matrix(
            rows: columns,
            columns: rows,
            grid: newGrid
        )

    }

    static public func * (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        return dot(lhs, rhs)
    }

    public func map(
        _ transform: (T) -> T
    ) -> Matrix<T> {

        Matrix(
            rows: rows,
            columns: columns,
            grid: grid.map(transform)
        )

    }

    static public func dot(_ a: Matrix<T>, _ b: Matrix<T>) -> Matrix<T> {
        precondition(a.columns == b.rows)

        var result = Matrix<T>(
            rows: a.rows, columns: b.columns, grid: Array(repeating: 0, count: a.rows * b.columns))

        for i in 0..<a.rows {
            for j in 0..<b.columns {
                var sum: T = 0
                for k in 0..<a.columns {
                    sum = sum + a[i, k] * b[k, j]
                }
                result[i, j] = sum
            }
        }

        return result
    }

}
extension Matrix {

    public static func random(rows: Int, columns: Int, in range: ClosedRange<Double> = -1.0...1.0)
        -> Matrix<Double>
    {

        let values = (0..<(rows * columns)).map { _ in
            Double.random(in: range)
        }

        return Matrix<Double>(
            rows: rows,
            columns: columns,
            grid: values
        )
    }

    public static func matrixLog(_ val: Matrix<T>) -> Matrix<T> {

        guard let x = val as? Matrix<Double> else {
            return Matrix<T>(rows: 1, columns: 1, grid: [404])
        }

        let testGrid = x.grid.map {
            Foundation.log(max($0, 1e-8))
        }

        guard let newGrid = testGrid as? [T] else {
            return Matrix<T>(rows: 1, columns: 1, grid: [404])
        }

        return Matrix<T>(
            rows: x.rows,
            columns: x.columns,
            grid: newGrid
        )
    }
}
