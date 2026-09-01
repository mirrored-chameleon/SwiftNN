//
//  Matrix.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 22/08/2026.
//

import Foundation

public struct Matrix<T: FloatingPoint> {
    public var rows: Int
    public var columns: Int
    public var grid: [T]

    public init(rows: Int, columns: Int, grid: [T]) {
        self.rows = rows
        self.columns = columns
        self.grid = grid
    }

    public subscript(row: Int, col: Int) -> T {
        get {
            return grid[row * columns + col]
        }
        set {
            grid[row * columns + col] = newValue
        }
    }

    public subscript(row: Int) -> [T] {
        get {
            precondition(row >= 0 && row < rows, "Row index out of range.")

            let startIndex = row * columns
            let endIndex = startIndex + columns

            return Array(grid[startIndex..<endIndex])
        }
        set {
            precondition(row >= 0 && row < rows, "Row index out of range.")
            precondition(
                newValue.count == columns,
                "Row must contain exactly \(columns) values."
            )

            let startIndex = row * columns
            let endIndex = startIndex + columns

            grid.replaceSubrange(startIndex..<endIndex, with: newValue)
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

    static public func * (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        return dot(lhs, rhs)
    }

    static public func / (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        var result = lhs

        for i in 0..<result.grid.count {
            result.grid[i] = lhs.grid[i] / rhs
        }

        return result
    }

    static public func dot(_ a: Matrix<T>, _ b: Matrix<T>) -> Matrix<T> {
        precondition(a.columns == b.rows)

        var result = Matrix<T>(
            rows: a.rows, columns: b.columns, grid: Array(repeating: 0, count: a.rows * b.columns))

        for i in 0..<a.rows {
            for j in 0..<b.columns {
                var sum: T = 0
                for k in 0..<a.columns {
                    sum = sum + a.grid[i * a.columns + k] * b.grid[k * b.columns + j]
                }
                result.grid[i * b.columns + j] = sum
            }
        }

        return result
    }

    public var transposed: Matrix<T> {
        var result = Matrix<T>(
            rows: columns,
            columns: rows,
            grid: Array(repeating: T.zero, count: grid.count)
        )

        for row in 0..<rows {
            for col in 0..<columns {
                result[col, row] = self[row, col]
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

        guard let newGrid = testGrid as? [T] else { return Matrix<T>(rows: 1, columns: 1, grid: [404]) }

        return Matrix<T>(
            rows: x.rows,
            columns: x.columns,
            grid: newGrid
        )
    }
}
