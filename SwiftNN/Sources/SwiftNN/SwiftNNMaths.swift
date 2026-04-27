// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation

public func relu(_ inputValue: Double) -> Double {
    return max(0.0, inputValue)
}

public func reluDerivative(_ inputValue: Double) -> Double {
    return inputValue > 0.0 ? 1.0 : 0.0
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
    var rows: Int
    var columns: Int
    var grid: [T]

    public subscript(row: Int, col: Int) -> T {
        get {
            return grid[row * columns + col]
        }
        set {
            grid[row * columns + col] = newValue
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

}
