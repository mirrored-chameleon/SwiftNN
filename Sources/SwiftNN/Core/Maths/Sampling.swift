//
//  Sampling.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 22/08/2026.
//

import Foundation

public func sampleIndex(from probabilities: [Double], temperature: Double = 1.0) -> Int {
    precondition(!probabilities.isEmpty, "Cannot sample from an empty array.")

    let safeTemperature = max(temperature, 0.0001)

    let adjustedValues = probabilities.map { probability in
        pow(max(probability, 0.0), 1.0 / safeTemperature)
    }

    let totalValue = adjustedValues.reduce(0.0, +)

    if totalValue == 0.0 {
        return Int.random(in: 0 ..< probabilities.count)
    }

    let normalizedValues = adjustedValues.map { value in
        value / totalValue
    }

    let randomValue = Double.random(in: 0.0 ..< 1.0)

    var cumulativeValue = 0.0

    for index in 0 ..< normalizedValues.count {
        cumulativeValue += normalizedValues[index]

        if randomValue < cumulativeValue {
            return index
        }
    }

    return normalizedValues.count - 1
}
