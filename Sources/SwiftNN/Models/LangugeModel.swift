//
//  LangugeModel.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 5/9/2026.
//

import Foundation

enum LanguageErrors: Error {
    case unsupportedData
    case unknownToken(String)
}

struct LanguageModel: Codable {
    var transformer: Transformer
    let learningRate: Double
    
    mutating func generate(
        from input: String,
        maxTokens: Int
    ) throws -> String {
        guard maxTokens > 0 else {
            return input
        }
        
        var tokens =
        input.split(separator: " ")
            .map(String.init)
        
        for _ in 0 ..< maxTokens {
            var inputIDs: [Double] = []
            
            for token in tokens {
                guard let id =
                        transformer.vocabulary.id(for: token)
                else {
                    throw LanguageErrors.unknownToken(token)
                }
                
                inputIDs.append(Double(id))
            }
            
            guard !inputIDs.isEmpty else {
                break
            }
            
            let input = Matrix<Double>(
                rows: inputIDs.count,
                columns: 1,
                grid: inputIDs,
            )
            
            let prediction =
            transformer.forward(input)
            
            let lastRow =
            prediction.rows - 1
            
            let logits =
            prediction[lastRow]
            
            guard let nextTokenID =
                    logits.indices.max(
                        by: {
                            logits[$0] < logits[$1]
                        },
                    )
            else {
                break
            }
            
            guard let nextToken =
                    transformer.vocabulary.token(
                        for: nextTokenID,
                    )
            else {
                break
            }
            
            tokens.append(nextToken)
            
            if nextToken == "<end>" {
                break
            }
        }
        
        return tokens.joined(separator: " ")
    }
    
    mutating func train(
        on examples: [(input: String, target: String)],
        epochs: Int
    ) {
        for epoch in 0 ..< epochs {
            var totalLoss = 0.0
            
            for example in examples {
                let inputTokens =
                example.input.split(separator: " ")
                
                let targetTokens =
                example.target.split(separator: " ")
                
                var inputIDs: [Double] = []
                
                for token in inputTokens {
                    guard let id =
                            transformer.vocabulary.id(
                                for: String(token),
                            )
                    else {
                        continue
                    }
                    
                    inputIDs.append(Double(id))
                }
                
                var targetIDs: [Double] = []
                
                for token in targetTokens {
                    guard let id =
                            transformer.vocabulary.id(
                                for: String(token),
                            )
                    else {
                        continue
                    }
                    
                    targetIDs.append(Double(id))
                }
                
                guard
                    !inputIDs.isEmpty,
                    !targetIDs.isEmpty
                else {
                    continue
                }
                
                let input = Matrix<Double>(
                    rows: inputIDs.count,
                    columns: 1,
                    grid: inputIDs,
                )
                
                let targetID =
                targetIDs[targetIDs.count - 1]
                
                let vocabularySize =
                transformer.vocabulary.idToToken.count
                
                var target = Matrix<Double>(
                    rows: 1,
                    columns: vocabularySize,
                    grid: Array(
                        repeating: 0.0,
                        count: vocabularySize,
                    ),
                )
                
                target[0, Int(targetID)] = 1.0
                
                let loss =
                transformer.trainStep(
                    input: input,
                    target: target,
                    learningRate: learningRate,
                )
                
                totalLoss += loss
            }
            
            print(
                "Epoch \(epoch) Loss: \(totalLoss)",
            )
        }
    }
    
    func export() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        let data = try encoder.encode(self)
        
        guard let json = String(data: data, encoding: .utf8) else {
            throw LanguageErrors.unsupportedData
        }
        
        return json
    }
    
    static func `import`(from json: String) throws -> LanguageModel {
        guard let data = json.data(using: .utf8) else {
            throw LanguageErrors.unsupportedData
        }
        
        return try JSONDecoder().decode(
            LanguageModel.self,
            from: data
        )
    }
}
