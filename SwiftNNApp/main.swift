//
//  main.swift
//  SwiftNN
//
//  Created by Davyn Monagle on 11/4/2026.
//

//
//  main.swift
//  SwiftNN
//

import SwiftNN
import Foundation
import Surge

// MARK: - GLOBAL VOCAB / EMBEDDINGS

// The forward tokens have all the tokens the model learns with their corresponding id in a dictionary. This pairs with the backwardTokens variable as a mirror of that. It is used in encoding so that you can look up token ids by searching the dictionary by the token name.
var forwardTokens: [String: Int] = [
    "<UNK>": 0,
    "<START>": 1,
    "<END>": 2
]

// Apply temperature allows the model to be more 'creative' with responses. Essentially it changes the difference between probabilites that the model generates, so if it was 1.5, the difference between the probabilities would reduce, so that it would choose the tokens with the lower probabilities more often. If it was lower, like 0.5, it would exagerate the difference between probabilities, so that the highest tokens will be chosen more often than the lower ones.
func applyTemperature(_ logits: [Double], temperature: Double) -> [Double] {
    return logits.map { $0 / temperature }
}

// A mirror for the forward tokens so you can decode the response from the model back into readable text.
var backwardTokens: [Int: String] = [
    0: "<UNK>",
    1: "<START>",
    2: "<END>"
]

// How many dimensions the embedding has. Each embedding corresponds to a word. The more dimensions they have, the more meaning a word has to the model. At the moment (as of writing) the embedding dimensions is equal to 32, so the model can train quicker, as the more dimensions an embedding has, the longer it takes training. The closer embeddings are to eachother. So for example, hello and hi will have very simmilar embeddings, since  they are synonyms.
let embeddingDimensions = 32


// The Matrix that stores all of the embeddings. The model expects up to 1000 words in it's vocabulary, as of writing, there are 706.
var embeddings = Matrix(
    rows: 1000,
    columns: embeddingDimensions,
    grid: (0..<1000 * embeddingDimensions).map { _ in Double.random(in: -1.0...1.0) }
)

// MARK: - HELPERS

// Flattens Matrixes back into 2D arrays to use, so looping through Matrixes becomes a more streamline process.
@inline(__always)
func flattenMatrix(_ matrix: Matrix<Double>) -> [Double] {
    var result: [Double] = []
    result.reserveCapacity(matrix.rows * matrix.columns)

    for r in 0..<matrix.rows {
        for c in 0..<matrix.columns {
            result.append(matrix[r, c])
        }
    }
    return result
}

// Looks up a token id in the forward tokens array I explained earlier.
func tokenID(_ token: String) -> Int {
    forwardTokens[token] ?? forwardTokens["<UNK>"]!
}

// This makes sure that the embeddings do not overflow in size, and if it does, increases the size of vocab, just incase.
func ensureEmbeddingCapacity(for tokenIdentifier: Int) {

    if tokenIdentifier < embeddings.rows {
        return
    }

    let newRowCount = max(tokenIdentifier + 1, embeddings.rows * 2)

    var newGrid: [Double] = []

    newGrid.reserveCapacity(newRowCount * embeddingDimensions)

    for rowIndex in 0..<embeddings.rows {
        for columnIndex in 0..<embeddingDimensions {
            newGrid.append(embeddings[rowIndex, columnIndex])
        }
    }

    let additionalRows = newRowCount - embeddings.rows

    for _ in 0..<additionalRows {
        for _ in 0..<embeddingDimensions {
            newGrid.append(Double.random(in: -1.0...1.0))
        }
    }

    embeddings = Matrix(
        rows: newRowCount,
        columns: embeddingDimensions,
        grid: newGrid
    )
}

// This is one of my favourite functions, because it has pretty much the same althroughout development in the project :D. What it does is takes a sentence, and splits it up into an array of strings, so if you had a sentence like "Hello, world!" it would split it up into ["hello", ",", "world", "!"]. This allows the encoding function to easily create an embedding for a sentence, or for the model to learn more vocab. This is quite a useful function that makes it easier to talk to the model.
func tokenize(sentence input: String) -> [String] {
    var word = ""
    var tokens: [String] = []

    for char in input.lowercased() {
        if char.isWhitespace {
            if !word.isEmpty {
                tokens.append(word)
                word = ""
            }
        } else if char.isPunctuation {
            if !word.isEmpty {
                tokens.append(word)
                word = ""
            }
            tokens.append(char.description)
        } else {
            word.append(char)
        }
    }

    if !word.isEmpty {
        tokens.append(word)
    }

    return tokens
}


// This function helps apply temperature and makes sure that the probabilities all add up to 1. This relates to argmax, but is more 'creative' and expiriments more with word choice, kind of like what I explained with temperature, but in this case it is on the higher side.
func sampleFromDistribution(_ probabilities: [Double], temperature: Double = 0.8) -> Int {

    var adjustedProbabilities: [Double] = []
    adjustedProbabilities.reserveCapacity(probabilities.count)

    for probability in probabilities {
        adjustedProbabilities.append(pow(probability, 1.0 / temperature))
    }

    let total = adjustedProbabilities.reduce(0.0, +)

    if total == 0.0 {
        return Int.random(in: 0..<probabilities.count)
    }

    var normalizedProbabilities: [Double] = []
    normalizedProbabilities.reserveCapacity(adjustedProbabilities.count)

    for value in adjustedProbabilities {
        normalizedProbabilities.append(value / total)
    }

    let randomValue = Double.random(in: 0.0..<1.0)
    var cumulative: Double = 0.0

    for index in 0..<normalizedProbabilities.count {
        cumulative += normalizedProbabilities[index]

        if randomValue < cumulative {
            return index
        }
    }

    return normalizedProbabilities.count - 1
}

// Argmax is kind of like sample from distribution, but in this case it sticks 'more to the rules' in a sense. It makes sure that tokens with a higher probability get chosen way more often. So in the temperature analogy it would be on the lower end.
func argmax(_ values: [Double]) -> Int {
    var bestIndex = 0
    var bestValue = values[0]

    for i in 1..<values.count {
        if values[i] > bestValue {
            bestValue = values[i]
            bestIndex = i
        }
    }

    return bestIndex
}

// MARK: - VOCAB BUILD

// The model needs to learn words, and this is what build vocabulary does. It adds elements to the forward and backward tokens variables, so that the model can increase it's word knowledge and diversity.
func buildVocabulary(from dataset: [(String, String)]) {
    for (input, output) in dataset {
        let tokens = tokenize(sentence: input + " " + output)

        for token in tokens {
            if forwardTokens[token] == nil {
                let id = forwardTokens.count
                forwardTokens[token] = id
                backwardTokens[id] = token
            }
        }
    }
}

// MARK: - MATRIX UTIL

// This function gets the embedding for a given id that corresponds to a word. So if hello was an id of 1, and you wanted to pass that id into the embedding vector function, it would return the embedding that hello has.
func embeddingVector(for id: Int) -> Matrix<Double> {
    var buffer = Array(repeating: 0.0, count: embeddingDimensions)

    if id < embeddings.rows {
        for i in 0..<embeddingDimensions {
            buffer[i] = embeddings[id, i]
        }
    }

    return Matrix(rows: embeddingDimensions, columns: 1, grid: buffer)
}

// This creates an array of empty zeros, except at one position where there is a one. This is used to represent words with meaning for the network, instead of just one id. It shows that the word has meaning here, and not releated to others.
func oneHotVector(for id: Int) -> Matrix<Double> {
    let size = forwardTokens.count
    var vector = Array(repeating: 0.0, count: size)

    if id < size {
        vector[id] = 1.0
    }

    return Matrix(rows: size, columns: 1, grid: vector)
}

// MARK: - TALENT

// This is the english talent. This library is supposed to be modular, and this is the talent designed to let the model speak. It has an input and output type of a String, and its quite simple, so it doesn't take up much performance.
struct EnglishTalent: Talent {

    typealias Input = String
    typealias Output = String

    var tokens: [Output] = []

    func encode(_ input: String) -> Matrix<Double> {

        let tokenList = tokenize(sentence: input)

        var sentenceEmbedding = Array(
            repeating: 0.0,
            count: embeddingDimensions
        )

        var tokenCount: Double = 0.0

        for token in tokenList {

            let tokenIdentifier = forwardTokens[token] ?? 0

            // Ensure embedding exists (IMPORTANT for dynamic vocab)
            ensureEmbeddingCapacity(for: tokenIdentifier)

            if tokenIdentifier < embeddings.rows {

                for dimensionIndex in 0..<embeddingDimensions {
                    sentenceEmbedding[dimensionIndex] += embeddings[tokenIdentifier, dimensionIndex]
                }

                tokenCount += 1.0
            }
        }

        // Normalize (average)
        if tokenCount > 0.0 {
            for dimensionIndex in 0..<embeddingDimensions {
                sentenceEmbedding[dimensionIndex] /= tokenCount
            }
        }

        return Matrix(
            rows: embeddingDimensions,
            columns: 1,
            grid: sentenceEmbedding
        )
    }

    func decode(_ output: Matrix<Double>) -> String {

        let flat = flatten(output)

        var result: [String] = []

        for v in flat {
            let id = Int(round(v))
            result.append(backwardTokens[id] ?? "<UNK>")
        }

        return result.joined(separator: " ")
    }
}

// MARK: - MODEL


// This is the model. The heart of the operation, and it is the largest neural network I have written so far.
struct SimpleModel: Network {

    public var talent: any Talent
    public var weights: [[Matrix<Double>]]
    public var bias: [Matrix<Double>]

    var layerOutputs: [[Double]] = []
    var learningRate: Double = 0.01

    var hiddenSize: Int = 128
    var inputWeights: Matrix<Double>
    var hiddenWeights: Matrix<Double>

    var outputWeights: [[Double]]
    var outputBias: [Double]

    // MARK: - Required by protocol

    mutating func predict(input inputMatrix: Matrix<Double>) -> Matrix<Double> {

        var currentInputs = flatten(inputMatrix)
        layerOutputs.removeAll(keepingCapacity: true)

        for layerIndex in 0..<weights.count {

            let isFinalLayer = (layerIndex == weights.count - 1)
            var currentOutputs: [Double] = []

            for neuronIndex in 0..<weights[layerIndex].count {

                var sum = 0.0

                for inputIndex in 0..<currentInputs.count {
                    sum += weights[layerIndex][neuronIndex][0, inputIndex] * currentInputs[inputIndex]
                }

                sum += bias[layerIndex][neuronIndex, 0]

                if !isFinalLayer {
                    sum = relu(sum)
                }

                currentOutputs.append(sum)
            }

            if isFinalLayer {
                currentOutputs = softmax(currentOutputs)
            }

            layerOutputs.append(currentOutputs)
            currentInputs = currentOutputs
        }

        return Matrix(rows: currentInputs.count, columns: 1, grid: currentInputs)
    }

    mutating func train(input inputMatrix: Matrix<Double>, target targetMatrix: Matrix<Double>) {

        let predicted = flatten(predict(input: inputMatrix))
        let target = flatten(targetMatrix)

        guard predicted.count == target.count else { return }

        var layerErrors = Array(repeating: [Double](), count: weights.count)

        // output error
        var outputError: [Double] = []
        for i in 0..<predicted.count {
            outputError.append(predicted[i] - target[i])
        }

        layerErrors[weights.count - 1] = outputError

        // backprop
        if weights.count > 1 {

            for layerIndex in stride(from: weights.count - 2, through: 0, by: -1) {

                let neuronCount = weights[layerIndex].count
                var errors = Array(repeating: 0.0, count: neuronCount)

                for neuron in 0..<neuronCount {

                    var propagated = 0.0

                    for nextNeuron in 0..<weights[layerIndex + 1].count {
                        propagated += weights[layerIndex + 1][nextNeuron][0, neuron]
                        * layerErrors[layerIndex + 1][nextNeuron]
                    }

                    let neuronOutput = layerOutputs[layerIndex][neuron]
                    errors[neuron] = propagated * reluDerivative(neuronOutput)
                }

                layerErrors[layerIndex] = errors
            }
        }

        // update weights
        for layerIndex in 0..<weights.count {

            let inputs: [Double] =
                (layerIndex == 0)
                ? flatten(inputMatrix)
                : layerOutputs[layerIndex - 1]

            for neuronIndex in 0..<weights[layerIndex].count {

                let error = layerErrors[layerIndex][neuronIndex]

                for inputIndex in 0..<inputs.count {
                    weights[layerIndex][neuronIndex][0, inputIndex] -=
                        learningRate * error * inputs[inputIndex]
                }

                bias[layerIndex][neuronIndex, 0] -= learningRate * error
            }
        }
    }

    // MARK: - RNN helper (unchanged logic)
    func updateHidden(
        previous: Matrix<Double>,
        input: Matrix<Double>,
        inputWeights: Matrix<Double>,
        hiddenWeights: Matrix<Double>
    ) -> Matrix<Double> {

        let inputPart = inputWeights * input
        let hiddenPart = hiddenWeights * previous

        let sum = inputPart + hiddenPart

        var newHidden = Array(repeating: 0.0, count: sum.rows)

        for i in 0..<sum.rows {

            // --- Candidate (like old tanh)
            let candidate = tanh(sum[i, 0])

            // --- Update gate (simple sigmoid)
            let z = 1.0 / (1.0 + exp(-sum[i, 0]))  // sigmoid

            // --- Blend old + new
            newHidden[i] = (1.0 - z) * previous[i, 0] + z * candidate
        }

        return Matrix(rows: sum.rows, columns: 1, grid: newHidden)
    }

    // MARK: - Output layer training (kept, but FIXED naming consistency)

    mutating func trainOutputLayer(hidden: Matrix<Double>, target: Int) {

        let hiddenFlat = flattenMatrix(hidden)
        let vocabSize = outputBias.count

        let targetVector = oneHotVector(for: target)
        let targetFlat = flattenMatrix(targetVector)

        var logits = Array(repeating: 0.0, count: vocabSize)

        for i in 0..<vocabSize {
            var sum = outputBias[i]

            for h in 0..<hiddenSize {
                sum += hiddenFlat[h] * outputWeights[i][h]
            }

            logits[i] = sum
        }

        let scaled = applyTemperature(logits, temperature: 0.7)
        let probs = softmax(scaled)

        for i in 0..<vocabSize {

            let error = probs[i] - targetFlat[i]

            for h in 0..<hiddenSize {
                outputWeights[i][h] -= learningRate * error * hiddenFlat[h]
            }

            outputBias[i] -= learningRate * error
        }
    }
    
    mutating func trainStep(
        previousHidden: Matrix<Double>,
        currentHidden: Matrix<Double>,
        inputVector: Matrix<Double>,
        targetIndex: Int
    ) {

        let hiddenFlat = flattenMatrix(currentHidden)
        let vocabSize = outputBias.count

        // --- Forward (output layer)
        var logits = Array(repeating: 0.0, count: vocabSize)

        for i in 0..<vocabSize {
            var sum = outputBias[i]
            for h in 0..<hiddenSize {
                sum += hiddenFlat[h] * outputWeights[i][h]
            }
            logits[i] = sum
        }

        let probs = softmax(logits)

        // --- Output error
        var outputErrors = Array(repeating: 0.0, count: vocabSize)
        for i in 0..<vocabSize {
            outputErrors[i] = probs[i] - (i == targetIndex ? 1.0 : 0.0)
        }

        // --- Gradient wrt hidden
        var hiddenGradient = Array(repeating: 0.0, count: hiddenSize)

        for h in 0..<hiddenSize {
            var sum = 0.0
            for i in 0..<vocabSize {
                sum += outputWeights[i][h] * outputErrors[i]
            }

            // tanh derivative
            hiddenGradient[h] = sum
        }

        // --- Update output layer (same as before)
        for i in 0..<vocabSize {

            let error = outputErrors[i]

            for h in 0..<hiddenSize {
                outputWeights[i][h] -= learningRate * error * hiddenFlat[h]
            }

            outputBias[i] -= learningRate * error
        }

        // --- Update inputWeights
        for h in 0..<hiddenSize {
            for i in 0..<inputVector.rows {
                inputWeights[h, i] -= learningRate * hiddenGradient[h] * inputVector[i, 0]
            }
        }

        // --- Update hiddenWeights
        for h in 0..<hiddenSize {
            for j in 0..<previousHidden.rows {
                hiddenWeights[h, j] -= learningRate * hiddenGradient[h] * previousHidden[j, 0]
            }
        }
    }

    // MARK: - init unchanged

    init(layers: [Int], talent: any Talent) {

        self.talent = talent
        self.weights = []
        self.bias = []

        self.hiddenWeights = Matrix(
            rows: hiddenSize,
            columns: hiddenSize,
            grid: (0..<hiddenSize * hiddenSize).map { _ in Double.random(in: -1...1) }
        )

        self.inputWeights = Matrix(
            rows: hiddenSize,
            columns: embeddingDimensions,
            grid: (0..<hiddenSize * embeddingDimensions).map { _ in Double.random(in: -1...1) }
        )

        for i in 0..<layers.count - 1 {

            let inSize = layers[i]
            let outSize = layers[i + 1]

            var layerWeights: [Matrix<Double>] = []

            for _ in 0..<outSize {

                layerWeights.append(Matrix(
                    rows: 1,
                    columns: inSize,
                    grid: (0..<inSize).map { _ in Double.random(in: -1...1) }
                ))
            }

            weights.append(layerWeights)

            bias.append(Matrix(
                rows: outSize,
                columns: 1,
                grid: Array(repeating: 0.0, count: outSize)
            ))
        }

        let vocabSize = forwardTokens.count


        let hiddenSizeValue = hiddenSize

        var tempOutputWeights: [[Double]] = []

        for _ in 0..<vocabSize {

            var row: [Double] = []

            for _ in 0..<hiddenSizeValue {
                row.append(Double.random(in: -1.0...1.0))
            }

            tempOutputWeights.append(row)
        }

        self.outputWeights = tempOutputWeights
        self.outputBias = Array(repeating: 0.0, count: vocabSize)
    }
}

// This function loads a data set I have, called Bubble Data, since this was originally going to be based off of Bubble from the amazing digital circus, I just have some basic data in there for now, not Bubble style yet. At some point it probably will be though.
func loadDataset(from path: String) -> [(String, String)] {

    guard let rawText = try? String(contentsOfFile: path, encoding: .utf8) else {
        print("❌ Failed to load dataset file at path: \(path)")
        return []
    }

    var dataset: [(String, String)] = []

    let lines = rawText.components(separatedBy: .newlines)

    for line in lines {

        let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmedLine.isEmpty { continue }

        let parts = trimmedLine.components(separatedBy: "->")

        if parts.count != 2 { continue }

        let input = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
        let output = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)

        if input.isEmpty || output.isEmpty { continue }

        dataset.append((input, output))
    }

    return dataset
}

// This is the function that tests the model, it contains a training loop, gets the data, trains the model and enters you into an interactive test, and it really ties the entire operation together.
func testModel() {
    
    let path = "/Users/davyn/Library/Mobile Documents/com~apple~CloudDocs/Development/SwiftNN/SwiftNNApp/bubble_data.txt"
    let dataset = loadDataset(from: path)
    
    guard !dataset.isEmpty else {
        print("❌ Dataset is empty")
        return
    }
    
    print("📚 Dataset loaded: \(dataset.count) pairs")
    
    buildVocabulary(from: dataset)
    
    print("📖 Vocabulary size: \(forwardTokens.count)")
    
    var model = SimpleModel(
        layers: [128, 64, 64, forwardTokens.count],
        talent: EnglishTalent()
    )
    
    let epochs = 30
    
    for epoch in 0..<epochs {

        var totalLoss: Double = 0.0
        var steps: Double = 0.0
        
        let shuffled = dataset.shuffled()

        for (input, output) in shuffled {

            let inputIDs = tokenize(sentence: input).map(tokenID)

            var outputTokens = tokenize(sentence: output)
            outputTokens.insert("<START>", at: 0)
            outputTokens.append("<END>")

            let outputIDs = outputTokens.map(tokenID)

            guard inputIDs.count > 0, outputIDs.count > 1 else { continue }

            // --- Reset hidden state
            var hiddenState = Matrix<Double>(
                rows: model.hiddenSize,
                columns: 1,
                grid: Array(repeating: 0.0, count: model.hiddenSize)
            )

            var previousHidden = hiddenState

            // --- Encode input (context)
            for id in inputIDs {
                let inputVector = embeddingVector(for: id)

                hiddenState = model.updateHidden(
                    previous: hiddenState,
                    input: inputVector,
                    inputWeights: model.inputWeights,
                    hiddenWeights: model.hiddenWeights
                )
            }

            // --- Train output sequence
            for t in 0..<(outputIDs.count - 1) {

                let currentID = outputIDs[t]
                let targetID = outputIDs[t + 1]

                let inputVector = embeddingVector(for: currentID)

                previousHidden = hiddenState

                hiddenState = model.updateHidden(
                    previous: previousHidden,
                    input: inputVector,
                    inputWeights: model.inputWeights,
                    hiddenWeights: model.hiddenWeights
                )

                // 🔥 FORWARD (single correct output path)
                let hiddenFlat = flattenMatrix(hiddenState)
                let vocabSize = model.outputBias.count

                var logits = Array(repeating: 0.0, count: vocabSize)

                for i in 0..<vocabSize {
                    var sum = model.outputBias[i]
                    for h in 0..<model.hiddenSize {
                        sum += hiddenFlat[h] * model.outputWeights[i][h]
                    }
                    logits[i] = sum
                }

                let probs = softmax(logits)

                // --- Loss
                let loss = -log(max(probs[targetID], 1e-12))
                totalLoss += loss
                steps += 1

                // 🔥 Backprop (matches this forward path)
                model.trainStep(
                    previousHidden: previousHidden,
                    currentHidden: hiddenState,
                    inputVector: inputVector,
                    targetIndex: targetID
                )
            }
        }

        let avgLoss = totalLoss / max(steps, 1)
        print("Epoch \(epoch) done | avgLoss ~ \(avgLoss)")
    }
     
    print("💬 Entering interactive test mode (type 'exit' to quit)\n")

    while true {

        print("You: ", terminator: "")
        guard let inputText = readLine(), !inputText.isEmpty else {
            continue
        }

        if inputText.lowercased() == "exit" {
            break
        }

        // Encode input into token IDs
        let inputTokens = tokenize(sentence: inputText)
        let inputIDs = inputTokens.map { tokenID($0) }

        var hiddenState = Matrix(
            rows: model.hiddenSize,
            columns: 1,
            grid: Array(repeating: 0.0, count: model.hiddenSize)
        )

        // 🔥 ENCODE USER INPUT INTO CONTEXT
        for id in inputIDs {
            let inputVector = embeddingVector(for: id)

            hiddenState = model.updateHidden(
                previous: hiddenState,
                input: inputVector,
                inputWeights: model.inputWeights,
                hiddenWeights: model.hiddenWeights
            )
        }

        var currentID = tokenID("<START>")
        var outputWords: [String] = []

        for _ in 0..<100 {

            let inputVector = embeddingVector(for: currentID)

            hiddenState = model.updateHidden(
                previous: hiddenState,
                input: inputVector,
                inputWeights: model.inputWeights,
                hiddenWeights: model.hiddenWeights
            )

            // Forward pass
            let outputMatrix = model.predict(input: hiddenState)
            var probabilities = flattenMatrix(outputMatrix)

            let repetitionPenalty: Double = 1.2

            for generatedWord in outputWords {

                if let tokenIdentifier = forwardTokens[generatedWord] {
                    probabilities[tokenIdentifier] = probabilities[tokenIdentifier] / repetitionPenalty
                }
            }

            let total = probabilities.reduce(0.0, +)

            if total > 0.0 {
                for index in 0..<probabilities.count {
                    probabilities[index] = probabilities[index] / total
                }
            }

            let nextID = sampleFromDistribution(probabilities, temperature: 0.8)

            // ✅ STOP CONDITIONS
            if nextID == tokenID("<END>") { break }
            if nextID == tokenID("<UNK>") { break }

            let word = backwardTokens[nextID] ?? "<UNK>"
            outputWords.append(word)

            currentID = nextID
        }

        print("Model: \(outputWords.joined(separator: " "))\n")
    }
}

testModel()
