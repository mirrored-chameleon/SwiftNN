# SwiftNN

[![Swift 6.0](https://shields.io)](https://swift.org)
[![License: MIT](https://shields.io)](https://opensource.org)
[![Platform: macOS | Linux | Windows](https://shields.io)](https://swift.org)

SwiftNN is a 100% cross-platform, open-source machine learning library. At the moment, it has completed a variety of tasks, such as playing Space Invaders, doing the XOR test, and acting as a buzzy bee trying to survive the monumental task of finding flowers and returning them to the hive (repo coming soon 🐝).

It was built for the science talent search and is designed to be lightweight, simple to import, and requires absolutely zero external dependencies. Built entirely in pure Swift, the entire core architecture fits neatly into a few folders, the core engine, math and extension folders such as SwiftNN Language to handle transformers.

I hope to see the open-source community help build upon this library, creating larger, more versatile models that expand beyond what I can currently produce. Good luck, and I look forward to seeing where the developer community takes this project!

---

## 🚀 Key Features

* **Zero Dependencies**: Built entirely without heavy external frameworks like PyTorch or Apple Accelerate.  
* **The Talent Pattern**: Employs a unique modular design pattern to abstract data handling. By splitting logic across the `Talent` and `Network` protocols, the exact same core model can be adapted for entirely separate purposes.  
* **Custom Matrix Engine**: Utilizes a single-dimensional flat array for cache-friendly memory index mapping. Features custom matrix operator overloading supporting modern feedforward and backpropagation requirements.

---

## 📦 Installation

Add this package to your `Package.swift` dependencies:

```swift
.package(url: "https://github.com/mirrored-chameleon/SwiftNN.git", from: "1.0.0")
```

---

## 🥳 Have Fun!

Good luck building upon this repository! I look forward to reviewing your pull requests and contributions. I hope you enjoy experimenting with this architecture as much as I enjoyed designing it. Have fun!
