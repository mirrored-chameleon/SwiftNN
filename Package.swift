// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftNN",
    products: [
        .library(
            name: "SwiftNN",
            targets: ["SwiftNN"]
        )
    ],
    targets: [
        .target(
            name: "SwiftNN"
        )
    ]
)
