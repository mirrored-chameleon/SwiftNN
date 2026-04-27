// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftNNTest",
    dependencies: [
        .package(path: "../SwiftNN")
    ],
    targets: [
        .executableTarget(
            name: "SwiftNNTest",
            dependencies: [
                .product(name: "SwiftNN", package: "SwiftNN")
            ]
        )
    ]
)