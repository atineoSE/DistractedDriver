//
//  DistractedDriverInferenceTests.swift
//  DistractedDriverInferenceTests
//
//  Created by Adrian Tineo on 29.05.19.
//  Copyright © 2019 adriantineo.com. All rights reserved.
//

import XCTest
import Vision
import CoreML
@testable import DistractedDriverInference

class DistractedDriverInferencePerformanceTests: XCTestCase {

    func processNativeClassifications(for request: VNRequest, error: Error?) -> (String, Float)? {
        guard let results = request.results else {
            print("Unable to classify image.\n\(error!.localizedDescription)")
            return nil
        }
        guard let classifications = results as? [VNClassificationObservation],
            !classifications.isEmpty else {
                print("Nothing recognized.")
                return nil
        }
        
        let topClassification = classifications.first!
        return (topClassification.identifier, topClassification.confidence)
    }
    
    func testPerformanceNativeClassifier() {
        // Load data
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverClassifier_1000_it().model)
        
        // Create Vision CoreML Request to classify an image
        let classificationRequest = VNCoreMLRequest(model: model, completionHandler: {request, error in
            if let (label, _) = self.processNativeClassifications(for: request, error: error) {
                if let value = classifications[label] {
                    classifications[label] = value + 1
                } else {
                    failedClassifications += 1
                }
            } else {
                failedClassifications += 1
            }
        })
        classificationRequest.imageCropAndScaleOption = .centerCrop
        
        // Classify test images
        print("Classifying \(testImgURLs.count) images")
        let start = CFAbsoluteTimeGetCurrent()
        for imgURL in testImgURLs {
            // Create handler and perform request
            let handler = VNImageRequestHandler(url: imgURL)
            do {
                try handler.perform([classificationRequest])
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
        let end = CFAbsoluteTimeGetCurrent()
        let diff = end - start
        
        // Print metrics
        print("Classifications collected:")
        for k in classifications.keys.sorted() {
            print("\(k) : \(classifications[k]!)")
        }
        print("Failed classifications: \(failedClassifications)")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
    
    func processKerasClassifications(for request: VNRequest, error: Error?) -> (String, Float)? {
        guard let results = request.results else {
            print("Unable to classify image.\n\(error!.localizedDescription)")
            return nil
        }
        guard let classifications = results as? [VNCoreMLFeatureValueObservation],
            !classifications.isEmpty else {
                print("Nothing recognized.")
                return nil
        }
        for entry in classifications {
            let multiArray = entry.featureValue.multiArrayValue!
            var maxProb: Float = 0.0
            var maxIdx = -1
            for idx in 0..<multiArray.count {
                let prob = Float(truncating: multiArray[idx])
                if prob > maxProb {
                    maxProb = prob
                    maxIdx = idx
                }
            }
            if maxIdx == -1 {
                print("Could not find any prob > 0")
                return nil
            }
            //print("returning c\(maxIdx), \(maxProb)")
            return ("c\(maxIdx)", maxProb)
        }
        
        return nil
    }
    
    func testPerformanceKerasClassifier() {
        // Load data
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        //let testImgURLs = [URL(fileURLWithPath: Bundle.main.path(forResource: "img_10003", ofType: "jpg")!)]
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverKeras().model)
        
        // Create Vision CoreML Request to classify an image
        let classificationRequest = VNCoreMLRequest(model: model, completionHandler: {request, error in
            if let (label, _) = self.processKerasClassifications(for: request, error: error) {
                if let value = classifications[label] {
                    classifications[label] = value + 1
                } else {
                    failedClassifications += 1
                }
            } else {
                failedClassifications += 1
            }
        })
        classificationRequest.imageCropAndScaleOption = .centerCrop
        
        // Classify test images
        print("Classifying \(testImgURLs.count) images")
        let start = CFAbsoluteTimeGetCurrent()
        for imgURL in testImgURLs {
            // Create handler and perform request
            let handler = VNImageRequestHandler(url: imgURL)
            do {
                try handler.perform([classificationRequest])
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
        let end = CFAbsoluteTimeGetCurrent()
        let diff = end - start
        
        // Print metrics
        print("Classifications collected:")
        for k in classifications.keys.sorted() {
            print("\(k) : \(classifications[k]!)")
        }
        print("Failed classifications: \(failedClassifications)")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
}
