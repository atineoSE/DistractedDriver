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
import UIKit
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
    
    func testPrecisionNativeClassifierCenterCrop() {
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverClassifier_1000_it().model)
        print("Using model:")
        print(model.description)
        
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
        switch (classificationRequest.imageCropAndScaleOption){
        case .centerCrop: print ("Using centerCrop")
        case .scaleFill: print("Using scaleFill")
        case .scaleFit: print("Using scaleFit")
        @unknown default:
            fatalError("Unkown case")
        }
        
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
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
    
    func testPrecisionNativeClassifierScaleFill() {
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverClassifier_1000_it().model)
        print("Using model:")
        print(model.description)
        
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
        classificationRequest.imageCropAndScaleOption = .scaleFill
        switch (classificationRequest.imageCropAndScaleOption){
        case .centerCrop: print ("Using centerCrop")
        case .scaleFill: print("Using scaleFill")
        case .scaleFit: print("Using scaleFit")
        @unknown default:
            fatalError("Unkown case")
        }
        
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
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
    
    func testPrecisionNativeClassifierCropCenterCrop() {
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverClassifier_1000_it_crop().model)
        print("Using model:")
        print(model.description)
        
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
        switch (classificationRequest.imageCropAndScaleOption){
        case .centerCrop: print ("Using centerCrop")
        case .scaleFill: print("Using scaleFill")
        case .scaleFit: print("Using scaleFit")
        @unknown default:
            fatalError("Unkown case")
        }
        
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
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
    
    func testPrecisionNativeClassifierCropScaleFill() {
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverClassifier_1000_it_crop().model)
        print("Using model:")
        print(model.description)
        
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
        classificationRequest.imageCropAndScaleOption = .scaleFill
        switch (classificationRequest.imageCropAndScaleOption){
        case .centerCrop: print ("Using centerCrop")
        case .scaleFill: print("Using scaleFill")
        case .scaleFit: print("Using scaleFit")
        @unknown default:
            fatalError("Unkown case")
        }
        
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
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
    
    
    func testPrecisionNativeSpecClassifierCenterCrop() {
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverClassifier_1000_it_spec().model)
        print("Using model:")
        print(model.description)
        
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
        switch (classificationRequest.imageCropAndScaleOption){
        case .centerCrop: print ("Using centerCrop")
        case .scaleFill: print("Using scaleFill")
        case .scaleFit: print("Using scaleFit")
        @unknown default:
            fatalError("Unkown case")
        }
        
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
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
    
    func testPrecisionNativeSpecClassifierScaleFill() {
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverClassifier_1000_it_spec().model)
        print("Using model:")
        print(model.description)
        
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
        classificationRequest.imageCropAndScaleOption = .scaleFill
        switch (classificationRequest.imageCropAndScaleOption){
        case .centerCrop: print ("Using centerCrop")
        case .scaleFill: print("Using scaleFill")
        case .scaleFit: print("Using scaleFit")
        @unknown default:
            fatalError("Unkown case")
        }
        
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
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
    
    // Does not work
    func testPrecisionNativeSpecClassifierScaleFillNoVision() {
        //let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        let testImgPaths = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil)
        
        func convert(uiImage: UIImage) -> CVPixelBuffer {
            let ciImage = CIImage(cgImage: uiImage.cgImage!)
            let size = CGSize(width: 640, height: 480)
            UIGraphicsBeginImageContextWithOptions(size, true, 0.0)
            let ciContext = CIContext(cgContext: UIGraphicsGetCurrentContext()!, options: nil)
            var cvPixelBuffer: CVPixelBuffer? = nil
            CVPixelBufferCreate(nil, 640, 480, kCVPixelFormatType_32ARGB, nil, &cvPixelBuffer)
            ciContext.render(ciImage, to: cvPixelBuffer!)
            
            return cvPixelBuffer!
        }
        
        let inputImages = testImgPaths.map { path -> CVPixelBuffer in
            let uiImage = UIImage(contentsOfFile: path)!
            let cvPixelBuffer = convert(uiImage: uiImage)
            return cvPixelBuffer
        }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model
        let model = DistractedDriverClassifier_1000_it_spec()
        
        // Classify test images
        print("Classifying \(testImgPaths.count) images")
        let start = CFAbsoluteTimeGetCurrent()
        for input in inputImages {
            if let prediction = try? model.prediction(image: input) {
                let label = prediction.classLabel
                if let value = classifications[label] {
                    classifications[label] = value + 1
                } else {
                    failedClassifications += 1
                }
            } else {
                failedClassifications += 1
            }
        }
        let end = CFAbsoluteTimeGetCurrent()
        let diff = end - start
 
        // Print metrics
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgPaths.count))")
        print("Average classification time \(diff / Double(testImgPaths.count))")
    }
    
    func testPrecisionKerasClassifierCenterCrop() {
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverKeras().model)
        print("Using model:")
        print(model.description)
        
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
        switch (classificationRequest.imageCropAndScaleOption){
        case .centerCrop: print ("Using centerCrop")
        case .scaleFill: print("Using scaleFill")
        case .scaleFit: print("Using scaleFit")
        @unknown default:
            fatalError("Unkown case")
        }
        
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
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
    
    func testPrecisionKerasClassifierScaleFill() {
        let testImgURLs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: nil).map { URL(fileURLWithPath: $0) }
        
        var classifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
        var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]
        var failedClassifications = 0
        
        // Load model into Vision framework
        let model = try! VNCoreMLModel(for: DistractedDriverKeras().model)
        print("Using model:")
        print(model.description)
        
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
        classificationRequest.imageCropAndScaleOption = .scaleFill
        switch (classificationRequest.imageCropAndScaleOption){
        case .centerCrop: print ("Using centerCrop")
        case .scaleFill: print("Using scaleFill")
        case .scaleFit: print("Using scaleFit")
        @unknown default:
            fatalError("Unkown case")
        }
        
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
        print("Classifications collected for model:")
        var nativeErrors = 0
        for k in classifications.keys.sorted() {
            let classificationsForKey = classifications[k]!
            print("\(k) : \(classificationsForKey)")
            nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
        }
        print("Failed classifications (could not make classification): \(failedClassifications)")
        print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
        print("Average classification time \(diff / Double(testImgURLs.count))")
    }
}
