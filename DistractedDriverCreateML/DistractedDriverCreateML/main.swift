//
//  main.swift
//  DistractedDriverCreateML
//
//  Created by Adrian Tineo on 26.05.19.
//  Copyright © 2019 adriantineo.com. All rights reserved.
//

import Foundation
import CreateML
import Vision
import CoreML

func createImageClassifier(trainingDataURL: URL, validationDataURL: URL, useSample: Bool) -> MLImageClassifier? {
    // Define training set
    let trainingData = getData(dataPath: trainingDataURL, useSample: useSample)
    let validationData = useSample ? nil : getData(dataPath: validationDataURL, useSample: useSample) 
    let numCategories = trainingData.keys.count
    var numImages = 0
    for key in trainingData.keys {
        numImages += trainingData[key]!.count
    }
    print("TRAINING IMAGE CLASSIFIER ON \(numImages) images in \(numCategories) CATEGORIES")
    
    // Define parameters for training
    let parameters = MLImageClassifier.ModelParameters(featureExtractor: .scenePrint(revision: 1),
                                                       validationData: validationData,
                                                       maxIterations: 1000,
                                                       augmentationOptions: [])

    // Train classifier
    let classifier =  try? MLImageClassifier(trainingData: trainingData, parameters: parameters)
    print("IMAGE CLASSIFIER:")
    print(classifier?.description ?? "NO MODEL CREATED")
    
    return classifier
}

func evaluateClassifier(model:MLImageClassifier, testDataURL: URL, useSample: Bool) {
    let testData = getData(dataPath: testDataURL, useSample: useSample)
    let evaluation = model.evaluation(on: testData)
    print("EVALUATION METRICS FOR MODEL:")
    print(evaluation.description)
    if let error = evaluation.error {
        print("EVALUATION ERROR: \(error.localizedDescription)")
    }
}

func showSamplePredictions(model: MLImageClassifier, from dataSetURL: URL) {
    let predictionData = getData(dataPath: dataSetURL, useSample: true)
    
    print("PRINTING SOME PREDICTIONS FOR IMAGES:")
    for label in predictionData.keys {
        print("PREDICTIONS FOR LABEL \(label):")
        let samples = predictionData[label]!.map { $0 }
        let predictions = try! model.predictions(from: samples)
        for (idx, prediction) in predictions.enumerated() {
            print("\tPredicting image \(idx) as \(prediction)")
        }
    }
}

func getData(dataPath: URL, useSample: Bool) -> [String : [URL]] {
    let dataSource = MLImageClassifier.DataSource.labeledDirectories(at: dataPath)
    guard let labelledImages = try? dataSource.labeledImages() else {
        print("GOT NO IMAGES FROM \(dataPath.absoluteString)")
        return [:]
    }
    
    let outputData: [String : [URL]]
    if useSample {
        var dict: [String : [URL]] = [:]
        for key in labelledImages.keys {
            dict[key] = labelledImages[key]![0..<10].map { $0 }
        }
        outputData = dict
    } else {
        outputData = labelledImages
    }
    
    return outputData
}

// Parse command line parameters
let numParams = CommandLine.argc
var useSample = false
if numParams < 2 {
    print("No arguments are passed. Using default.")
} else {
    if numParams > 1 {
        if CommandLine.arguments[1] == "sample" {
            print("Training classifier with small sample of training data.")
            useSample = true
        }
    }
}

let train = true
let runThroughVision = false

if train {
    // 1. Import data
    let basePath = "/Volumes/Data/ML_data_catalina/DistractedDriver/DistractedDriverCreateML/DistractedDriverCreateML/"
    let trainingDataURL = URL(fileURLWithPath: basePath + "/TrainingData")
    let validationDataURL = URL(fileURLWithPath: basePath + "/ValidationData")
    let testDataURL = URL(fileURLWithPath: basePath + "/TestData")
    
    // 2. Create ML Models
    var start = CFAbsoluteTimeGetCurrent()
    let model = createImageClassifier(trainingDataURL: trainingDataURL, validationDataURL: validationDataURL, useSample: useSample)
    var end = CFAbsoluteTimeGetCurrent()
    var diff = end - start
    
    print("Training time: \(diff)")
    
    // 3. Save model early in case we cut the script short
    let outputPath = basePath + "/Models/DistractedDriverClassifier.mlmodel"
    print("SAVING MODEL TO \(outputPath)")
    let metadata = MLModelMetadata(author: "Adrian Tineo",
                                   shortDescription: "Predict class of behavior for possibly distracted driver (c0-c9)",
                                   license: "GPL",
                                   version: "0.1",
                                   additional: nil)
    try? model?.write(to: URL(fileURLWithPath: outputPath), metadata: metadata)
    
    // 4. Show some predictions
    if let model = model {
        showSamplePredictions(model: model, from: testDataURL)
    } else {
        print("No model to predict with")
    }
    
    // 5. Evaluate Model
    start = CFAbsoluteTimeGetCurrent()
    if let model = model {
        evaluateClassifier(model: model, testDataURL: testDataURL, useSample: useSample)
    } else {
        print("No model to evaluate")
    }
    end = CFAbsoluteTimeGetCurrent()
    diff = end - start
    print("Evaluation time: \(diff)")
}

if runThroughVision {
    
    // 1. Import data
    let fileManager = FileManager.default
    var testImgURLs: [URL] = []
    let basePath = "/Volumes/Data/ML_data_catalina/DistractedDriver/DistractedDriverCreateML/DistractedDriverCreateML/TestData/"
    for i in 0..<10 {
        let classImgURLs = try! fileManager.contentsOfDirectory(atPath: basePath + "/c\(i)/").map { URL(fileURLWithPath: basePath + "/c\(i)/" + $0) }
        testImgURLs += classImgURLs
    }
    
    // 2. Load Models in Vision
    let nativeModel = try! VNCoreMLModel(for: DistractedDriverClassifier_1000_it().model)
    let kerasModel = try! VNCoreMLModel(for: DistractedDriverKeras().model)
    
    var nativeClassifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
    var kerasClassifications = ["c0":0, "c1":0, "c2":0, "c3":0, "c4":0, "c5":0, "c6":0, "c7":0, "c8":0, "c9":0]
    var expectedClassifications = ["c0":521, "c1":450, "c2":461, "c3":476, "c4":494, "c5":423, "c6":459, "c7":401, "c8":377, "c9":405]

    var nativeFailedClassifications = 0
    var kerasFailedClassifications = 0

    // 3. Classify images and print metrics for native model
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
    
    let nativeClassificationRequest = VNCoreMLRequest(model: nativeModel, completionHandler: {request, error in
        if let (label, _) = processNativeClassifications(for: request, error: error) {
            if let value = nativeClassifications[label] {
                nativeClassifications[label] = value + 1
            } else {
                nativeFailedClassifications += 1
            }
        } else {
            nativeFailedClassifications += 1
        }
    })
    nativeClassificationRequest.imageCropAndScaleOption = .scaleFill
    
    print("Classifying \(testImgURLs.count) images with native model")
    switch (nativeClassificationRequest.imageCropAndScaleOption){
    case .centerCrop: print ("Using centerCrop")
    case .scaleFill: print("Using scaleFill")
    case .scaleFit: print("Using scaleFit")
    @unknown default:
        fatalError("Unknown case")
    }
    var start = CFAbsoluteTimeGetCurrent()
    for imgURL in testImgURLs {
        // Create handler and perform request
        let handler = VNImageRequestHandler(url: imgURL)
        do {
            try handler.perform([nativeClassificationRequest])
        } catch {
            print("Failed to perform classification.\n\(error.localizedDescription)")
        }
    }
    var end = CFAbsoluteTimeGetCurrent()
    var diff = end - start
    
    print("Classifications collected for native model:")
    var nativeErrors = 0
    for k in nativeClassifications.keys.sorted() {
        let classificationsForKey = nativeClassifications[k]!
        print("\(k) : \(classificationsForKey)")
        nativeErrors += abs(expectedClassifications[k]! - classificationsForKey)
    }
    print("Failed classifications (could not make classification): \(nativeFailedClassifications)")
    print("Classification errors: (\(nativeErrors)) / (\(testImgURLs.count))")
    print("Average classification time \(diff / Double(testImgURLs.count))")
    
    // 3. Classify images and print metrics for keras model
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
    
    let kerasClassificationRequest = VNCoreMLRequest(model: kerasModel, completionHandler: {request, error in
        if let (label, _) = processKerasClassifications(for: request, error: error) {
            if let value = kerasClassifications[label] {
                kerasClassifications[label] = value + 1
            } else {
                kerasFailedClassifications += 1
            }
        } else {
            kerasFailedClassifications += 1
        }
    })
    kerasClassificationRequest.imageCropAndScaleOption = .scaleFill
    
    print("Classifying \(testImgURLs.count) images with keras model")
    switch (kerasClassificationRequest.imageCropAndScaleOption){
    case .centerCrop: print ("Using centerCrop")
    case .scaleFill: print("Using scaleFill")
    case .scaleFit: print("Using scaleFit")
    @unknown default:
        fatalError("Unknown case")
    }
    start = CFAbsoluteTimeGetCurrent()
    for imgURL in testImgURLs {
        // Create handler and perform request
        let handler = VNImageRequestHandler(url: imgURL)
        do {
            try handler.perform([kerasClassificationRequest])
        } catch {
            print("Failed to perform classification.\n\(error.localizedDescription)")
        }
    }
    end = CFAbsoluteTimeGetCurrent()
    diff = end - start
    
    print("Classifications collected for keras model:")
    var kerasErrors = 0
    for k in kerasClassifications.keys.sorted() {
        let classificationsForKey = kerasClassifications[k]!
        print("\(k) : \(classificationsForKey)")
        kerasErrors += abs(expectedClassifications[k]! - classificationsForKey)
    }
    print("Failed classifications (could not make classification): \(kerasFailedClassifications)")
    print("Classification errors: (\(kerasErrors)) / (\(testImgURLs.count))")
    print("Average classification time \(diff / Double(testImgURLs.count))")
}

