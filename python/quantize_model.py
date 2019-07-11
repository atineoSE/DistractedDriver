import coremltools.models.utils

keras_model = load_model("./Model/distracted_driver.h5")
coreml_model = convert(keras_model, image_input_names = "input1")
coreml_model.save("./Model/DistractedDriverKeras.mlmodel")
