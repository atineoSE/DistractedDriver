from coremltools.models.neural_network.flexible_shape_utils import *
from coremltools.models.utils import load_spec, save_spec

base_path = "../DistractedDriverCreateML/DistractedDriverCreateML/Models/"
spec = load_spec(base_path + "DistractedDriverClassifier_1000_it.mlmodel")

print("Description before adding new sizes")
print(spec.description)

image_sizes = [NeuralNetworkImageSize(480, 640)]
add_enumerated_image_sizes(spec, feature_name='image', sizes=image_sizes)

# img_size_ranges = NeuralNetworkImageSizeRange()
# img_size_ranges.add_height_range((299,480))
# img_size_ranges.add_width_range((299,640))
#update_image_size_range(spec, feature_name='image', size_range=img_size_ranges)

print("Description after adding new sizes")
print(spec.description)
save_spec(spec, base_path + "DistractedDriverClassifier_1000_it_spec.mlmodel")

# Does not work properly with Create ML model with 299... x 299... flexibility