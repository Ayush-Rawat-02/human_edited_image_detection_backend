from keras.models import Model, load_model
import numpy as np




# from PIL import Image
from PIL import Image, ImageChops, ImageEnhance
model=load_model('./model/model_vgg16_fake_detection.h5')

image_size = (128, 128)





def convert_to_ela_image(path, quality):
    # creating a temporary filename for an intermediate image
    temp_filename = 'temp_file_name.jpg'
    # filename for ela image that will be generated
    ela_filename = 'temp_ela.png'
    
    # open image and convert to RGB
    image = Image.open(path).convert('RGB')

    # save image as jpg and keep quality as before
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    # calculate pixel difference between original image and RGB (new image) 
    # which will represents areas of image that have been altered.
    ela_image = ImageChops.difference(image, temp_image)
    
    # calculating minimum and maximum pixel values in the images
    extrema = ela_image.getextrema()

    # finds the maximum difference value among the extrema. This value is used to scale the ELA image.
    max_diff = max([ex[1] for ex in extrema])

    # ensuring max_diff is not zero to avoid division by zero.
    if max_diff == 0:
        max_diff = 1

    # calculates a scaling factor based on the maximum difference value. This factor
    # is used to stretch the ELA image's pixel values across the full 0-255 range.
    scale = 255.0 / max_diff
    
    # enhances the brightness of the ELA image by applying the previously calculated 
    # scaling factor for making the manipulated regions stand out more distinctly.
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image







def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0









def detect_fake_or_not():
    # Load the image from the given path
    # img_path = 'D:/major_project/a/one/Identifying-Human-Edited-Images-master/data/real_and_fake_face/training_real1/real_00004.jpg'

    img_path = 'D:/major_project/a/one/Identifying-Human-Edited-Images-master/data/real_and_fake_face/training_real1/real_00008.jpg'

    img_path = 'C:/Users/01raw/Desktop/60df81b1bd2a392a1cf84fd0_ghost1.png'

    # img_path = 'D:/major_project/a/one/Identifying-Human-Edited-Images-master/data/real_and_fake_face/training_real1/real_00024.jpg'
    #img_path = 'D:/major_project/a/one/Identifying-Human-Edited-Images-master/data/real_and_fake_face/training_fake/easy_11_1111.jpg'
    #img_path = 'D:/major_project/a/one/Identifying-Human-Edited-Images-master/data/real_and_fake_face/training_fake/easy_9_1010.jpg'  # Replace with the actual image path

    # image = Image.open(img_path)





    # Preprocess the image using the same method you used for your dataset
    prepared_image = prepare_image(img_path)  # Assuming 'prepare_image' function is defined

    # Reshape the image to match the model's input shape
    prepared_image = prepared_image.reshape(-1, 128, 128, 3)

    # Make predictions using your trained model
    predictions = model.predict(prepared_image)

    # Assuming that your model is binary (two classes: fake and real)
    class_names = ['fake', 'real']
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]

    # Get the confidence score for the predicted class
    confidence_score = np.amax(predictions) * 100

    # Print the predicted class and confidence score
    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence Score: {confidence_score:.2f}%')


# detect_fake_or_not()

# INPUT IMAGE from frontend and detect whether it is fake or not
def detect_image(image):
    # Preprocess the image using the same method you used for your dataset
    prepared_image = prepare_image(image)  # Assuming 'prepare_image' function is defined

    # Reshape the image to match the model's input shape
    prepared_image = prepared_image.reshape(-1, 128, 128, 3)

    # Make predictions using your trained model
    predictions = model.predict(prepared_image)

    # Assuming that your model is binary (two classes: fake and real)
    class_names = ['fake', 'real']
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]

    # Get the confidence score for the predicted class
    confidence_score = np.amax(predictions) * 100

    # Print the predicted class and confidence score
    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence Score: {confidence_score:.2f}%')
    return {
        'imageClass' : predicted_class,
        'confidenceScore': confidence_score
    }