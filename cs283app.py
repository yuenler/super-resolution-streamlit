import streamlit as st
from PIL import Image


def autoencoder(image):
  import cv2
  import numpy as np
  import tensorflow as tf
  from tensorflow.keras import layers, models


  # read in image with opencv
  low_res_img = np.array(image)
  y_channel, i_channel, q_channel = cv2.split(cv2.cvtColor(low_res_img, cv2.COLOR_BGR2YCrCb))
  img = y_channel


  # get list of all patches
  patches = []
  window_size = 20
  for i in range(0,img.shape[0]-window_size):
      for j in range(0,img.shape[1]-window_size):
          patch = img[i:i+window_size,j:j+window_size]
          patches.append(patch)
  patches = np.array(patches)


  downsampled_patches = []
  for patch in patches:
      patch = cv2.resize(patch, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
      # now make it back to original size
      patch = cv2.resize(patch, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
      downsampled_patches.append(patch)

  downsampled_patches = np.array(downsampled_patches)


  input_shape = (window_size, window_size, 1)

  # Define the convolutional autoencoder model
  def autoencoder_model(input_shape):
      inputs = layers.Input(shape=input_shape)

      # Encoder
      encoded = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
      encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)
      encoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  # new layer

      # Decoder
      decoded = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(encoded)  # new layer
      decoded = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(decoded)
      decoded = layers.UpSampling2D((2, 2))(decoded)
      decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

      model = models.Model(inputs=inputs, outputs=decoded)
      return model


  def ssim_loss(y_true, y_pred):
      return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

  # Create the autoencoder model
  model = autoencoder_model(input_shape)

  # Compile the model with SSIM loss
  model.compile(optimizer='adam', loss=ssim_loss, metrics=['accuracy'])

  # Display the model architecture
  model.summary()

  downsampled_patches = downsampled_patches.reshape((-1, window_size, window_size, 1))
  patches = patches.reshape((-1, window_size, window_size, 1))

  # normalize data
  downsampled_patches = downsampled_patches / 255
  patches = patches / 255
  model.fit(downsampled_patches, patches, epochs=10, batch_size=32)

  # add a border of 5 pixels
  # Constants
  window = 5  # Original window size
  scale_factor = 4  # Scaling factor for the super-resolution
  stride = 3  # Stride for iteration

  # Adding border and initializing SR image
  img = cv2.copyMakeBorder(img, window, window, window, window, cv2.BORDER_CONSTANT, value=0)
  SR_image = np.zeros((img.shape[0]*scale_factor, img.shape[1]*scale_factor))

  # Initialize a list for each pixel to store values from different patches
  SR_image_accumulator = [[[] for _ in range(img.shape[1]*scale_factor)] for _ in range(img.shape[0]*scale_factor)]

  # Iterate through the image with a stride of 2 pixels
  for i in range(0, img.shape[0]-window, stride):
    for j in range(0, img.shape[1]-window, stride):
      patch = img[i:i+window, j:j+window]
      patch = cv2.resize(patch, (window*scale_factor, window*scale_factor), interpolation=cv2.INTER_CUBIC)
      patch = patch.reshape((1, window*scale_factor, window*scale_factor, 1))
      patch = patch / 255

      prediction = model.predict(patch)

      prediction = prediction.reshape((window*scale_factor, window*scale_factor))  
      prediction = prediction * 255

      # Append the values of the patch to the corresponding pixels
      for di in range(window*scale_factor):
          for dj in range(window*scale_factor):
              SR_image_accumulator[i*scale_factor+di][j*scale_factor+dj].append(prediction[di, dj])

  # Fill empty pixels with corresponding pixels from the upscaled low-res image
  upscaled_low_res_img = cv2.resize(img, (img.shape[1]*scale_factor, img.shape[0]*scale_factor), interpolation=cv2.INTER_CUBIC)
  for i in range(len(SR_image_accumulator)):
      for j in range(len(SR_image_accumulator[0])):
          if not SR_image_accumulator[i][j]:  # Check if the list is empty
              SR_image_accumulator[i][j].append(upscaled_low_res_img[i][j])

  # Compute the average for each pixel from its accumulated values
  for i in range(len(SR_image_accumulator)):
      for j in range(len(SR_image_accumulator[0])):
          SR_image[i][j] = sum(SR_image_accumulator[i][j]) / len(SR_image_accumulator[i][j])

  img = img[window:-window, window:-window]
  SR_image = SR_image[window_size:-window_size, window_size:-window_size]

  # reshape color channels to be the same size as the SR image
  i_channel_resized = cv2.resize(i_channel, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
  q_channel_resized = cv2.resize(q_channel, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

  stacked_channels = np.stack([SR_image, i_channel_resized, q_channel_resized], axis=2)

  if stacked_channels.dtype != np.uint8:
      stacked_channels = stacked_channels.astype(np.uint8)

  colored_SR_image = cv2.cvtColor(stacked_channels, cv2.COLOR_YCrCb2BGR)
  return colored_SR_image



def example(image):
  import cv2
  import numpy as np
  from sklearn.neighbors import NearestNeighbors
  def generate_patches(img, window_size, scale):
    window = window_size * scale
    patches = []
    downsampled_patches = []
    for i in range(0, img.shape[0] - window):
        for j in range(0, img.shape[1] - window):
            patch = img[i:i + window, j:j + window]
            downsampled_patch = cv2.resize(patch, (0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
            patches.append(patch.flatten())
            downsampled_patches.append(downsampled_patch.flatten())
    return np.array(patches, dtype=np.float32), np.array(downsampled_patches, dtype=np.float32)

  scale = 4
  window_size = 5

  low_res_img = np.array(image)
  y_channel, i_channel, q_channel = cv2.split(cv2.cvtColor(low_res_img, cv2.COLOR_BGR2YCrCb))
  img = y_channel

  patches_at_scale, downsampled_at_scale_patches = generate_patches(img, window_size, scale)
  patches_at_half_scale, downsampled_at_half_scale_patches = generate_patches(img, window_size, scale // 2)

  k_neighbors = 1
  nn_model_at_scale = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto')
  nn_model_at_scale.fit(downsampled_at_scale_patches)

  nn_model_at_half_scale = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto')
  nn_model_at_half_scale.fit(downsampled_at_half_scale_patches)

  def get_SR_patch(distances, indices, patch):
      if (min(distances) > 50):
          return None
      return patches_at_scale[indices][0].reshape(window_size*scale, window_size*scale)


  def get_SR_patch_at_half_scale(distances, indices, patch):
      if (min(distances) > 50):
          return None
      half_scale_patch = patches_at_half_scale[indices][0].reshape(window_size*scale//2, window_size*scale//2)
      return cv2.resize(half_scale_patch, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    
  stride = 2
  # Iterate through all patches
  img = cv2.copyMakeBorder(img, window_size, window_size, window_size, window_size, cv2.BORDER_CONSTANT, value=0)

  # Initialize a list for each pixel to store values and distances from different patches
  SR_image_accumulator = [[[] for _ in range(img.shape[1]*scale)] for _ in range(img.shape[0]*scale)]
  upscaled_low_res_img = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

  def process_patch(scale, window_size, nn_model, scale_amount):
      num = 0
      for i in range(0, img.shape[0]-window_size, stride):
          for j in range(0, img.shape[1]-window_size, stride):
              patch = img[i:i+window_size, j:j+window_size].astype(np.float32).reshape(1, -1)
              distances, indices = nn_model.kneighbors(patch)
              selected_patch = None
              if scale_amount == 'half':
                  selected_patch = get_SR_patch_at_half_scale(distances, indices, patch)
              else:
                  selected_patch = get_SR_patch(distances, indices, patch)

              if selected_patch is None:
                  continue

              num += 1
              min_distance = np.min(distances)

              # Append the values of the patch and the distance to the corresponding pixels
              for di in range(window_size*scale):
                  for dj in range(window_size*scale):
                      SR_image_accumulator[i*scale+di][j*scale+dj].append((selected_patch[di, dj], min_distance))

      return num

  num_x4 = process_patch(scale, window_size, nn_model_at_scale, 'full')
  print('Proportion of patches with match for x4 scale:', num_x4 / ((img.shape[0]-window_size)//2 * (img.shape[1]-window_size)//2))

  # Process for x2 scale
  num_x2 = process_patch(scale, window_size, nn_model_at_half_scale, 'half')
  print('Proportion of patches with match for x2 scale:', num_x2 / ((img.shape[0]-window_size)//2 * (img.shape[1]-window_size)//2))


  # Fill empty pixels with corresponding pixels from the upscaled low-res image
  for i in range(len(SR_image_accumulator)):
      for j in range(len(SR_image_accumulator[0])):
          if not SR_image_accumulator[i][j]:
              SR_image_accumulator[i][j].append((upscaled_low_res_img[i][j], float('inf')))

  # Select the pixel value with the lowest distance for each pixel
  SR_image = np.zeros((img.shape[0]*scale, img.shape[1]*scale), dtype=np.float32)
  for i in range(len(SR_image_accumulator)):
      for j in range(len(SR_image_accumulator[0])):
          if SR_image_accumulator[i][j]:
              # Sort the tuples based on the second element
              sorted_values = sorted(SR_image_accumulator[i][j], key=lambda x: x[1])

              # Calculate the index to slice the lowest 20%
              slice_index = max(1, int(len(sorted_values) * 0.2))

              # Select the lowest 20% of values and calculate the average of their first elements
              avg_value = sum(value[0] for value in sorted_values[:slice_index]) / slice_index

              SR_image[i][j] = avg_value

  img = img[window_size:-window_size, window_size:-window_size]
  SR_image = SR_image[window_size*scale:-window_size*scale, window_size*scale:-window_size*scale]

  i_channel_resized = cv2.resize(i_channel, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
  q_channel_resized = cv2.resize(q_channel, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
  stacked_channels = np.stack([SR_image, i_channel_resized, q_channel_resized], axis=2)

  if stacked_channels.dtype != np.uint8:
      stacked_channels = stacked_channels.astype(np.uint8)

  colored_SR_image = cv2.cvtColor(stacked_channels, cv2.COLOR_YCrCb2BGR)

  return colored_SR_image



# Function to handle the image processing (to be implemented by you)
def process_image(uploaded_image):
    # Replace these with your actual processing code
    example_based_sr_image = example(uploaded_image)
    autoencoder_sr_image = autoencoder(uploaded_image)
    return example_based_sr_image, autoencoder_sr_image

def main():
    st.title("Single-image Super Resolution")

    uploaded_file = st.file_uploader("Upload a low resolution image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if image.size[0] * image.size[1] > 25000:
            st.warning("The uploaded image is large and may take longer to process. Consider selecting a smaller image.")

        if st.button('Increase Resolution'):
            # Creating placeholders
            processing_text = st.empty()
            processing_text.text("Processing...")

            # Image processing
            example_sr_image, autoencoder_image = process_image(image)

            # Clear the processing text once done
            processing_text.empty()

            # Display the processed images
            st.image(example_sr_image, caption='Example-Based SR Algorithm Result (4x Resolution)', use_column_width=True)
            st.image(autoencoder_image, caption='Autoencoder Algorithm Result (4x Resolution)', use_column_width=True)

if __name__ == "__main__":
    main()