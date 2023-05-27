setwd("C:/Users/Noah/Desktop/DeepFish2/Segmentation")
remotes::install_github("r-tensorflow/unet")
install_tensorflow()
library(unet)
library(keras)
library(tfdatasets)
library(tidyverse)
library(rsample)
library(remotes)
library(reticulate)
library(abind)
library(tibble)
library(parallel)
library(doParallel)
library(foreach)
tf$constant("Hello Tensorflow")

images <- tibble(
  img = list.files(here::here("images/valid"), full.names = TRUE),
  mask = list.files(here::here("masks/valid"), full.names = TRUE)
) %>% 
  sample_n(2) %>% 
  map(. %>% magick::image_read() %>% magick::image_resize("128x128"))

out <- magick::image_append(c(
  magick::image_append(images$img, stack = TRUE), 
  magick::image_append(images$mask, stack = TRUE)
)
)
print(out)

data <- tibble(
  img = list.files(here::here("images/valid"), full.names = TRUE),
  mask = list.files(here::here("masks/valid"), full.names = TRUE)
)

data <- initial_split(data, prop = 0.8)

training_dataset <- training(data) %>%  
  tensor_slices_dataset() %>% 
  dataset_map(~.x %>% list_modify(
    # decode_jpeg yields a 3d tensor of shape (1280, 1918, 3)
    img = tf$image$decode_jpeg(tf$io$read_file(.x$img)),
    # decode_gif yields a 4d tensor of shape (1, 1280, 1918, 3),
    # so we remove the unneeded batch dimension and all but one 
    # of the 3 (identical) channels
    mask = tf$image$decode_gif(tf$io$read_file(.x$mask))[1,,,][,,1,drop=FALSE]
  ))

example <- training_dataset %>% as_iterator() %>% iter_next()
example

training_dataset <- training_dataset %>% 
  dataset_map(~.x %>% list_modify(
    img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
    mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float32)
  ))

training_dataset <- training_dataset %>% 
  dataset_map(~.x %>% list_modify(
    img = tf$image$resize(.x$img, size = shape(128, 128)),
    mask = tf$image$resize(.x$mask, size = shape(128, 128))
  ))
random_bsh <- function(img) {
  img %>% 
    tf$image$random_brightness(max_delta = 0.3) %>% 
    tf$image$random_contrast(lower = 0.5, upper = 0.7) %>% 
    tf$image$random_saturation(lower = 0.5, upper = 0.7) %>% 
    # make sure we still are between 0 and 1
    tf$clip_by_value(0, 1) 
}

training_dataset <- training_dataset %>% 
  dataset_map(~.x %>% list_modify(
    img = random_bsh(.x$img)
  ))

example <- training_dataset %>% as_iterator() %>% iter_next()
example$img %>% as.array() %>% as.raster() %>% plot()

create_dataset <- function(data, train, batch_size = 32L) {
  
  dataset <- data %>% 
    tensor_slices_dataset() %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img)),
      mask = tf$image$decode_gif(tf$io$read_file(.x$mask))[1,,,][,,1,drop=FALSE]
    )) %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
      mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float32)
    )) %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$resize(.x$img, size = shape(128, 128)),
      mask = tf$image$resize(.x$mask, size = shape(128, 128))
    ))
  
  # data augmentation performed on training set only
  if (train) {
    dataset <- dataset %>% 
      dataset_map(~.x %>% list_modify(
        img = random_bsh(.x$img)
      )) 
  }
  
  # shuffling on training set only
  if (train) {
    dataset <- dataset %>% 
      dataset_shuffle(buffer_size = batch_size*128)
  }
  
  # train in batches; batch size might need to be adapted depending on
  # available memory
  dataset <- dataset %>% 
    dataset_batch(batch_size)
  
  dataset %>% 
    # output needs to be unnamed
    dataset_map(unname) 
}

#create the training and test datasets
training_dataset <- create_dataset(training(data), train = TRUE)
validation_dataset <- create_dataset(testing(data), train = FALSE)

# U-net 128 -----------------------------------------------------

input_size <- 128

epochs <- 30
batch_size <- 16

orig_width <- 1918
orig_height <- 1280

threshold <- 0.5

train_samples <- 5088
train_index <- sample(1:train_samples, round(train_samples * 0.8)) # 80%
val_index <- c(1:train_samples)[-train_index]

images_dir <- "images/valid/" 
masks_dir <- "masks/valid/"

K <- backend()

#metrics
dice_coef <- custom_metric("dice", function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})

bce_dice_loss <- custom_metric("dice_loss", function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - dice_coef(y_true, y_pred))
  return(result)
})

get_unet_128 <- function(input_shape = c(128, 128, 3),
                         num_classes = 1) {
  
  inputs <- layer_input(shape = input_shape)
  # 128
  
  down1 <- inputs %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 64
  
  down2 <- down1_pool %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 32
  
  down3 <- down2_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 16
  
  down4 <- down3_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down4_pool <- down4 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 8
  
  center <- down4_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  # center
  
  up4 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 16
  
  up3 <- up4 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 32
  
  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 64
  
  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 128
  
  classify <- layer_conv_2d(up1,
                            filters = num_classes, 
                            kernel_size = c(1, 1),
                            activation = "sigmoid")
  
  
  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate  = 0.000001),
    loss = bce_dice_loss,
    metrics = list(metric_mean_iou(num_classes = 3), dice_coef, bce_dice_loss, metric_binary_accuracy)
  )
  
  return(model)
}

model25 <- get_unet_128()
model2 <- get_unet_128()
model3 <- get_unet_128()




model2 <- unet(input_shape = c(128, 128, 3))
summary(model25)




model2 %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 1e-3),
  loss = "binary_crossentropy",
  metrics = list(dice_coef, metric_binary_accuracy, bce_dice_loss)
)

?fit.keras.engine.training.Model


#Train Model
history2 <- fit(model3,
               training_dataset,
               epochs = 2,
               class_weight = as.list(1,
                                      1,
                                      1000),
               validation_data = validation_dataset,
               shuffle = TRUE,
               verbose = 1,
               initial_epoch = 150)
 plot(history)
 plot(history2)
 ?plot

save_model_hdf5(object = model175 ,filepath = "model_saved", include_optimizer = TRUE)
model200 <- load_model_hdf5(filepath = "model_saved", custom_objects = c(dice_coef, bce_dice_loss), compile = TRUE)

#predictions
batch <- training_dataset %>% as_iterator() %>% iter_next()
predictions <- predict(model200, batch[[1]], batch_size = 80)

images <- tibble(
  image = batch[[1]] %>% array_branch(1),
  predicted_mask = predictions[,,,1] %>% array_branch(1),
  mask = batch[[2]][,,,1]  %>% array_branch(1)
) %>% 
  sample_n(4) %>% 
  map_depth(2, function(x) {
    as.raster(x) %>% magick::image_read()
  }) %>% 
  map(~do.call(c, .x))


out <- magick::image_append(c(
  magick::image_append(images$mask, stack = TRUE),
  magick::image_append(images$image, stack = TRUE), 
  magick::image_append(images$predicted_mask, stack = TRUE)
)
)

plot(out)




