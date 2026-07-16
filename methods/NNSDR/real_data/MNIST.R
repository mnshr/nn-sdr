library(keras)

num_classes <- 10L
epochs <- 20L
batch_size <- 128L

################################################################################
###                     Loading & Prepair MNIST dataset                      ###
################################################################################

c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()

x_train <- array_reshape(x_train, c(nrow(x_train), prod(dim(x_train)[-1])))
x_test  <- array_reshape(x_test , c(nrow(x_test ), prod(dim(x_test )[-1])))

# x_train <- x_train / 255
# x_test  <- x_test  / 255
center <- apply(x_train, 2, mean)
x_train <- (x_train - center) / 128
x_test  <- (x_test  - center) / 128

y_train <- to_categorical(y_train, num_classes)
y_test  <- to_categorical(y_test , num_classes)

################################################################################
###                              Model Creation                              ###
################################################################################

model <- keras_model_sequential(name = 'base_model')
model %>%
    layer_dense(units = 256L, activation = 'relu',
                input_shape = ncol(x_train)) %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128L, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = num_classes, activation = 'softmax')

summary(model)

model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'RMSProp',
    metrics = c('accuracy')
)

################################################################################
###                           Base Model Training                            ###
################################################################################

history.base <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1L,
    validation_split = 0.1
)

plot(history.base)

score <- model %>% evaluate(x_test, y_test, verbose = 0L)

cat('Test loss:     ', score[['loss']],     '\n',
    'Test accuracy: ', score[['accuracy']], '\n', sep = '')

################################################################################
###                           OPG Data Reduction                            ###
################################################################################
library(tensorflow)
library(ggplot2)

G <- local({
    X = tf$cast(x_train, 'float32')
    with(tf$GradientTape() %as% tape, {
        tape$watch(X)
        Y <- model(X)
    })
    as.matrix(tape$gradient(Y, X))
})
eig <- eigen(var(G), symmetric = TRUE)
B.opg <- eig$vectors[, 1:2]

# ggplot(data.frame(values = eig$values[1:25]), aes(x = seq_along(values), y = values)) +
#     geom_line()

ggplot(data.frame(x_test %*% B.opg, y = factor(apply(y_test, 1, which.max))),
       aes(x = X1, y = X2, group = y, color = y)) +
    geom_point()

################################################################################
###                             Refinement Model                             ###
################################################################################
weights <- model$get_weights()

model.ref <- keras_model_sequential(name = 'Refinement')
model.ref %>%
    layer_dense(units = ncol(B.opg), activation = 'relu',
                input_shape = ncol(x_train), use_bias = FALSE,
                weights = list(B.opg)) %>%
    layer_dense(units = 256L, activation = 'relu',
                weights = list(crossprod(B.opg, weights[[1]]), weights[[2]])) %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128L, activation = 'relu',
                weights = weights[3:4]) %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = num_classes, activation = 'softmax',
                weights = weights[5:6])

summary(model.ref)

model.ref %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'RMSProp',
    metrics = c('accuracy')
)

################################################################################
###                        Refinement Model Training                         ###
################################################################################

history.ref <- model.ref %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1L,
    validation_split = 0.1
)

plot(history.ref)

score <- model.ref %>% evaluate(x_test, y_test, verbose = 0L)

cat('Test loss:     ', score[['loss']],     '\n',
    'Test accuracy: ', score[['accuracy']], '\n', sep = '')

### Combine Histories
hist <- structure(list(
    params = list(
        verbose = min(history.base$params$verbose, history.ref$params$verbose),
        epochs = history.base$params$epochs + history.ref$params$epochs,
        steps = max(history.base$params$steps, history.ref$params$steps)
    ),
    metrics = lapply(structure(names(history.base$metrics), names = names(history.base$metrics)),
        function(name) c(history.base$metrics[[name]], history.ref$metrics[[name]]))
), class = "keras_training_history")

plot(hist, smooth = FALSE)


B.ref <- model.ref$get_weights()[[1]]

ggplot(data.frame(x_test %*% B.ref, y = factor(apply(y_test, 1, which.max))),
       aes(x = X1, y = X2, group = y, color = y)) +
    geom_point()


B.pca <- eigen(var(x_train), symmetric = TRUE)$vectors[, 1:2]
ggplot(data.frame(x_test %*% B.pca, y = factor(apply(y_test, 1, which.max))),
       aes(x = X1, y = X2, group = y, color = y)) +
    geom_point()

image.ref <- matrix(((B.ref - min(B.ref)) / abs(diff(range(B.ref))))[, 2], 28, 28)
plot(c(0, 28), c(0, 28), type = "n", xlab = "", ylab = "")
rasterImage(image.ref, 0, 0, 28, 28, interpolate = TRUE)
