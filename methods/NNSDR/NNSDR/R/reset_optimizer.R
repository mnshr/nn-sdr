
#' Reset TensorFlow optimizer.
#'
#' @param optimizer a \pkg{tensorflow} optimizer instance
#'
#' @examples
#' # Create example toy data
#'
#' # library(tensorflow) # v2
#' K <- tf$keras
#' model <- K$models$Sequential(list(
#'     K$layers$Dense(units = 7L, input_shape = list(3L)),
#'     K$layers$Dense(units = 1L)
#' ))
#' model$compile(loss = 'MSE', optimizer = 'RMSprop')
#'
#' \donttest{
#' model$fit(input) # Fit the model
#' }
#'
#' reinitialize_weights(model)
#' reset_optimizer(model$optimizer)
#'
#' \donttest{
#' model$fit(input) # Fit the model again completely independent of the first fit.
#' }
#'
#' @note Works for Adam, RMSprop properly (other optimizes are not tested!)
#' @note see source and search for `_create_slots` and `add_slot`.
#' @seealso https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/optimizer_v2.py
#' @seealso https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/rmsprop.py
#' @seealso https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/adam.py
#'
#' @export
reset_optimizer <- function(optimizer) {
    for (var in optimizer$variables()) {
        var$assign(tf$zeros_like(var))
    }
}
