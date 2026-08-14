
#' Chemical ingredients of wine dataset
#'
#' @docType data
#'
#' @description
#' This dataset is about chemical analysis of 178 wines in a particular region of Italy. There are three cultivars with 59, 71, and 48, respectively.
#' @usage wine
#'
#' @format
#' A data frame with 178 observations on 14 variables.
#'
#' Attributes
#' 1. cultivar
#' 2. alcohol
#' 3. malic acid
#' 4. ash
#' 5. alcalinity of ash
#' 6. magnesium
#' 7. total phenols
#' 8. flavanoids
#' 9. nonflavanoid phenols
#' 10. oroanthocyanins
#' 11. color intensity
#' 12. hue
#' 13. OD280/OD315 of diluted wines
#' 14. proline.
#'
#' @keywords datasets
#'
#' @references Forina, M., Leardi, R., Armanino, C., Lanteri, S., Conti, P., & Princi, P. (1988). PARVUS: An extendable package of programs for data exploration, classification and correlation. Journal of Chemometrics, 4(2), 191-193..
#'
#' @source https://archive.ics.uci.edu/ml/datasets/wine

"wine"




#' Pen-Based Recognition of Handwritten Digits Data Set (training dataset)
#'
#' @docType data
#'
#' @description
#' The data is about the recognition of handwritten numbers from 0 to 9.
#' There are 30 writers in the training dataset and each participant are asked to write 250 digits in random order.
#' Without missing data, this dataset has 7494 observations.
#' The experiment uses WACOM tablet, which has 500 x 500 pixel resolutions and normalized it to a maximum scale of 100.
#' The researcher considers spatial resampling.
#' Thus, for each digit, eight pairs of 2 dimensional (x axis and y axis) locations are recorded, which makes this dataset have 16 dimensional predictor variables.
#'
#' @usage pendigits.tra
#'
#' @format
#' A data frame with 2219 obsevations on 17 variables.
#'
#' The first column to the 16th column represent resampled values of the pairs of points.
#' The 17th column presents the digit (0 to 9).
#'
#' @keywords datasets
#'
#' @references F. Alimoglu (1996) Combining Multiple Classifiers for Pen-Based Handwritten Digit Recognition, MSc Thesis, Institute of Graduate Studies in Science and Engineering, Bogazici University.
#'            F. Alimoglu, E. Alpaydin, "Methods of Combining Multiple Classifiers Based on Different Representations for Pen-based Handwriting Recognition," Proceedings of the Fifth Turkish Artificial Intelligence and Artificial Neural Networks Symposium (TAINN 96), June 1996, Istanbul, Turkey.
#'
#' @source https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits

"pendigits.tra"




#' Pen-Based Recognition of Handwritten Digits Data Set (testing dataset)
#'
#' @docType data
#'
#' @description
#' The data is about the recognition of handwritten numbers from 0 to 9.
#' There are 30 writers in the training dataset and each participant are asked to write 250 digits in random order.
#' Without missing data, this dataset has 7494 observations.
#' The experiment uses WACOM tablet, which has 500 x 500 pixel resolutions and normalized it to a maximum scale of 100.
#' The researcher considers spatial resampling.
#' Thus, for each digit, eight pairs of 2 dimensional (x axis and y axis) locations are recorded, which makes this dataset have 16 dimensional predictor variables.
#'
#' @usage pendigits.tes
#'
#' @format
#' A data frame with 2219 observations on 17 variables.
#'
#' The first column to the 16th column represent resampled values of the pairs of points.
#' The 17th column presents the digit (0 to 9).
#'
#' @keywords datasets
#'
#' @references F. Alimoglu (1996) Combining Multiple Classifiers for Pen-Based Handwritten Digit Recognition, MSc Thesis, Institute of Graduate Studies in Science and Engineering, Bogazici University.
#'            F. Alimoglu, E. Alpaydin, "Methods of Combining Multiple Classifiers Based on Different Representations for Pen-based Handwriting Recognition," Proceedings of the Fifth Turkish Artificial Intelligence and Artificial Neural Networks Symposium (TAINN 96), June 1996, Istanbul, Turkey.
#'
#' @source https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits

"pendigits.tes"

