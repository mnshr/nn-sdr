library(ggplot2)

################################################################################
###                             General Helpers                              ###
################################################################################

# Read/Combine simulation data logs
read.logs <- function(pattern) {
    # Folder containing log files
    path <- if (file.exists('results/')) './' else './simulations/'
    path <- paste0(path, 'results/')
    # Read all log files and augment with meta-parameters
    file.names <- list.files(path, pattern, full.names = TRUE)
    sim <- do.call(rbind, lapply(file.names, function(path) {
        # Read simulation log (one file)
        sim <- read.csv(path, comment.char = '#')
        # Add simulation arguments as columns
        args <- Filter(function(line) startsWith(line, '#'), readLines(path))
        args <- sub('# ?', '', args)
        args <- regmatches(args, regexpr(' ', args), invert = TRUE)
        # Try to convert meta-parameters from string into int/num/bool/...
        for (arg in args) {
            val <- tryCatch(
                eval(parse(text = arg[2])),
                error = function(err) arg[2]
            )
            if (length(val) > 1) val <- paste(val, collapse = ',')
            sim[[arg[1]]] <- val
        }
        sim
    }))
    # Convert methods to factors
    sim$method <- factor(sim$method,
        levels = c("opg", "mave", "cve", "sir", "save", "phdy", "nn.opg", "nn.ref"),
        labels = c("OPG", "MAVE", "CVE", "SIR", "SAVE", "PHD",  "NN-OPG", "NN-Ref")
    )
    sim
}


################################################################################
###                                 Bit Data                                 ###
################################################################################

# Read/Combine big data simulation logs
sim <- read.logs('sim_big_.*csv')

# Compute repetition mean and standard deviation over replications
(aggr <- merge(
    aggregate(dist.subspace ~ dataset + n + p + method, sim, mean),
    aggregate(dist.subspace ~ dataset + n + p + method, sim, sd),
    by = c("dataset", "n", "p", "method"),
    suffixes = c(".mean", ".sd")
))

# plots and tables
ggplot(aggr, aes(x = n, y = dist.subspace.mean,
                 group = interaction(dataset, method),
                 color = method, linetype = dataset)) +
    geom_line() +
    geom_errorbar(aes(
        ymin = dist.subspace.mean - dist.subspace.sd,
        ymax = dist.subspace.mean + dist.subspace.sd
    ), width = 0.2) +
    scale_x_continuous(trans = 'log2')

################################################################################
###                             Binary Response                              ###
################################################################################

# Read/Combine binary data simulation logs
sim <- read.logs('sim_binary_[0-9_]*\\.csv')

# Aggregated Tables
aggr.formula <- cbind(dist.subspace, accuracy) ~ dataset + method
aggr <- merge(
    aggregate(aggr.formula, sim, mean,
              na.action = na.pass),
    aggregate(aggr.formula, sim, sd,
              na.action = na.pass),
    by = attr(terms(aggr.formula), "term.labels"),
    suffixes = c(".mean", ".sd")
)
print(aggr[with(aggr, order(dataset, dist.subspace.mean)), ], digits = 3)


# box-plot subspace comparison
ggplot(sim, aes(x = method, y = dist.subspace,
                group = interaction(dataset, method),
                color = dataset)) +
    geom_boxplot() +
    labs(title = "Sim. Binary", x = "Methods", y = "Subspace Dist.", color = "Datasets")
