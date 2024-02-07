library("tidyverse")

#Functions
vectorToDf <- function(meanVector, seVector) {
  df <- data.frame("Bin" = rep(1:2, each = 2), "Action" = c("Disengage", "Engage"),
                   "Action_Counter" = c(meanVector[c(2, 5)], meanVector[c(3, 6)]),
                   "se" = c(seVector[c(2, 5)], seVector[c(3, 6)]))
  return(df)
}

ggBarPlot <- function(df1) {
  plot <- ggplot2::ggplot(data = df1, mapping = ggplot2::aes(x = as.factor(Bin), y = mean_proportion, fill = Action)) + 
    ggplot2::geom_bar(stat = "identity", position = "dodge") +
    ylab("%Regulation Choices") +
    scale_x_discrete(breaks = c(2, 3), labels = c("Low", "High"), name = "Stimulus Intensity") +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = mean_proportion - 1.96 * se_proportion, ymax = mean_proportion + 1.96 * se_proportion),
                           position = "dodge", size = 1) +
    jaspGraphs::themeJaspRaw() +
    jaspGraphs::geom_rangeframe() +
    theme(legend.position = "right")
  return(plot)
}

ggPointPlot <- function(df1, noYAxis = FALSE) {
  axisPos <- if (noYAxis) "b" else "bl"
  plot <- ggplot2::ggplot(data = df1, mapping = ggplot2::aes(x = as.factor(Bin), y = mean_proportion, color = Action)) + 
    ggplot2::geom_point(stat = "identity", size = 4) +
    ylab("%Regulation Choices") +
    scale_x_discrete(breaks = c(2, 3), labels = c("Low", "High"), name = "Stimulus Intensity") +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = mean_proportion - 1.96 * se_proportion, ymax = mean_proportion + 1.96 * se_proportion),
                           size = 1.5) +
    jaspGraphs::themeJaspRaw() +
    jaspGraphs::geom_rangeframe(sides = axisPos) +
    theme(legend.position = "right")
  return(plot)
}

.readAll <- function(all_file_names, path) {
  all_datasets <- list()
  for (i in 1:length(all_file_names)) {
    file_location <- paste0(path, all_file_names[i])
    dataset <- as.data.frame(read.csv(file_location))
    all_datasets[[i]] <- dataset[,2:3]
  }
  return(all_datasets)
}

.getProportions <- function(all_datasets) {
  actionsVector <- lapply(all_datasets, c, recursive=TRUE)
  actionsVectorWide <- do.call(cbind, actionsVector)
  actionsDf <- as.data.frame(actionsVectorWide) %>% 
    cbind(Bin = rep(1:3, 2), Action = rep(c("Distraction", "Reappraisal"), each = 3)) %>% 
    filter(Bin != 1) %>%
    pivot_longer(cols = 1:(ncol(.)-2)) %>% 
    group_by(Bin, name) %>% 
    mutate(Proportion = value/sum(value) * 100) %>%
    ungroup() %>% 
    group_by(Action, Bin) %>% 
    summarise(mean_proportion = mean(Proportion), se_proportion = sd(Proportion)/sqrt(50)) %>% 
    ungroup()
  return(actionsDf)
}


# Full theory
path_ft <- "../datasets/Reference_Phenomenon/"
all_file_names_ft <- list.files(path_ft)

all_datasets_ft <- .readAll(all_file_names_ft, path_ft)

#find out how many time the agent learned something not in line with the hypothesis
unlist_and_compare_to_reference <- function(x){
  x <- unlist(x) 
  return(x[2] > x[5] &&  x[3] < x[6])
}

lapply(all_datasets_ft, unlist_and_compare_to_reference) %>% 
  unlist() %>% 
  sum()

unlist(all_datasets_ft[1])

actionsDf_ft <- .getProportions(all_datasets_ft)

png("FullTheoryResults_revised.png", height = 600, width = 800)
ggPointPlot(actionsDf_ft)
dev.off()

# No long-term differences
path_nlt <- "../datasets/Theory_no_LT/"
all_file_names_nlt <- list.files(path_nlt)

all_datasets_nlt <- .readAll(all_file_names_nlt, path_nlt)

actionsDf_nlt <- .getProportions(all_datasets_nlt)

# png("noLTTheoryResults.png", height = 600, width = 800)
# ggBarPlot(actionsDf_nlt)
# dev.off()

noLTplot <- ggPointPlot(actionsDf_nlt)


# No short-term differences
path_nst <- "../datasets/Theory_no_ST/"
all_file_names_nst <- list.files(path_nst)

all_datasets_nst <- .readAll(all_file_names_nst, path_nst)

actionsDf_nst <- .getProportions(all_datasets_nst)

# png("noSTTheoryResults.png", height = 600, width = 800)
# ggBarPlot(actionsDf_nst)
# dev.off()

noSTplot <- ggPointPlot(actionsDf_nst, noYAxis = T)


noLTplot <- noLTplot + scale_y_continuous(limits = c(0, 100)) + theme(legend.position = "none") + ggtitle("No Long-Term Differences")
noSTplot <- noSTplot + ggtitle("No Short-Term Differences") + theme(axis.text.y = element_blank(),
                                                                    axis.title.y = element_blank(),
                                                                    axis.ticks.y = element_blank())

png("reducedModelMatrixPlot.png", height = 600, width = 800)
jaspGraphs::ggMatrixPlot(plotList = matrix(list(noLTplot, noSTplot), ncol = 2))
dev.off()

# Example extension

path_ee <- "../datasets/Example_extension/"
all_file_names_ee <- list.files(path_ee)

all_datasets_ee <- .readAll(all_file_names_ee, path_ee)

actionsDf_ee <- .getProportions(all_datasets_ee)
 
# png("exampleExtensionResults.png", height = 600, width = 800)
# ggBarPlot(actionsDf_ee)
# dev.off()


png("exampleExtensionResults_revised.png", height = 600, width = 800)
ggPointPlot(actionsDf_ee)
dev.off()


