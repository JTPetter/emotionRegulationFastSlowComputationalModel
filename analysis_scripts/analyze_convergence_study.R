library("tidyverse")


# load data 
path <- "../datasets/Manuscript/convergence_study/"
all_file_names <- list.files(path)

all_datasets <- list()
for (i in 1:length(all_file_names)) {
  file_location <- paste0(path, all_file_names[i])
  dataset <- as.data.frame(read.csv(file_location))
  dataset <- dataset["Q_delta"] # only need this col
  all_datasets[[i]] <- dataset
}

# arrange as data.frame
all_dfs <- all_datasets[[1]]
for(i in 2:length(all_datasets)) {
  all_dfs <- cbind(all_dfs, all_datasets[[i]]$Q_delta)  
}

#only absolute values 
all_dfs <- abs(all_dfs)
#take the means to reduce size
all_dfMeans <- data.frame(Run = 1:1e5, delta = rowMeans(all_dfs[,-1]))

#create plot
xBreaks <- pretty(1:100000)
xLabels <- format(xBreaks, scientific = FALSE) 
yLimits <- c(0, 10)
yBreaks <- pretty(0:10)
ggplot() +
  geom_smooth(data = all_dfMeans, mapping = aes(y = delta, x = Run)) +
  scale_x_continuous(breaks = xBreaks, labels = xLabels, name = "Simulation Runs") +
  scale_y_continuous(limits = yLimits, breaks = yBreaks, name = "Q-Table Delta") +
  jaspGraphs::geom_rangeframe() +
  jaspGraphs::themeJaspRaw()





