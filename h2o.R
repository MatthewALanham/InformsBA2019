################################################################################
# UR4A Class: H2O Examples
################################################################################
# load package
library(h2o)

# start your h2o cluster
# By default, your h2o instance will be allowed to use all your cores and 
# 25% of your system memory unless you specify otherwise.
h2o.init()
h2o.clusterInfo()

# If you want to specify cores and memory to use you can like so:
h2o.shutdown()                          # shutdown your cluster if running
Y
h2o.init(nthreads=2, max_mem_size="4g") # specify what you want
h2o.clusterInfo()                       # inspect cluster info 

################################################################################
# Here is your web GUI "Flow" if you want to use it
# http://localhost:54321/flow/index.html

################################################################################
# The data set we want to investigate is online
# source: "Practical Machine Learning with H2O" O'Reilly book
datasets <- "https://raw.githubusercontent.com/DarrenCook/h2o/bk/datasets/"

# Create an H2O frame on the cluster which is like a data.frame in R. 
# It recognizes that the first line in the csv file was a header row, so it has 
# automatically named the columns. Also, recognizes that the class column is 
# categorical.
data <- h2o.importFile(paste0(datasets,"iris_wheader.csv"))
head(data)
str(data)

# prepare the data
y <- "class"                                # target variable to learn
x <- setdiff(names(data), y)                # features are all other columns
parts <- h2o.splitFrame(data, 0.8, seed=99) # randomly partition data into 80/20
train <- parts[[1]]                         # random set of training obs
test <- parts[[2]]                          # random set of testing obs

# notice in your Environment the class type of your "data" set
# It is an H2O data.frame, not an R data.frame
str(data)
# we could coerce it to an R data.frame like so:
d <- as.data.frame(data)
# However, notice the difference in the memory size of the data distrubuted
# on the H2O cluster versus what it what would be if it were on your physical
# machine
object.size(data)
object.size(d)

################################################################################
# train a model
################################################################################
# Here are some other models you might try
#h2o.randomForest    # random forest
#h2o.gbm             # gradient boosting machine
#h2o.glm             # generalized linear model (logit, multinomial logit)
#h2o.naiveBayes      # naive bayes
#h2o.stackedEnsemble # ensemble model
?h2o.randomForest

# train an rf model on the H2O training data.frame
rf <- h2o.randomForest(x, y, train)

# (Option 1: Default settings) train the model
dl <- h2o.deeplearning(x, y, train)
?h2o.deeplearning
# (Option 2: Custom settings) train the model (will only use one core)
#m <- h2o.deeplearning(x, y, train, seed=99, reproducible=T)

################################################################################
# examining the performance of the trained model
################################################################################
h2o.mse(rf)
h2o.confusionMatrix(rf)
h2o.confusionMatrix(dl)

# make predictions
p <- h2o.predict(dl, test)
# calling p will only show the first six predictions
# 1st column is predicted class
# remaining columns are are confidence/probabilities (each row should sum to 1)
p
# to see all predictions, you have to actually download them (be careful if
# data is large). 
myTestPreds <- as.data.frame(p)
# is myTestPreds an H2O frame or R data.frame?
str(myTestPreds)
str(p)

################################################################################
# Creating a new data frame in H2O versus on your local machine
# obtain predictions versus actuals in test set
################################################################################
# this way combines the records in the cluster to make a new H2O data frame in 
# the cluster called 'p2'. This does not download the data to your machine.
as.data.frame(p2 <- h2o.cbind(p$predict, test$class))
str(p2)

# predictions versus actuals in test set
# column from p is downloaded, column from test is downloaded, then combined in
# R's memory to make a data.frame. First option above usually better because
# you are not actually downloading the data to your machine
p2 <- cbind(as.data.frame(p$predict), as.data.frame(test$class))
str(p2)

################################################################################
# classification accuracy on test set
mean(p$predict == test$class)

# stats on test set
# the hit ratio section tells us the model was 
#   93.3% right when assigning to the most likely one class
#   100%  right if it could pick the top two most likely classes
#   100%  right if it could pick the top three classes (obviously this will 
#         always be 100% in this case as there are only 3 classes in total)
h2o.performance(dl, test)

# while caret was designed for binary classification and regression problems
# the confusionMatrix() function can also calculate stats for multiple classes
results <- cbind(as.data.frame(test$class),as.data.frame(p))
head(results)
library(caret)
(confusionMatrix(data=results$class, results$predict))

# My talk talk on caret at the 2018 INFORMS BA conference if you're interested
# https://github.com/MatthewALanham/informsba2018

################################################################################
# train with AutoML
auto <- h2o.automl(x, y, train, max_runtime_secs=30)
?h2o.automl
auto

################################################################################
# To get an estimate of model runtime (in days, seconds, hours) and not just
# percentage complete:
# (1) go to your flow - http://localhost:54321/flow/index.html
# (2) click Admin -> Jobs
# (3) click on the model currently running
# silly example of doing 10,000 trees for a random forest on the iris dataset
rf2 <- h2o.randomForest(x, y, train, ntrees=100000)

################################################################################
# save the model
setwd("C:\\Users\\Matthew A. Lanham\\Dropbox\\_Conferences\\_Talks this year\\2019 INFORMS BA (Austin, TX)\\INFORMS BA h2o\\")
model_path <- h2o.saveModel(object=rf, path=getwd(), force=TRUE)
print(model_path)

# load the model
mattsModel <- h2o.loadModel(model_path)
str(mattsModel)
################################################################################
# shutdown your cluster
h2o.shutdown()
Y

