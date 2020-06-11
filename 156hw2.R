library(randomForest)
library(MASS)
library(data.table)
library(mltools)
# gunzip the files
R.utils::gunzip("train-images-idx3-ubyte.gz")
R.utils::gunzip("train-labels-idx1-ubyte.gz")
R.utils::gunzip("t10k-images-idx3-ubyte.gz")
R.utils::gunzip("t10k-labels-idx1-ubyte.gz")

# helper function for visualization
show_digit = function(arr784, col = gray(12:1 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}

# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# load images
train = load_image_file("train-images-idx3-ubyte")
test  = load_image_file("t10k-images-idx3-ubyte")

# load labels
trainlabel = load_label_file("train-labels-idx1-ubyte")
testlabel  = load_label_file("t10k-labels-idx1-ubyte")

# load labels
ytrain = matrix(, nrow = 60000, ncol = 10)
ytest = matrix(,nrow=10000,ncol = 10)
count = 0
for (val in trainlabel)
{
  count = count + 1
  ytrain[count,] = diag(10)[val+1,]
}
count = 0
for (val in testlabel)
{
  count = count + 1
  ytest[count,] = diag(10)[val+1,]
}

# view test image
show_digit(train1[2, ])

resultlabel = vector()
#linear regression
mydata <- data.frame(cbind(train,ytrain))
testdata <-data.frame(cbind(test,ytest))

fitLR = lm(ytrain~., data = mydata)
test_predLR = predict(fitLR, testdata)
for (i in 1:10000)
{
  resultlabel[i] = which.max(test_predLR[i,])-1
}

mean(resultlabel == testlabel)
table(predictedLR = resultlabel, actual = testlabel)


#logit
fitLG = glm(ytrain~.,data= mydata[1:1000,], family="binomial")
fitLG$confusion
test_predLG = predict(fitLG,testdata)
test_predLG = round(test_predLG, digits=0)
mean(test_predLG == test1$y)

#LDA
fitLDA = lda(y~.,data=train1[1:1000,])
fitLDA$confusion



# testing classification on subset of training data
fitRanF= randomForest::randomForest(y ~ ., data = train1[1:1000,])
fitRanF$confusion
test_predRanF = predict(fitRanF, test)
mean(test_predRanF == test$y)
table(predictedRanF = test_predRanF, actual = test$y)