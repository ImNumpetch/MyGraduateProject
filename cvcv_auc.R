rm(list=ls())
cat("\f")
library(compiler)
library(glmnet)
library(pROC)
require(caret)
library(preprocessCore)
require(MLmetrics)

# create pairwise sample
pairwise_sample <- function(X, y){
  #Input: Matrix X[1..nsample, 1..d] and column vector y[1..nsample]
  #Output: Matrix Z[1..p*n, 1..d]
  d = ncol(X)                      # number of variables 
  p = sum(y==1)                    # number of positive samples = sum(ytrain)    
  n = sum(y==0)                    # number of negative samples
  Z = matrix(0, p*n, d)            # initialization matrix Z with p*n rows and d col
  Xpos = X[(which(y==1)),]         # input samples of positive class
  Xneg = X[(which(y==0)),]         # input samples of negative class
  k = 1;                           # row index of Z
  for (i in 1:floor(p/2)) {        # 
    for (j in 1:n) {               # create pairwise input of negative class
      Z[k,] = -Xpos[i,] + Xneg[j,] # Z[1..floor(p/2)*n, 1..d]
      k = k+1                      # increase row index of Z
    }                              #
  }                                #
  int_n = floor(p/2) + 1           # create pairwise input of positive class
  for (i in int_n:p) {             # Z[floor(p/2)*n..p*n, 1..d]
    for (j in 1:n) {               # 
      Z[k,] = Xpos[i,] - Xneg[j,]  #
      k = k+1                      # increase row index of Z
    }                              #
  }                                #
  return(Z)                        # return pairwise input
}
pairwise_sample = cmpfun(pairwise_sample)

# perform nested cross-validation
cvcv_pairwise <- function(X, y, nfolds, alhpa){
  #Input: Matrix X[1..nsample, 1..d], column vector y[1..nsample] and nfolds
  #Output: AUC and Cross Entropy
  set.seed(ntime)                                        # duplicate
  cvSplits <- createFolds(y, nfolds, returnTrain = TRUE) # duplicate
  grid = 10^seq(10,-2,length=100)                        # duplicate
  nvar = rep(0, nfolds)                                  # duplicate
  auc = rep(0, nfolds)                                   # duplicate
  acc = rep(0, nfolds)                                   # duplicate
  acc2 = rep(0, nfolds)                                  # duplicate
  ce = rep(0, nfolds)                                    # duplicate
  ce2 = rep(0, nfolds)                                   # duplicate
  for(i in 1:nfolds){
    Z = pairwise_sample(X[cvSplits[[i]],],y[cvSplits[[i]]]) # create training input
    ytrain = y[cvSplits[[i]]]                            # create raw training output
    p = sum(ytrain==1)                                   # number of positive raw samples = sum(ytrain)
    n = sum(ytrain==0)                                   # number of negative raw samples
    o = data=c(array(0,round(p/2)*n),array(1,p*n-round(p/2)*n)) # create training output
    # perform cross-validate regularized logistic regression without intercept
    cvob = cv.glmnet(Z,o,standardize = FALSE,family="binomial",intercept=FALSE,alpha=alpha,nfolds=nfolds)
    pred = predict(cvob,X[-cvSplits[[i]],],s="lambda.1se",standardize=FALSE,intercept=FALSE,type="response")
    auc[i] = auc(y[-cvSplits[[i]]], as.vector(pred))     # duplicate
    acc[i] = mean((round(pred) == y[-cvSplits[[i]]]))    # duplicate
    ce[i] = LogLoss(pred, as.numeric(y[-cvSplits[[i]]])) # duplicate
    if (length(dim(predict(cvob,type="nonzero")))==0){   # duplicate
      nvar[i] = 0;                                       # duplicate
    } else {                                             # duplicate
      nvar[i] = nrow(predict(cvob,type="nonzero"))       # duplicate
    }                                                    # duplicate
    
    # Step 2
    coef = predict(cvob,type="coefficients",s="lambda.1se")
    v1 = X[cvSplits[[i]], ]%*%coef[-1]                   # create new training input
    v1 = cbind(v1, 0)                                    # append input for glmnet that requrie min 2 vars

    # perform cross-validate regularized logistic regression with intercept
    cvob2 = cv.glmnet(v1,y[cvSplits[[i]]],standardize=FALSE,family="binomial",alpha=alpha,nfolds=nfolds)
    
    v2 = X[-cvSplits[[i]], ]%*%coef[-1]                  # create new testing input
    v2 = cbind(v2, 0)                                    # append input for glmnet that requrie min 2 vars
    pred2 = predict(cvob2,v2,s="lambda.1se",standardize=FALSE,type="response")
    acc2[i] = mean((round(pred2) == y[-cvSplits[[i]]]))  # duplicate
    ce2[i] = LogLoss(pred2, as.numeric(y[-cvSplits[[i]]])) # duplicate
  }
  out = c(mean(auc), mean(acc), mean(acc2), mean(ce), mean(ce2), mean(nvar)) # duplicate
  return(out)                                            # return AUC ACC CE and #var
}
cvcv_pairwise = cmpfun(cvcv_pairwise)

# perform nested cross-validation
cvcv <- function(X, y, nfolds, alpha){
  #Input: Matrix X[1..nsample, 1..d], column vector y[1..nsample] and nfolds
  #Output: AUC and Cross Entropy
  nvar = rep(0, nfolds)                                  # duplicate
  auc = rep(0, nfolds)                                   # duplicate
  acc = rep(0, nfolds)                                   # duplicate
  ce = rep(0, nfolds)                                    # duplicate
  set.seed(ntime)                                        # duplicate
  cvSplits <- createFolds(y, nfolds, returnTrain = TRUE) # duplicate
  grid=10^seq(10,-3,length=100)                          # duplicate
  for(i in 1:nfolds){
    # perform cross-validate regularized logistic regression with intercept
    cvob = cv.glmnet(X[cvSplits[[i]],],y[cvSplits[[i]]],standardize=FALSE,family="binomial",alpha=alpha,nfolds=nfolds)
    pred = predict(cvob,X[-cvSplits[[i]],],s="lambda.1se",standardize=FALSE,type="response")
    auc[i] = auc(y[-cvSplits[[i]]], as.vector(pred))     # duplicate
    acc[i] = mean((round(pred) == y[-cvSplits[[i]]]))    # duplicate
    ce[i] = LogLoss(pred, as.numeric(y[-cvSplits[[i]]])) # duplicate
    if (length(dim(predict(cvob,type="nonzero")))==0){   # duplicate
      nvar[i] = 0;                                       # duplicate
    } else {                                             # duplicate
      nvar[i] = nrow(predict(cvob,type="nonzero"))       # duplicate
    }                                                    # duplicate
  }
  out = c(mean(auc), mean(acc), mean(ce), mean(nvar))    # duplicate
  return(out)                                            # return AUC ACC CE and #var
}
cvcv = cmpfun(cvcv)

# perform nested cross-validation
cvcv_lda <- function(X, y, nfolds, alpha){
  #Input: Matrix X[1..nsample, 1..d], column vector y[1..nsample] and nfolds
  #Output: AUC and Cross Entropy
  nvar = rep(0, nfolds)                                  # duplicate
  auc = rep(0, nfolds)                                   # duplicate
  acc = rep(0, nfolds)                                   # duplicate
  ce = rep(0, nfolds)                                    # duplicate
  set.seed(ntime)                                        # duplicate
  cvSplits <- createFolds(y, nfolds, returnTrain = TRUE) # duplicate
  grid=10^seq(10,-2,length=100)                          # duplicate
  for(i in 1:nfolds){
    Xtrain = X[cvSplits[[i]],]                           # create training input
    ytrain = y[cvSplits[[i]]]                            # create training output
    Xpos = Xtrain[which(ytrain == 1),]                   # create training positive class input 
    Xneg = Xtrain[which(ytrain == 0),]                   # create training negative class input
    mu = colMeans(Xpos) - colMeans(Xneg)                 # calculate pairwise mean
    sigma = cov(Xpos) + cov(Xneg)                        # claculate covariance
    # perform cross-validate regularized linear regression without intercept
    cvob = cv.glmnet(sigma,mu,standardize=FALSE,family="gaussian",intercept=FALSE,alpha=alpha,nfolds=nfolds)
    pred = predict(cvob,X[-cvSplits[[i]],],s="lambda.1se",standardize=FALSE,type="response")
    auc[i] = auc(y[-cvSplits[[i]]], as.vector(pred))     # duplicate
    acc[i] = mean((round(pred) == y[-cvSplits[[i]]]))    # duplicate
    ce[i] = LogLoss(pred, as.numeric(y[-cvSplits[[i]]])) # duplicate
    if (length(dim(predict(cvob,type="nonzero")))==0){   # duplicate
      nvar[i] = 0;                                       # duplicate
    } else {                                             # duplicate
      nvar[i] = nrow(predict(cvob,type="nonzero"))       # duplicate
    }                                                    # duplicate
  }
  out = c(mean(auc), mean(acc), mean(ce), mean(nvar))    # duplicate
  return(out)                                            # return AUC ACC CE and #var
}
cvcv_lda = cmpfun(cvcv_lda)

#load("~/Dropbox/3 work with kampol/Rcode/colon.rda")
#X = colon.x
#y = colon.y
#y = as.character(y)
load("~/Dropbox/3 work with kampol/Rcode/leukemia.rda")
X = lx.original
y = ly.original
y = as.character(y)
#load("~/Dropbox/3 work with kampol/Rcode/prostate.rda")
#X = prostate.x
#y = prostate.y
#y = as.character(y)
#load("~/Dropbox/3 work with kampol/Rcode/lymphoma.rda")
#X = lymphoma.x
#y = lymphoma.y
#y[c(which(y != 2))]=0
#y[c(which(y == 2))]=1
#y = as.character(y)
#load("~/Dropbox/3 work with kampol/Rcode/srbct.rda")
#X = srbct.x
#y = srbct.y
#y[c(which(y != 3))]=0
#y[c(which(y == 3))]=1
#y = as.character(y)
#load("~/Dropbox/3 work with kampol/Rcode/brain.rda")
#X = brain.x
#y = brain.y
#y[c(which(y != 3))]=0
#y[c(which(y == 3))]=1
#y = as.character(y)
#load("~/Dropbox/3 work with kampol/Rcode/west.RData")
#X = west$x
#X = as.matrix(X)
#y = west$y
#y = as.character(y)
#y[which(y=="negative")] <- "0"
#y[which(y=="positive")] <- "1"
#load("~/Dropbox/3 work with kampol/Rcode/shipp.RData")
#X = shipp$x
#X = as.matrix(X)
#y = shipp$y
#y = as.character(y)
#y[which(y=="DLBCL")] <- "0"
#y[which(y=="FL")] <- "1"

nfolds = 10   # number of fold in cross-validation
ntimes = 100 # number of repeat experiment
auc = Matrix(data=NA, nrow=ntimes, ncol=14)
alpha = 1
for (i in 1:ntimes){
  ntime = i
  print(i)
  #print('AUC Logistic')
  auc[i,1:6] = cvcv_pairwise(X, y, nfolds, alpha)
  #print('Logistic')
  auc[i,7:10] = cvcv(X, y, nfolds, alpha)
  #print('AUC LDA')
  auc[i,11:14] = cvcv_lda(X, y, nfolds, alpha)
  print(auc[i,],digits=3)
}
#write.csv(as.numeric(auc),"~/Dropbox/3 work with kampol/Rcode/colon10.csv")