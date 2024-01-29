#R code to import and prepare the EWCS dataset

ewcs=read.table("C:\\Users\\ultra\\Documents\\Machine_Learning_Project\\EWCS_2016.csv",sep=",",header=TRUE)

ewcs[,][ewcs[, ,] == -999] <- NA

kk=complete.cases(ewcs)

ewcs=ewcs[kk,]


#R code to import and prepare the student performance dataset

school1=read.delim("C:\\Users\\ultra\\Documents\\Machine_Learning_Project\\student-mat.csv",sep=";",header=TRUE)

school2=read.delim("C:\\Users\\ultra\\Documents\\Machine_Learning_Project\\student-por.csv",sep=";",header=TRUE)

schools=merge(school1,school2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))



#R code to import the bank marketing dataset

bank=read.delim("C:\\Users\\ultra\\Documents\\Machine_Learning_Project\\bank.csv",sep=";",header=TRUE)


#######################################
#PART ONE
#######################################

#Principal Component Analysis

library(ISLR)
apply(ewcs,2,mean)
pairs(~.,panel=panel.smooth,data=ewcs,main="Scatterplot")

pr.out=prcomp(ewcs,scale=TRUE)
pr.out$rotation
summary(pr.out)
biplot(pr.out, scale=0)
pr.out$sdev
pr.var=pr.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve
plot(pve, xlab = "Principal Component",ylab = "Proportion of Variance",ylim = c(0,1),type = 'b')
plot(cumsum(pve), xlab = "Principal Component",ylab = "Proportion of Variance",ylim = c(0,1),type = 'b')


#####################################################
#PART TWO
#####################################################

install.packages("dplyr")
install.packages("car")
install.packages("randomForest")
library(dplyr)
library(car)
library(MASS)
library(rpart)
library(tree)
library(randomForest)

dim(schools)

str(schools)

summary(schools)

sum(is.na(schools))

schools=dplyr::select(schools,age,Medu,Fedu,traveltime.x,studytime.x,failures.x,famrel.x,
                       freetime.x,goout.x,Dalc.x,Walc.x,health.x,absences.x,G1.x,G2.x,G3.x,
                       traveltime.y,studytime.y,failures.y,famrel.y,freetime.y,goout.y,Dalc.y,
                       Walc.y,health.y,absences.y,G1.y,G2.y,G3.y)

schools_data=dplyr::select(schools,-c(G1.x,G2.x,G1.y,G2.y))
dim(schools_data)

set.seed(1)
train_data=sample(1:nrow(schools_data),200)
schools_train=schools_data[train_data,]
dim(schools_train)
schools_test=schools_data[-train_data,]
dim(schools_test)
summary(schools_train)
cor(schools_train)

#Regression Tree - Mat Sample

set.seed(25)
schoolstree1=tree(G3.x~Medu+Fedu+traveltime.x+studytime.x+failures.x+famrel.x+freetime.x+goout.x+Dalc.x+Walc.x+health.x+absences.x,
                 schools_data,subset=train_data)
summary(schoolstree1)
plot(schoolstree1)
text(schoolstree1,pretty=0)

cv.schoolstree1=cv.tree(schoolstree1)
plot(cv.schoolstree1$size,cv.schoolstree1$dev,type='b')
schoolstree1.prune=prune.tree(schoolstree1,best=3)
plot(schoolstree1.prune)
text(schoolstree1.prune,pretty=0)
schools_test$G3.x=predict(schoolstree1.prune,newdata=schools_data[-train_data,])
schoolstest1=schools_data[-train_data,"G3.x"]
plot(schools_test$G3.x,schoolstest1)
abline(0,1)
mean((schools_test$G3.x-schoolstest1)^2)


#Regression Tree - Por Sample

set.seed(3)
schoolstree2=tree(G3.y~Medu+Fedu+traveltime.y+studytime.y+failures.y+famrel.y+freetime.y+goout.y+Dalc.y+Walc.y+
                   health.y+absences.y,schools_data,subset=train_data)
summary(schoolstree2)
plot(schoolstree2)
text(schoolstree2,pretty=0)

cv.schoolstree2=cv.tree(schoolstree2)
plot(cv.schoolstree2$size,cv.schoolstree2$dev,type='b')
schoolstree2.prune=prune.tree(schoolstree2,best=5)
plot(schoolstree2.prune)
text(schoolstree2.prune,pretty=0)
schools_test$G3.y=predict(schoolstree2.prune,newdata=schools_data[-train_data,])
schoolstest2=schools_data[-train_data,"G3.y"]
plot(schools_test$G3.y,schoolstest2)
abline(0,1)
mean((schools_test$G3.y-schoolstest2)^2)

#Random Forest - Mat Sample

set.seed(5)
fingrade1=randomForest(G3.x~Medu+Fedu+traveltime.x+studytime.x+failures.x+famrel.x+freetime.x+goout.x+Dalc.x+Walc.x+health.x+absences.x,
                       data=schools_train,importance=TRUE)
fingrade1
yhat.fingrade1=predict(fingrade1,newdata=schools_data[-train_data,])
mean((yhat.fingrade1-schoolstest1)^2)
importance(fingrade1)
varImpPlot(fingrade1)

#Random Forest - Por Sample

set.seed(30)
fingrade2=randomForest(G3.y~Medu+Fedu+traveltime.y+studytime.y+failures.y+famrel.y+freetime.y+goout.y+Dalc.y+Walc.y+
                        health.y+absences.y,data=schools_train,importance=TRUE)
fingrade2
yhat.fingrade2=predict(fingrade2,newdata=schools_data[-train_data,])
mean((yhat.fingrade2-schoolstest2)^2)
importance(fingrade2)
varImpPlot(fingrade2)


##########################################
#PART THREE
##########################################


install.packages("caret")
install.packages("descr")
library(lattice)
library(ggplot2)
library(caret)
library(descr)
library(tree)
dim(bank)
names(bank)
summary(bank)
sum(is.na(bank))

bank$age=as.numeric(bank$age)
bank$previous=as.numeric(bank$previous)
bank$pdays=as.numeric(bank$pdays)
bank$duration=as.numeric(bank$duration)
bank$campaign=as.numeric(bank$campaign)
bank$day=as.numeric(bank$day)
bank$balance=as.numeric(bank$balance)
bank$poutcome=as.factor(bank$poutcome)
bank$month=as.factor(bank$month)
bank$contact=as.factor(bank$contact)
bank$loan=as.factor(bank$loan)
bank$housing=as.factor(bank$housing)
bank$default=as.factor(bank$default)
bank$education=as.factor(bank$education)
bank$marital=as.factor(bank$marital)
bank$job=as.factor(bank$job)
bank$y=as.factor(bank$y)
summary(bank)

#Classification Tree

tree.bank=tree(y~.-duration,bank)
summary(tree.bank)
plot(tree.bank)
text(tree.bank,pretty=0)

set.seed(10)
bank.train=sample(1:nrow(bank),3000)
bank.test=bank[-bank.train,]

tree.bank=tree(y~.-duration,bank,subset=bank.train)
tree.bank.pred=predict(tree.bank,bank.test,type="class")
tab=table(tree.bank.pred,bank.test$y)
tab
correct=tab[1,1]+tab[2,2]
total=nrow(bank.test)
(acc=correct/total)

#Pruning Tree

set.seed(15)
cv.banks=cv.tree(tree.bank,FUN=prune.misclass)
names(cv.banks)
cv.banks
par(mfrow=c(1,2))
plot(cv.banks$size,cv.banks$dev,type="b")
plot(cv.banks$k,cv.banks$dev,type="b")

prune.banks=prune.misclass(tree.bank,best=2)
plot(prune.banks)
text(prune.banks,pretty=0)

tree.bank.pred=predict(prune.banks,bank.test,type="class")
tab=table(tree.bank.pred,bank.test$y)
tab
correct=tab[1,1]+tab[2,2]
total=nrow(bank.test)
(acc=correct/total)

#K-Nearest Neighbours

bank.train=bank[bank.train,]
bank.knn=train(y~.-duration,data=bank.train,method="knn",maximize=TRUE,trControl=trainControl(method="cv",number=10),
               preProcess=c("center","scale"))

predknn=predict(bank.knn,newdata=bank.test)
confusionMatrix(predknn,bank.test$y)

CrossTable(bank.test$y,predknn,prop.chisq=FALSE,prop.c=FALSE,prop.r=FALSE,dnn=c('actual default','predicted default'))
