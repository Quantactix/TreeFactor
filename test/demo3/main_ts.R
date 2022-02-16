library(TreeFactor)
library(rpart)
library(ranger)

tf_residual = function(fit,Y,Z,H,months,no_H){
  # Tree Factor Models
  regressor = Z
  for(j in 1:dim(Z)[2])
  {
    regressor[,j] = Z[,j] * fit$ft[months + 1]
  }
  if(!no_H)
  {
    regressor = cbind(regressor, H)
  }
  # print(fit$R2*100)
  x <- as.matrix(regressor)
  y <- Y
  b_tf = solve(t(x)%*%x)%*%t(x)%*%y
  haty <- (x%*%b_tf)[,1]
  print(b_tf)
  return(Y-haty)
}

###### parameters #####

start = 1
split = 80
end   = 100

case='demo' 
max_depth=4
min_leaf_size = 10
max_depth_boosting = 3
num_iter = 1000
num_cutpoints = 3
equal_weight = TRUE
no_H = TRUE
abs_normalize = TRUE
weighted_loss = FALSE
stop_no_gain = FALSE
nu = 1

# this tiny regularization ensures the matrix inversion
# penalty for the sigma (sigma + lambda I)^{-1} * mu
lambda = 1e-3
eta=1

##### load data #####

load("../../data/simu_data.rda")
print(names(da))

data <- da
data['lag_me'] = 1

tmp = data[,c('id', 'date','xret','lag_me', 
              # 5
              'c1', 'c2', 'c3', 'c4', 'c5', # 9
              'm1','m2', # 11
              'mkt' # 12
              )]
data = tmp
rm(tmp)
rm(da)

# chars

all_chars <- names(data)[c(5:9)]
top5chars <- c(1:5)
instruments = all_chars[top5chars]
splitting_chars <- all_chars

chars_and_macro <- append(splitting_chars,
  c("m1","m2","mkt"))

first_split_var = c(6:8)-1 # 2 macro variables
second_split_var = c(1:5)-1
third_split_var = c(1:5)-1
deep_split_var = 1:5

##### train-test split #####

data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]
data2 <- data[(data[,c('date')]>split) & (data[,c('date')]<=end), ]

# rm(data)

###### train data for all boosting steps #####
X_train = data1[,chars_and_macro]
Y_train = data1[,c("xret")]
R_train = data1[,c("xret")]
months_train = as.numeric(as.factor(data1[,c("date")]))
months_train = months_train - 1 # start from 0
stocks_train = as.numeric(as.factor(data1[,c("id")])) - 1
Z_train = data1[, instruments]
Z_train = cbind(1, Z_train)
H_train = data1[,c("mkt")]
H_train = H_train * Z_train
portfolio_weight_train = data1[,c("lag_me")]
loss_weight_train = data1[,c("lag_me")]
num_months = length(unique(months_train))
num_stocks = length(unique(stocks_train))

##### use x_{t-1} time-series cut-point #####

xt <- xt[start:split,]
f <- f[start:split,]
first_split_mat = cbind(
    quantile(xt[,c("m1")], c( 0.3, 0.5, 0.7)),
    quantile(xt[,c("m2")], c( 0.3, 0.5, 0.7)),
    quantile(f,            c( 0.3, 0.5, 0.7))
)

print(first_split_mat)

####################################################
############# split the time-series #############
####################################################


t = proc.time()
fit = TreeFactor_APTree_2(R_train, Y_train, X_train, Z_train, H_train, portfolio_weight_train, loss_weight_train, stocks_train, months_train, 
                          first_split_var, first_split_mat, 
                          second_split_var, third_split_var, deep_split_var, 
                          num_stocks, num_months, 
                          min_leaf_size, max_depth, num_iter, num_cutpoints, lambda,
                          equal_weight, no_H, abs_normalize, weighted_loss, 
                          stop_no_gain)
t = proc.time() - t

# this function tells you the first split variable and point
fit$cutpoint
fit$cutvalue

print(t)

print(names(X_train[fit$cutpoint + 1]))
