#!/usr/bin/Rscript
# usage: Rscript bootstrapping.R

library("pacman")
pacman::p_load("tidyverse", "parallel", "foreach", "doParallel")

# ===== load data ===== 
d0 <- read.csv("/Users/dteng/Documents/nmr_targeted/mlgrad/results/mlgrad-pred-lproline_ph3-20230805_1838/df_conc_training_data_t.csv")

num_cores <- 4
num_bootstrap_samples = 50
grad_ls = seq(from=1.25E-9, 
              to=1.34E-9, 
              by=1E-16)

# ============================== multi-threaded run ============================== 
# Register the parallel backend
cl <- makeCluster(num_cores)
registerDoParallel(cl)

t0 <- Sys.time()

result <- foreach(i = 1:num_bootstrap_samples, .combine = "rbind", .packages = c("dplyr")) %dopar% {
  # get bootstrap sample
  bsample <- sample_n(d0, size = nrow(d0), replace = TRUE)
  # get AUC values
  auc_q_ls <- bsample$auc_q
  auc_r_ls <- bsample$auc_r
  
  # get SSE
  m_q <- matrix(c(as.vector(grad_ls))) %*% as.vector(auc_q_ls)
  m_r <- matrix(c(as.vector(grad_ls))) %*% as.vector(auc_r_ls)
  m = m_q - m_r - 1115
  idx_min <- which.min(rowSums(m^2))
  best_grad <- grad_ls[idx_min]
}

print("Time elapsed:")
print(Sys.time() - t0)

# Stop the parallel backend
stopCluster(cl)

# write out
write.csv(result, file = "temp.csv", row.names = FALSE)
# ============================== single-threaded run ============================== 
# code kept for archival reasons
# t0 <- Sys.time()
# 
# # init empty list to populate lsse bootstrap gradients
# best_grad_ls <- numeric(num_bootstrap_samples)
# for (i in 1:num_bootstrap_samples) {
#   # get bootstrap sample
#   bsample <- sample_n(d0, size = nrow(d0), replace = TRUE)
#   # get AUC values
#   auc_q_ls <- bsample$auc_q
#   auc_r_ls <- bsample$auc_r
#   
#   # get SSE
#   m_q <- matrix(c(as.vector(grad_ls))) %*% as.vector(auc_q_ls)
#   m_r <- matrix(c(as.vector(grad_ls))) %*% as.vector(auc_r_ls)
#   m = m_q - m_r - 1115
#   #sse_m <- rowSums(m^2)
#   idx_min <- which.min(rowSums(m^2))
#   best_grad <- grad_ls[idx_min]
#   
#   best_grad_ls[i] <- best_grad
# }
# 
# print("Time elapsed:")
# print(Sys.time() - t0)
