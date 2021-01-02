library(tidyverse)
library(recipes)
library(modelr)
library(keras)
library(tensorflow)

PATH <- "../input/lish-moa/"
SEEDS <- c(4, 8, 15, 16, 23, 42, 108)
KFOLDS <- 10


cat(list.files(PATH), sep = "\n")


tr <- read_csv(str_c(PATH, "train_features.csv"))
te <- read_csv(str_c(PATH, "test_features.csv"))


keep_rows <- tr$cp_type != "ctl_vehicle"
tr <- tr[keep_rows, ]

Y0 <- read_csv(str_c(PATH, "train_targets_nonscored.csv")) %>% 
  select(-sig_id) %>% 
  filter(keep_rows) %>%
  data.matrix()
Y <- read_csv(str_c(PATH, "train_targets_scored.csv")) %>% 
  select(-sig_id) %>% 
  filter(keep_rows) %>%
  data.matrix()
sub <- read_csv(str_c(PATH, "sample_submission.csv")) %>% 
  mutate(across(where(is.numeric), ~ 0)) 


(rec <- tr %>%
    recipe(~ .) %>%
    step_rm(sig_id, cp_type) %>% 
    step_mutate(g_mean = apply(across(starts_with("g-")), 1, mean),
                c_mean = apply(across(starts_with("c-")), 1, mean)) %>% 
    step_mutate_at(starts_with("cp_"), fn = list(as_factor)) %>% 
    step_mutate_at(contains("g-"), fn = list(copy_g = function(x) x)) %>%
    step_mutate_at(contains("c-"), fn = list(copy_c = function(x) x)) %>%
    step_dummy(starts_with("cp_")) %>% 
    step_normalize(all_numeric()) %>%
    step_pca(contains("copy_g"), num_comp = 2, prefix = "g_pca") %>%
    step_pca(contains("copy_c"), num_comp = 175, prefix = "c_pca") %>%                                             
    prep())


X <- juice(rec, composition = "matrix")
X_te <- bake(rec, te, composition = "matrix")


create_nn <- function(ncol_X, ncol_Y) {
  keras_model_sequential() %>% 
    layer_batch_normalization(input_shape = ncol_X) %>% 
    layer_dropout(0.2) %>% 
    layer_dense(1028, "relu") %>% 
    layer_batch_normalization() %>%
    layer_dropout(0.1) %>%
    layer_dense(260, "relu") %>% 
    layer_batch_normalization() %>% 
    layer_dense(ncol_Y, "sigmoid") %>% 
    keras::compile(loss = "binary_crossentropy", optimizer = "nadam") 
}


callbacks <- function() {
  list(callback_early_stopping(patience = 5, min_delta = 1e-05),
       callback_model_checkpoint("model.h5", save_best_only = TRUE, verbose = 0, mode = "auto"),
       callback_reduce_lr_on_plateau(factor = 0.1, patience = 5, verbose = 0, mode = "auto"))
}

scores <- c()
for (s in SEEDS) {
  set.seed(s)
  for (rs in crossv_kfold(tr, KFOLDS)$train) {
    tri <- as.integer(rs)
    
    m_nn0 <- create_nn(ncol(X), ncol(Y0)) 
    m_nn0 %>% keras::fit(X[tri, ], Y0[tri, ],
                         epochs = 200,
                         batch_size = 128,
                         validation_data = list(X[-tri, ], Y0[-tri, ]),
                         callbacks = callbacks(),
                         view_metrics = FALSE,
                         verbose = 2)
    load_model_weights_hdf5(m_nn0, "model.h5")
    
    m_nn <- create_nn(ncol(X), ncol(Y))
    for (i in 1:(length(m_nn$layers)-1)) set_weights(m_nn$layers[[i]], get_weights(m_nn0$layers[[i]]))
    hist <- m_nn %>% keras::fit(X[tri, ], Y[tri, ],
                                epochs = 200,
                                batch_size = 128,
                                validation_data = list(X[-tri, ], Y[-tri, ]),
                                callbacks = callbacks(),
                                view_metrics = FALSE,
                                verbose = 2)
    load_model_weights_hdf5(m_nn, "model.h5")
    
    scores <- c(scores, min(hist$metrics$val_loss))
    cat("Best val-loss:", min(hist$metrics$val_loss), "at", which.min(hist$metrics$val_loss), "step\n")
    
    sub[, -1] <- sub[, -1] + predict(m_nn, X_te) / KFOLDS / length(SEEDS)
    
    rm(tri, m_nn0, m_nn, hist)
    file.remove("model.h5")
  }
}


cat("\nMean score:", mean(scores), "\n")


sub[te$cp_type == "ctl_vehicle", -1] <- 0
write_csv(sub, "submission.csv")



