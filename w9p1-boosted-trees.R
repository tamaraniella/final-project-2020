## ----setup, include = FALSE-------------------------------------------------------------
knitr::opts_chunk$set(fig.width = 13, 
                      message = FALSE, 
                      warning = FALSE,
                      echo = TRUE,
                      cache = TRUE)

library(tidyverse)

update_geom_defaults('path', list(size = 3, color = "cornflowerblue"))
update_geom_defaults('point', list(size = 5, color = "gray60"))
theme_set(theme_minimal(base_size = 25))


## ----sim-lin-reg------------------------------------------------------------------------
set.seed(8675309)
n <- 1000
x <- rnorm(n)

a <- 5
b <- 1.3
e <- 4

y <- a + b*x + rnorm(n, sd = e)



## ----plot-sim---------------------------------------------------------------------------
sim_d <- tibble(x = x, y = y)
ggplot(sim_d, aes(x, y)) +
  geom_point() 


## ----estimate-sim-----------------------------------------------------------------------
summary(lm(y ~ x))


## ----grad-descent-----------------------------------------------------------------------
gd <- function(x, y, a, b, learning_rate) {
  n <- length(y)
  yhat <- a + (b * x)
  resid <- y - yhat
 
  # update theta
  a_update <- a - ((1/n) * sum(-2*resid)) * learning_rate
  b_update <- b - ((1/n) * sum(-2*x*resid)) * learning_rate
  
  # Return updated parameter estimates
  c(a_update, b_update)
}


## ----gd-by-hand-------------------------------------------------------------------------
params <- gd(x, y, a = 0, b = 0, 0.01)
params <- gd(x, y, params[1], params[2], 0.01)
params

coef(lm(y ~ x))


## ----gd-loop----------------------------------------------------------------------------
params <- gd(x, y, a = 0, b = 0, 0.01)

for(i in seq_len(1000)) {
  params <- gd(x, y, params[1], params[2], 0.01)  
}

params
coef(lm(y ~ x))


## ----compare-plot-----------------------------------------------------------------------
ggplot(sim_d, aes(x, y)) +
  geom_point() +
  geom_smooth(method = "lm", size = 2, se = FALSE) +
  geom_abline(intercept = params[1], slope = params[2], 
              color = "magenta")


## ----cost-------------------------------------------------------------------------------
mse <- function(x, y, a, b) {
  pred <- a + b*x
  resid2 <- (y - pred)^2
  1/length(y)*sum(resid2)
}

estimate_gradient <- function(x, y, a, b, learning_rate, iter) {
  pars <- gd(x, y, a, b, learning_rate)
  
  c(iter, pars[1], pars[2], mse(x, y, a, b))
}


## ----estimate-gradient------------------------------------------------------------------
iter <- 1000

# set up empty data frame
estimates <- data.frame(iteration = integer(iter),
                        intercept = double(iter),
                        slope = double(iter),
                        cost = double(iter))

# store first row of estimates
estimates[1, ] <- estimate_gradient(x, y, 0, 0, 0.01, 1)

# Estimate remain rows, using previous row as input
for(i in 2:iter) {
  estimates[i, ] <- estimate_gradient(x, y, 
                                      a = estimates$intercept[i - 1],
                                      b = estimates$slope[i - 1],
                                      learning_rate = 0.01,
                                      iter = i)
}


## ----gradient---------------------------------------------------------------------------
head(estimates)
tail(estimates)


## ----cost-reduction---------------------------------------------------------------------
ggplot(estimates, aes(iteration, cost)) +
  geom_point() +
  geom_line()


## ----slope-change-estimate-echo, eval = FALSE-------------------------------------------
## library(gganimate)
## ggplot(estimates[1:300, ]) +
##   geom_point(aes(x, y), sim_d) +
##   geom_smooth(aes(x, y), sim_d,
##               method = "lm", se = FALSE) +
##   geom_abline(aes(intercept = intercept,
##                   slope = slope),
##               color = "#de4f60") +
##   transition_manual(frames = iteration)


## ----slope-change-estimate-eval, echo = FALSE, fig.height = 9---------------------------
library(gganimate)
ggplot(estimates[1:300, ]) +
  geom_point(aes(x, y), sim_d) +
  geom_smooth(aes(x, y), sim_d, 
              method = "lm", se = FALSE) +
  geom_abline(aes(intercept = intercept,
                  slope = slope),
              color = "#de4f60") +
  transition_manual(frames = iteration)


## ----cost-surface-----------------------------------------------------------------------
library(colorspace)
grid <- expand.grid(a = seq(-5, 10, 0.1), b = seq(-5, 5, 0.1))
surface <- grid %>% 
  mutate(cost = map2_dbl(a, b, ~mse(x, y, .x, .y))) %>% 
  ggplot(aes(a, b)) +
    geom_raster(aes(fill = cost)) +
     scale_fill_continuous_sequential(palette = "Terrain")


## ----show-cost-surface, echo = FALSE----------------------------------------------------
surface


## ----cost-space-movie, eval = FALSE-----------------------------------------------------
## library(rayshader)
## 
## plot_gg(surface, multicore = TRUE)
## render_movie(filename = here::here("slides", "img", "cost-surface.mp4"),
##              title_text = 'Cost surface',
##              phi = 30 , theta = -45)


## ----load-data--------------------------------------------------------------------------
library(tidyverse)
library(tidymodels)
set.seed(41920)
d <- read_csv(here::here("data",
                         "edld-654-spring-2020",
                         "train.csv")) %>% 
  select(-classification) %>% 
  sample_frac(0.05)

splt <- initial_split(d)
train <- training(splt)
cv <- vfold_cv(train)


## ----recipe-----------------------------------------------------------------------------
rec <- recipe(score ~ ., train) %>% 
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt))) %>% 
  update_role(contains("id"), ncessch, new_role = "id vars") %>% 
  step_nzv(all_predictors(), freq_cut = 0, unique_cut = 0) %>% 
  step_novel(all_nominal()) %>% 
  step_unknown(all_nominal()) %>% 
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id vars"))  %>% 
  step_dummy(all_nominal(), -has_role("id vars"))


## ----specify-model----------------------------------------------------------------------
mod <- boost_tree() %>% 
  set_engine("xgboost", nthreads = parallel::detectCores()) %>% 
  set_mode("regression") 



## ----default-wf-------------------------------------------------------------------------
wf_df <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(mod)


## ----k-fold-default---------------------------------------------------------------------
library(tictoc)
tic()
fit_default <- fit_resamples(wf_df, cv)
toc()


## ----default-performance----------------------------------------------------------------
collect_metrics(fit_default)


## ----tune_lr, cache = TRUE--------------------------------------------------------------
tune_lr <- mod %>% 
  set_args(trees = 5000,
           learn_rate = tune(),
           stop_iter = 20,
           validation = 0.2)

wf_tune_lr <- wf_df %>% 
  update_model(tune_lr)

grd <- expand.grid(learn_rate = seq(0.0001, 0.3, length.out = 30))

tic()
tune_tree_lr <- tune_grid(wf_tune_lr, cv, grid = grd)
toc()


## ----tune-plot-prep---------------------------------------------------------------------
to_plot <- tune_tree_lr %>% 
  unnest(.metrics) %>% 
  group_by(.metric, learn_rate) %>% 
  summarize(mean = mean(.estimate, na.rm = TRUE)) %>% 
  filter(learn_rate != 0.0001) 

highlight <- to_plot %>% 
  filter(.metric == "rmse" & mean == min(mean)) %>%
  ungroup() %>% 
  select(learn_rate) %>% 
  semi_join(to_plot, .)


## ----tune-plot1-------------------------------------------------------------------------
ggplot(to_plot, aes(learn_rate, mean)) +
  geom_point() +
  geom_point(color = "#de4f69", data = highlight) +
  facet_wrap(~.metric, scales = "free_y")


## ----tune_tree_depth, cache = TRUE------------------------------------------------------
tune_depth <- tune_lr %>% 
  finalize_model(select_best(tune_tree_lr, "rmse")) %>% 
  set_args(tree_depth = tune(),
           min_n = tune())

wf_tune_depth <- wf_df %>% 
  update_model(tune_depth)

grd <- grid_max_entropy(tree_depth(), min_n(), size = 30)

tic()
tune_tree_depth <- tune_grid(wf_tune_depth, cv, grid = grd)
toc()


## ----autoplot-tree-depth----------------------------------------------------------------
autoplot(tune_tree_depth)


## ----show-best-tree-depth---------------------------------------------------------------
show_best(tune_tree_depth, "rmse")


## ----tune-reg---------------------------------------------------------------------------
tune_reg <- tune_depth %>% 
  finalize_model(select_best(tune_tree_depth, "rmse")) %>% 
  set_args(loss_reduction = tune())

wf_tune_reg <- wf_df %>% 
  update_model(tune_reg)

grd <- expand.grid(loss_reduction = seq(0, 100, 5))

tic()
tune_tree_reg <- tune_grid(wf_tune_reg, cv, grid = grd)
toc()


## ----autoplot-tune-reg------------------------------------------------------------------
autoplot(tune_tree_reg)


## ----show-best-tune-reg-----------------------------------------------------------------
show_best(tune_tree_reg, "rmse")


## ----tune-randomness--------------------------------------------------------------------
tune_rand <- tune_reg %>%
  finalize_model(select_best(tune_tree_reg, "rmse")) %>% 
  set_args(mtry = tune(),
           sample_size = tune())

wf_tune_rand <- wf_df %>% 
  update_model(tune_rand)

grd <- grid_max_entropy(finalize(mtry(), juice(prep(rec))), 
                        sample_size = sample_prop(), 
                        size = 30)

tic()
tune_tree_rand <- tune_grid(wf_tune_rand, cv, grid = grd)
toc()


## ----autoplot-rand----------------------------------------------------------------------
autoplot(tune_tree_rand)


## ----show-best-rand---------------------------------------------------------------------
show_best(tune_tree_rand, "rmse")


## ----final-mod--------------------------------------------------------------------------
check_lr <- tune_rand %>% 
  finalize_model(select_best(tune_tree_rand, "rmse")) %>% 
  set_args(learn_rate = tune())

wf_final_lr <- wf_df %>% 
  update_model(check_lr)

tic()
final_lr <- tune_grid(wf_final_lr, cv, grid = 30)
toc()


## ----final-lr-autoplot------------------------------------------------------------------
autoplot(final_lr)


## ----final-lr-show-best-----------------------------------------------------------------
show_best(final_lr, "rmse")


## ----xgboost-data-prep------------------------------------------------------------------
X <- juice(prep(rec)) %>% 
  select(-score) %>% 
  as.matrix()

Y <- juice(prep(rec)) %>% 
  select(score) %>% 
  as.matrix()


## ----10-fold-cv-------------------------------------------------------------------------
library(xgboost)
tic()
def_mod <- xgb.cv(
    data = X,
    label = Y,
    nrounds = 5000,
    objective = "reg:linear",
    early_stopping_rounds = 20, 
    nfold = 10,
    verbose = 0
  ) 
toc()
def_mod$evaluation_log[def_mod$best_iteration, ]


## ----pull-eval-fun----------------------------------------------------------------------
pull_eval <- function(m) {
  m[["evaluation_log"]] %>% 
    pivot_longer(-iter,
                 names_to = c("set", NA, "stat"),
                 names_sep = "_",
                 values_to = "val") %>% 
    pivot_wider(names_from = "stat", 
                values_from = "val") 
}


## ----pull-eval-def-mod------------------------------------------------------------------
def_mod %>% 
  pull_eval() %>% 
  filter(iter > 7) %>% 
  ggplot(aes(iter, mean, color = set)) +
    geom_line() +
    geom_point()

