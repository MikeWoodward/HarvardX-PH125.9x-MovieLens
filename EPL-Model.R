# Harvard PH128.9x Capstone project: English Premiership prediction
# =================================================================

# Machine learning model development

# Clears out the r workspace each time this file is run. 
rm(list=ls())
# Clears graphics settings
while (!is.null(dev.list())) dev.off()

# Install packages if necessary
if(!require(tidyverse)) install.packages(
  "tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages(
  "caret", repos = "http://cran.us.r-project.org")
if(!require(elasticnet)) install.packages(
  "elasticnet", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages(
  "data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# Start the clock to time the script execution
ptm <- proc.time()

# Check data folders exist
match_folder = 'data_match'
if (!dir.exists(match_folder)) {
  stop(message(sprintf("No %s folder - run download and cleaning", 
                       match_folder)))
}

# Load and prepare data
# =====================
load(file=file.path(match_folder, "match_results.rda"))

# Select just the seasons we can analyze
match_results <- match_results %>% filter(Season > '2010-2011')

# Season week number
# ------------------
# Add the season week number field
start_date <- match_results %>% 
  select(Season, Date) %>% 
  unique() %>% 
  group_by(Season) %>% 
  summarize(Season=Season[1],
            start_date=min(Date),
            day_offset=7*floor(yday(min(start_date)) /7))  
match_results <- match_results %>% 
  left_join(start_date, by='Season') %>%
  mutate(WeekNumber=week(Date - day_offset))

# Points
# ------
temp <- match_results %>% 
  mutate(HomeTeamPoints=3*(FTR=='H') + (FTR=='D'),
         AwayTeamPoints=3*(FTR=='A') + (FTR=='D'))

home_points <- temp %>% 
  select(Season, Date, HomeTeamAbbreviation, HomeTeamPoints) %>%
  rename(c("TeamPoints"="HomeTeamPoints",
           "TeamAbbreviation"="HomeTeamAbbreviation"))

away_points <- temp %>% 
  select(Season, Date, AwayTeamAbbreviation, AwayTeamPoints) %>%
  rename(c("TeamPoints"="AwayTeamPoints",
           "TeamAbbreviation"="AwayTeamAbbreviation"))

points <- rbind(home_points, away_points) %>%
  arrange(Season, TeamAbbreviation, Date) %>%
  group_by(Season, TeamAbbreviation) %>%
  mutate(r=rowid(TeamAbbreviation),
         PriorMeanCumPoints=
           ifelse(rowid(TeamAbbreviation) > 1, 
                  (cumsum(TeamPoints)-TeamPoints)/(r-1),
                  0)) %>%
  select(Season, Date, TeamAbbreviation, PriorMeanCumPoints)
head(points)

match_results <- match_results %>% 
  left_join(points,
            by=c("Season", "Date", 
                 "HomeTeamAbbreviation"="TeamAbbreviation")) %>%
  rename(c("HomePriorMeanCumPoints"="PriorMeanCumPoints"))  %>% 
  left_join(points,
            by=c("Season", "Date", 
                 "AwayTeamAbbreviation"="TeamAbbreviation")) %>%
  rename(c("AwayPriorMeanCumPoints"="PriorMeanCumPoints"))

# Function to calculate RMSE. 
# ==========================
RMSE <- function(actual_goals, forecast_goals){
  sqrt(mean((actual_goals - forecast_goals)^2))
}

# Test, train and validation
# ==========================

set.seed(1000, sample.kind="Rounding") 

# EPL and holdout
# ---------------
holdout_index <- createDataPartition(
  y = match_results$FTHG, 
  times = 1, 
  p = 0.1, 
  list = FALSE)
epl <- match_results[-holdout_index,]
temp <- match_results[holdout_index,]

holdout <- temp %>% 
  semi_join(epl, by = "AwayTeamAbbreviation") %>%
  semi_join(epl, by = "HomeTeamAbbreviation")

# Add rows removed from holdout set back into epl set
removed <- anti_join(temp, holdout)
epl <- rbind(epl, removed)

# Test and train
# --------------
test_index <- createDataPartition(
  y = epl$FTHG, 
  times = 1, 
  p = 0.1, 
  list = FALSE)
train <- epl[-test_index,]
temp <- epl[test_index,]

# Make sure AwayTeamAbbreviation and HomeTeamAbbreviation in test set are also 
# in train set
test <- temp %>% 
  semi_join(epl, by = "AwayTeamAbbreviation") %>%
  semi_join(epl, by = "HomeTeamAbbreviation")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed, temp)

# Home and away data sets
# -----------------------
# We've now split the data into three groups: holdout, test, and train. We need 
# to create home and away subsets of this data

munge_home_away <- function(matches_,at_home_) {
  
  start_date <- matches_ %>% 
    select(Season, Date) %>% 
    unique() %>% 
    group_by(Season) %>% 
    summarize(Season=Season[1],
              start_date=min(Date),
              day_offset=7*floor(yday(min(start_date)) /7))  
  
  if (at_home_) {
    munged <- matches_ %>% 
      mutate(Home=at_home_,
             Goals=FTHG,
             ValueDifference=HomeTeamValue-AwayTeamValue,
             ForeignDifference=HomeTeamForeignPlayers-AwayTeamForeignPlayers,
             MeanAgeDifference=HomeTeamMeanAge-AwayTeamMeanAge,
             PointsDifference=HomePriorMeanCumPoints-AwayPriorMeanCumPoints)
  } else {
    munged <- matches_ %>% 
      mutate(Home=at_home_,
             Goals=FTAG,
             ValueDifference=-HomeTeamValue+AwayTeamValue,
             ForeignDifference=-HomeTeamForeignPlayers+AwayTeamForeignPlayers,
             MeanAgeDifference=-HomeTeamMeanAge+AwayTeamMeanAge,
             PointsDifference=-HomePriorMeanCumPoints+AwayPriorMeanCumPoints)   
  }
  munged <- munged %>% select(Season, 
                              Date,
                              HomeTeamAbbreviation,
                              AwayTeamAbbreviation,
                              Home,
                              Goals, 
                              ValueDifference, 
                              ForeignDifference, 
                              MeanAgeDifference, 
                              WeekNumber, 
                              PointsDifference)
}

holdout_away <- munge_home_away(holdout, FALSE)
holdout_home <- munge_home_away(holdout, TRUE)
  
train_away <- munge_home_away(train, FALSE)
train_home <- munge_home_away(train, TRUE)
  
test_away <- munge_home_away(test, FALSE)
test_home <- munge_home_away(test, TRUE)

# Machine learning
# ================

# Baseline
# --------
# Use mean scores to calculate baseline RMSEs I need to improve upon.
RMSE_baseline_away <- RMSE(test$FTAG, mean(train$FTAG))
RMSE_baseline_home <- RMSE(test$FTHG, mean(train$FTHG))

# Generalized linear model
# ------------------------
glm_model <- function(train_, test_) {
  fit_ <- train(Goals ~ ValueDifference + ForeignDifference + 
                  MeanAgeDifference + WeekNumber + PointsDifference, 
                method = "glm",
                data = train_,
                trControl = trainControl(method = "cv", number = 10, p = 0.9),
                metric='RMSE',
                maximize=FALSE)
  predict_ <- predict(fit_, 
                      newdata=test_)  
  RMSE(test_$Goals, predict_)
}

RMSE_glm_away <- glm_model(train_away, test_away)
RMSE_glm_home <- glm_model(train_home, test_home)

# glmnet
# -----
lambdas <- seq(0, 0.5, by=0.005)
glmnet_model <- function(train_, test_) {
  fit_ <- train(Goals ~ ValueDifference + ForeignDifference + 
                  MeanAgeDifference + WeekNumber + PointsDifference, 
                method = "glmnet",
                data = train_,
                metric='RMSE',
                maximize=FALSE,
                trControl = trainControl(method = "cv", number = 10, p = 0.9),
                tuneGrid = expand.grid(alpha = 0:1, 
                                       lambda = lambdas))
  print(fit_$finalModel$lambdaOpt)
  predict_ <- predict(fit_, 
                      newdata=test_)  
  RMSE(test_$Goals, predict_)
}

RMSE_glmnet_away <- glmnet_model(train_away, test_away)
RMSE_glmnet_home <- glmnet_model(train_home, test_home)

# Random Forest
# -------------
rf_model <- function(train_, test_) {
  fit_ <- train(Goals ~ ValueDifference + ForeignDifference + 
                  MeanAgeDifference + WeekNumber + PointsDifference, 
                method = "rf",
                data = train_,
                metric='RMSE',
                maximize=FALSE,
                trControl = trainControl(method = "cv", number = 10, p = 0.9),
                tuneGrid = data.frame(mtry = seq(10)),
                ntree=100)
  print(fit_$bestTune)
  predict_ <- predict(fit_, 
                      newdata=test_)  
  RMSE(test_$Goals, predict_)
}

RMSE_rf_away <- rf_model(train_away, test_away)
RMSE_rf_home <- rf_model(train_home, test_home)


# Ensemble
# --------

# Tidying up
# ==========

modeling_folder = 'model'
if (!dir.exists(modeling_folder)) {dir.create(modeling_folder)}
save(RMSE_baseline_home_goals,
     RMSE_baseline_away_goals,
     file=file.path(modeling_folder, "model.rda"))

# Script duration
script_duration <- as.numeric((proc.time() - ptm)['elapsed'])
script_duration <- sprintf('%d minutes %.1f seconds', 
                           script_duration%/%60, 
                           script_duration%%60)
print(paste("Script duration was", script_duration))