# MovieLens project
# =================

# This is the r code for the MovieLens project. The code is broken into six
# sections:
# 1. Housekeeping. Package loading and function definitions.
# 2. ETL. Extract Transform Load. Load, clean, and prepare the data for use.
# 3. Exploratory analysis.  
# 4. Model building. Test the contributions to the model using the training
#    and test subsets of the edx variable.
# 5. Regularization.
# 6. Final model evaluation. This is the *only* place where the validation
#    (holdout) data is used.
# The rubric asked for RMSE calculations. In this code, all RMSE variables
# are prefixed by rmse_.
# The script is memory hungry, to reduce memory footprint, I remove
# unnecessary variables in the code. 
# On my modern, high-spec Mac Book Pro computer (multiple cores, fast processor, 
# large memory), the script runs in 11 minutes 30 seconds.
# To aid the reader, I mark main sections in the code with ==== comments
# and subsections with ---- comments.
# Standard r guidelines suggest 80 character lines and I have followed that 
# guideline here.
# My r version is 4.02. For clarity, I have removed all conditional code for
# running with r versions less than 4.

# Housekeeping
# ============
# Loading packages and defining functions etc.

# Clears out the r workspace each time this file is run. 
rm(list=ls())
# Clears graphics settings
while (!is.null(dev.list())) dev.off()

# Start the clock to time the script execution
ptm <- proc.time()

# Creates data folder if not already present, empties it otherwise
data_folder = 'data'
if (!dir.exists(data_folder)) {
  dir.create(data_folder)
} else {
  f <- list.files(data_folder, 
                  include.dirs = FALSE, full.names = TRUE, recursive = TRUE)
  file.remove(f)
  rm(f)
}

# Install packages if necessary
if(!require(tidyverse)) install.packages(
  "tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages(
  "caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages(
  "data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# Function to calculate RMSE.  
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# ETL
# ===
# Extract Transform Load - data cleansing and data preparation.

# Get file from grouplens site
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Formatting data
ratings <- fread(
  text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
  col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(
  readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Add columns needed for exploration and analysis
# release_year is the year the movie was released
# review_week was the week the movie as reviewed
# release_review is the number of years between release and review
movielens <- movielens %>% mutate(
  release_year = as.numeric(
    str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),
  review_week = round_date(as_datetime(timestamp), unit = "week"),
  review_release = year(as_datetime(timestamp)) - release_year)

# Validation set will be 10% of MovieLens data. Note this data is used
# for final validation only.
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(
  y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure all userIds and movieIds in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove unnecessary variables to keep memory overhead down
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Create train and test set for machine learning
# ----------------------------------------------
# Note these are the sets used for modeling and regularization. The validation
# set is only used for final validation.
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(
  y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed, temp)

# Remove unnecessary variables
rm(test_index, temp, removed)

# Data quality checks.
# -------------------
# Stop the script execution if basic quality checks not passed.

# Check edx loaded correctly. Basic checks - if Drama numbers correct,
# other genres should be too. Drama chosen at random as an example.
if ((edx %>% select(movieId) %>% n_distinct() != 10677) 
    | (edx %>% select(userId) %>% n_distinct() != 69878)
    | (edx %>% filter(str_detect(genres, 'Drama')) %>% 
       select(rating) %>% nrow() != 3910127)) {
  stop(message("edx data not correctly loaded. Stopping execution."))
}
# Check validation loaded correctly. Check counts are correct.
if ((validation %>% select(movieId) %>% n_distinct() != 9809) 
    | (validation %>% select(userId) %>% n_distinct() != 68534)
    | (validation %>% filter(str_detect(genres, 'Drama')) %>% 
       select(rating) %>% nrow() != 434071)) {
  stop(message("validation data not correctly loaded. Stopping execution."))
}

# Check for NAs in the data. If we find any, we need to stop and clean the 
# data.
if (anyNA(edx)) {
  stop(message("NAs found in edx data - need to clean data."))
}
if (anyNA(validation)) {
  stop(message("NAs found in validation data - need to clean data."))
}

# Data exploration
# ================

print("In exploration")
print("==============")

# Quick views of data
# -------------------
# Quick look at the top of data, using only the original fields and not the 
# ones I added
edx_original_head <- head(edx %>% 
  select(userId, movieId, timestamp, title, genres, rating))
print(edx_original_head)
# Observations:
# 1. Release year coded into title
# 2. timestamp is Unix coded timestamp of review
# 3. genres field has multiple values separated by a |

# View rating values
edx_unique_ratings <- sort(unique(edx$rating))
print(edx_unique_ratings)
# Observations:
# Ratings vary from 0.5 to 5 in steps of 0.5. 

edx_summary <-edx %>% summarize(no_users = n_distinct(userId), 
    no_movies = n_distinct(movieId), 
    no_ratings = n(),
    mean_rate_user = no_ratings/no_users,
    mean_rate_movie = no_ratings/no_movies)
print(edx_summary)
# Observations:
# 1. 69878 users, 10677 movies - more users than movies
# 2. ~129 ratings per user - seems like very active users
# 3. ~843 ratings per movie - lots of ratings per movie on average
# High averages suggest we should look more closely at distribution.

# Views of data
# -------------
# Ratings distribution histogram
edx_rating_histogram <- edx %>% ggplot(aes(rating)) + 
  geom_histogram(binwidth=0.5, color="darkblue", fill="lightblue") + 
  ggtitle("edx ratings histogram") + 
  ylab("count of ratings") +
  theme(aspect.ratio=0.5)
print(edx_rating_histogram)
# Observations:
# 1. Half star ratings less common than full star ratings
# 2. Most common rating is 4
# 3. Distribution is not uniform - significant difference in frequency

# Mean ratings by movie histogram
edx_movies_histogram <- edx %>% 
  group_by(movieId) %>%
  summarize(mean_rating=mean(rating), .groups='keep') %>%
  ggplot(aes(mean_rating)) + 
  geom_histogram(bins=30, color="darkblue", fill="lightblue") + 
  ggtitle("edx count of movies vs. mean ratings") +
  ylab("count of movies") +
  xlab("mean ratings") +
  theme(aspect.ratio=0.5)
print(edx_movies_histogram)
# Observations
# 1. Strong movie effect
# 2. Distribution is left-skewed

# Mean ratings by user histogram
edx_users_histogram <- edx %>% 
  group_by(userId) %>%
  summarize(mean_rating=mean(rating), .groups='keep') %>%
  ggplot(aes(mean_rating)) + 
  geom_histogram(bins=30, color="darkblue", fill="lightblue") + 
  ggtitle("edx count of users vs. mean ratings") +
  ylab("count of users") +
  xlab("mean ratings") +
  theme(aspect.ratio=0.5)
print(edx_users_histogram)
# Observations:
# 1. Strong user effect
# 2. Distribution appears to be roughly normal

# Individual genres. 
# Gets alphabetically sorted list of the individual genres present in the
# genres field...
edx_unique_genres <- sort(unique(
  unlist(strsplit(paste(unique(edx$genres), collapse="|"), "\\|"))))
print(edx_unique_genres)
# Counts the number of distinct genre combinations in the genre field 
edx_distinct_genre_combos <- n_distinct(edx$genres)
print(edx_distinct_genre_combos)

# Mean ratings by genre. Error bar is 1.96*se: the 95% confidence interval.
# Show all genres with more than 50,000 ratings
edx_genres_ratings_micro <- edx %>% 
  group_by(genres) %>% 
  summarize(n_ratings = n(),
            mean_rating = mean(rating),
            se = sd(rating)/sqrt(n()),
            .groups="keep") %>%
  filter(n_ratings >= 50000) %>%
  ggplot(aes(x=reorder(genres, -mean_rating), 
             y=mean_rating, 
             ymin = mean_rating - 1.96*se, 
             ymax = mean_rating + 1.96*se)) +
    geom_point(color='blue') + 
    geom_errorbar(color='blue') + 
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), 
          aspect.ratio=0.4) +
    ggtitle("edx average rating vs. genre") +
    ylab("mean rating") +
    xlab("genre")
print(edx_genres_ratings_micro)
# Show all ratings, but suppress labeling of genre axis because it will be
# too cluttered to read.
edx_genres_ratings_macro <- edx %>% 
  group_by(genres) %>% 
  summarize(n_ratings = n(),
            mean_rating = mean(rating),
            se = sd(rating)/sqrt(n()),
            .groups="keep") %>%
  ggplot(aes(x=reorder(genres, -mean_rating), 
             y=mean_rating, 
             ymin = mean_rating - 1.96*se, 
             ymax = mean_rating + 1.96*se)) +
  geom_point(color='blue') + 
  geom_errorbar(color='blue') +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line = element_line(),
        aspect.ratio=0.5) +
  ggtitle("edx average rating vs. genre") +
  ylab("mean rating") +
  xlab("genre")
print(edx_genres_ratings_macro)
# Observations
# 1. Very clear genre effect
# 2. Some genres have high variability, which may lead to a poor fit

# Mean ratings by release_year. 1.96 gives us the 95% confidence interval.
edx_ratings_release_year <- edx %>% 
  group_by(release_year) %>%
  summarize(mean_rating=mean(rating), 
            se = sd(rating)/sqrt(n()),
            .groups='keep') %>%
  ggplot(aes(x=release_year, 
             y=mean_rating,
             ymin = mean_rating - 1.96*se, 
             ymax = mean_rating + 1.96*se)) + 
  geom_point(color='blue') +
  geom_errorbar(color='blue') +
  ggtitle("edx average rating vs. release year") +
  ylab("mean rating") +
  xlab("year") +
  theme(aspect.ratio=0.5)
print(edx_ratings_release_year)
# Observations:
# 1. Not as big an effect as user, movie, and genre, but none the less,
#    there is an effect.
# 2. Range is about 3.3 to 4.05
# 3. Effect becomes more pronounced after about 1970.

# Mean rating by review week
edx_ratings_review_week <- edx %>% 
  group_by(review_week) %>%
  summarize(mean_rating=mean(rating), 
            se = sd(rating)/sqrt(n()),
            .groups='keep') %>%
  ggplot(aes(x=review_week, 
             y=mean_rating,
             ymin = mean_rating - 1.96*se, 
             ymax = mean_rating + 1.96*se)) + 
  geom_point(color='blue') +
  geom_errorbar(color='blue') +
  ylim(3, 4.75) +
  ggtitle("edx count of review week vs. mean ratings") +
  ylab("mean rating") +
  xlab("year") +
  theme(aspect.ratio=0.5)
print(edx_ratings_review_week)
# Observations
# 1. Small effect, range of about 3.4 to 3.6, excluding outliers.
# 2. Not clear this is worth including in model.

# Mean rating by review_release
edx_ratings_review_release <- edx %>% 
  group_by(review_release) %>%
  summarize(mean_rating=mean(rating), 
            se = sd(rating)/sqrt(n()),
            .groups='keep') %>%
  ggplot(aes(x=review_release, 
             y=mean_rating,
             ymin = mean_rating - 1.96*se, 
             ymax = mean_rating + 1.96*se)) + 
  geom_point(color='blue') +
  geom_errorbar() +
  ggtitle("edx mean rating vs year difference") +
  ylab("mean rating") +
  xlab("year difference") +
  theme(aspect.ratio=0.5)
print(edx_ratings_review_release)

## Save plots and data
ggsave(filename=file.path(data_folder, "edx_ratings_histogram.png"), 
       width=6, height=3, units='in',
       plot=edx_rating_histogram)
ggsave(filename=file.path(data_folder, "edx_movies_histogram.png"), 
       width=6, height=3, units='in',
       plot=edx_movies_histogram)
ggsave(filename=file.path(data_folder, "edx_users_histogram.png"), 
       width=6, height=3, units='in',
       plot=edx_users_histogram)
ggsave(filename=file.path(data_folder, "edx_genres_ratings_macro.png"), 
       width=6, height=3, units='in',
       plot=edx_genres_ratings_macro)
ggsave(filename=file.path(data_folder, "edx_genres_ratings_micro.png"),
       width=6, height=4, units='in',
       plot=edx_genres_ratings_micro)
ggsave(filename=file.path(data_folder, "edx_ratings_release_year.png"), 
       width=6, height=3, units='in',
       plot=edx_ratings_release_year)
ggsave(filename=file.path(data_folder, "edx_ratings_review_week.png"), 
       width=6, height=3, units='in',
       plot=edx_ratings_review_week)
ggsave(filename=file.path(data_folder, "edx_ratings_review_release.png"), 
       width=6, height=3, units='in',
       plot=edx_ratings_review_release)

# These are all variables I'll be using in the Rmd file.
save(edx_original_head, 
     edx_unique_ratings,
     edx_summary,
     edx_unique_genres,
     edx_distinct_genre_combos,
     file=file.path(data_folder, 'movielens_exploration.rda'))

# To preserve memory, remove variables not needed
rm(edx_original_head,
   edx_unique_ratings, 
   edx_summary, 
   edx_rating_histogram, 
   edx_movies_histogram,
   edx_users_histogram,
   edx_unique_genres,
   edx_distinct_genre_combos,
   edx_genres_ratings_micro,
   edx_genres_ratings_macro,
   edx_ratings_release_year,
   edx_ratings_review_week,
   edx_ratings_review_release)

# Analysis
# ========

print("In analysis")
print("===========")

# Modeling features
# ------------------
# Simplest of models, naive mean for everything.
mu <- mean(train$rating)
rmse_naivemean <- RMSE(test$rating, mu)
model_scores <- data.frame(method='Naive means', 
                           rmse=rmse_naivemean)

# Adding movie effects
b_i <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu),
            .groups='keep')
predicted_ratings <- mu + test %>% 
  left_join(b_i, by='movieId') %>%
  pull(b_i)
rmse_movie <- RMSE(test$rating, predicted_ratings)
model_scores <- rbind(model_scores, c("Movie effects", rmse_movie))

# Adding user effects
b_u <- train %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i),
            .groups='keep')
predicted_ratings <- test %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
rmse_user <- RMSE(test$rating, predicted_ratings)
model_scores <- rbind(model_scores, c("User effects", rmse_user))

# Adding genres - not in course
b_g <- train %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u), 
            .groups='keep')
predicted_ratings <- test %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
rmse_genres = RMSE(test$rating, predicted_ratings)
model_scores <- rbind(model_scores, c("Genres", rmse_genres))

# Add release review effect - not in course
b_rr <- train %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  group_by(review_release) %>%
  summarize(b_rr = mean(rating - mu - b_i - b_u - b_g), 
            .groups='keep')
predicted_ratings <- test %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  left_join(b_rr, by='review_release') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_rr) %>%
  pull(pred)
rmse_releasereview = RMSE(test$rating, predicted_ratings)
model_scores<-rbind(model_scores, c("Release review", rmse_releasereview))

# Add release year - not in course
b_ry <- train %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  left_join(b_rr, by='review_release') %>%
  group_by(release_year) %>%
  summarize(b_ry = mean(rating - mu - b_i - b_u - b_g - b_rr), 
            .groups='keep')
predicted_ratings <- test %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  left_join(b_rr, by='review_release') %>%
  left_join(b_ry, by='release_year') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_rr + b_ry) %>%
  pull(pred)
rmse_releaseyear = RMSE(test$rating, predicted_ratings)
model_scores <-rbind(model_scores, c("Release year", rmse_releaseyear))
# rmse column is character string when it should be numeric
model_scores[,'rmse'] <- as.numeric(model_scores[,'rmse'])
# Calculate a % difference from previous entry
model_scores['% change from previous'] <- 
  100*(lag(model_scores['rmse']) - 
       model_scores['rmse'])/lag(model_scores['rmse'])

# Show the model RMSE scores for each stage in the model creation
print(model_scores)

# Regularization
# --------------
# Complex lambda sequence is to cover minimum in small steps, but have
# large steps as we step away from minimum
lambdas <- sort(c(seq(0,     0.1, 0.1),
                  seq(0.2,   0.6, 0.025), 
                  seq(0.62,  0.8, 0.2),
                  seq(0.825, 4,   0.25)))

# Applies lambda to function that calculates RMSE for each lambda.
rmses_regularization <- sapply(lambdas, function(lambda) {
  
  mu <- mean(train$rating)

  # Adding movie effects
  b_i <- train %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n() + lambda),
              .groups='keep')

  # Adding user effects
  b_u <- train %>% 
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + lambda),
              .groups='keep')

  # Genre
  b_g <- train %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + lambda), 
              .groups='keep')
  
  # Add release review effect
  b_rr <- train %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    group_by(review_release) %>%
    summarize(b_rr = sum(rating - mu - b_i - b_u - b_g)/(n() + lambda), 
              .groups='keep')
  
  # Release year
  b_ry <- train %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    left_join(b_rr, by='review_release') %>%
    group_by(release_year) %>%
    summarize(b_ry = sum(rating - mu - b_i - b_u - b_g - b_rr)/(n() + lambda), 
              .groups='keep')
  
  # Predicted rating
  predicted_ratings <- test %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    left_join(b_rr, by='review_release') %>%
    left_join(b_ry, by='release_year') %>%
    mutate(pred = mu + b_i + b_u + b_g + b_rr + b_ry) %>%
    pull(pred)
 
  RMSE(test$rating, predicted_ratings)
})

# Convert lambda data to data frame for easier handling
regularization <- data.frame(lambda = lambdas, 
                             rmse = rmses_regularization)

# Plot regularization chart
regularization_chart <- regularization %>% ggplot(aes(x=lambda, y=rmse)) + 
  geom_point(color='blue') +
  ggtitle("Regularization. rmse vs lambda") +
  theme(aspect.ratio=0.5)
print(regularization_chart)

# Find lambda minima
idx <- which.min(regularization[,2])
lambda <- regularization[idx,1]

# Save key analysis
save(model_scores, lambda, 
     file=file.path(data_folder, "movielens_analysis.rda"))

ggsave(filename=file.path(data_folder, "regularization_chart.png"), 
       width=6, height=3, units='in',
       plot=regularization_chart)

# Final model evaluation
# ======================

print("In final evaluation")
print("===================")

# Training on on whole edx data set
# ---------------------------------
# This is the last step before final evaluation. I'm training my model on the
# edx data set using all the existing features developed using test and
# train.
mu <- mean(edx$rating)

# Adding movie effects
b_i <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + lambda),
            .groups='keep')

# Adding user effects
b_u <- edx %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n() + lambda),
            .groups='keep')

# Genre
b_g <- edx %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + lambda), 
            .groups='keep')

# Add review_release effect
b_rr <- edx %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  group_by(review_release) %>%
  summarize(b_rr = sum(rating - mu - b_i - b_u - b_g)/(n() + lambda), 
            .groups='keep')

# Release year
b_ry <- edx %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  left_join(b_rr, by='review_release') %>%
  group_by(release_year) %>%
  summarize(b_ry = sum(rating - mu - b_i - b_u - b_g - b_rr)/(n() + lambda), 
            .groups='keep')

# Predicted rating
# ----------------
predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  left_join(b_rr, by='review_release') %>%
  left_join(b_ry, by='release_year') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_rr + b_ry) %>%
  pull(pred)

# Final evaluation
# ----------------
# Note this is the only place in the code where the validation data is used
# Target RMSE from course.
rmse_target <- 0.86490
# Final model valuation
rmse_final <- RMSE(validation$rating, predicted_ratings)
print(sprintf("Final RMSE is %f", rmse_final))
print(ifelse(rmse_final < rmse_target, 
             "which is better than required by project", 
             "worse than required by project"))

# Clipping
# Ratings can go from 0.5 to 5, clipping 'clips' predicted ratings to a 
# max of 5 and a minimum of 0.5.
predicted_ratings <- ifelse(predicted_ratings > 5, 5, predicted_ratings)
predicted_ratings <- ifelse(predicted_ratings < 0.5, 0.5, predicted_ratings)
rmse_clipped <- RMSE(validation$rating, predicted_ratings)
sprintf("Clipped RMSE is %f", rmse_clipped)

# Script duration
script_duration <- as.numeric((proc.time() - ptm)['elapsed'])
script_duration <- sprintf('%d minutes %.1f seconds', 
                           script_duration%/%60, 
                           script_duration%%60)
print(paste("Script duration was", script_duration))

# Save final results
save(rmse_target, rmse_final, rmse_clipped, script_duration, 
     file=file.path(data_folder, "movielens_evaluation.rda"))

print("End of script")
