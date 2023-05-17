library(argparser)
library(assertthat)
library(rlang)
library(data.table)
library(vctrs)
library(ricu)

source("R/misc.R")
source("R/steps.R")
source("R/sequential.R")
source("R/obs_time.R")


# Create a parser
p <- arg_parser("Extract and preprocess ICU sepsis data")
p <- add_argument(p, "--src", help="source database", default="mimic_demo")
argv <- parse_args(p)

src <- argv$src 
conf <- ricu:::read_json("../YAIB/ricu-interface/cohorts/config.json")
path <- file.path(conf$output_dir, "sepsis")


cncpt_env <- new.env()

# Task description
time_flow <- "sequential" # sequential / continuous
time_unit <- hours
freq <- 1L
max_len <- hours(7 * 24)  # = 7 days

static_vars <- c("age", "sex", "height", "weight")

dynamic_vars <- c("alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir",
          "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", 
          "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact",
          "lymph", "map", "mch", "mchc", "mcv", "methb", "mg", "na", "neut", 
          "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp", 
          "temp", "tnt", "urine", "wbc")

# cross-sectional vs longitudinal
predictor_type <- "dynamic" # static / dynamic
outcome_type   <- "dynamic" # static / dynamic


patients <- stay_windows(src, interval = time_unit(freq))
patients <- as_win_tbl(patients, index_var = "start", dur_var = "end", interval = time_unit(freq))

# Only keep patients in the base cohort (see base_cohort.R)
base <- arrow::read_parquet(file.path(conf$output_dir, "base", src, "sta.parquet"))
patients <- patients[id_col(patients) %in% id_col(base)]


# Define outcome ----------------------------------------------------------

args <- list(cache = TRUE)
if (src %in% c("eicu", "eicu_demo", "hirid")) {
  args <- c(args, list(si_mode = "abx"))
}

outc <- do.call(load_step, args = c(list(x = dict["sep3_alt"]), args))
outc <- summary_step(outc, "first")


# Define observation times ------------------------------------------------

stop_obs_at(outc, offset = hours(6L), by_ref = TRUE)
stop_obs_at(patients, offset = ricu:::re_time(max_len, time_unit(freq)), by_ref = TRUE)


# Apply exclusion criteria ------------------------------------------------

# Exclusions 1.-5. are defined in base_cohort.R

# 6. Low sepsis prevalence
prevalence <- function(concept, hospital_ids, ...) {
  assert_that(is_logical(data_col(concept)))
  var <- data_var(concept)
  cncpt_per_hosp <- concept[hospital_ids]
  cncpt_per_hosp[, (var) := ricu::replace_na(.SD[[var]], FALSE)]
  prevalence <- cncpt_per_hosp[, .(prev = mean(.SD[[var]])), by = hospital_id]
  res <- merge(hospital_ids, prevalence, by = "hospital_id")
  rm_cols(res, "hospital_id")
}

if (src %in% c("eicu", "eicu_demo")) {
  x1 <- do.call(load_step, args = c(list(x = dict["sep3_alt"]), args))
  x2 <- summary_step(x1, "exists")
  x3 <- load_step(dict["hospital_id"])
  x4 <- function_step(x2, prevalence, hospital_ids = x3)
  x5 <- filter_step(x4, ~ . == 0)
  
  excl6 <- unique(x5[, id_vars(x5), with = FALSE])
} else {
  excl6 <- patients[0]
}


# 7. Sepsis onset before 6h in the ICU
x1 <- load_step(dict["sep3_alt"], cache = TRUE)
x2 <- summary_step(x1, "first")
x3 <- filter_step(x2, ~ . < 6, col = index_col)

excl7 <- unique(x3[, id_vars(x3), with = FALSE])





# Apply exclusions
patients <- exclude(patients, mget(paste0("excl", 6:7)))
attrition <- as.data.table(patients[c("incl_n", "excl_n_total", "excl_n")])
patients <- patients[['incl']]
patient_ids <- patients[, .SD, .SDcols = id_var(patients)]



# Prepare data ------------------------------------------------------------

# Get predictors
dyn <- load_step(dict[dynamic_vars], cache = TRUE)
sta <- load_step(dict[static_vars], cache = TRUE)

# Transform all variables into the target format
assert_that(outcome_type == "dynamic", time_flow == "sequential")

outc_fmt <- function_step(outc, map_to_grid)
outc_fmt <- function_step(outc_fmt, outcome_window, window = c(6L, 6L))
rename_cols(outc_fmt, c("stay_id", "time", "label"), by_ref = TRUE)

dyn_fmt <- function_step(dyn, map_to_grid)
rename_cols(dyn_fmt, c("stay_id", "time"), meta_vars(dyn_fmt), by_ref = TRUE)

sta_fmt <- sta[patient_ids]  # TODO: make into step
rename_cols(sta_fmt, c("stay_id"), id_vars(sta), by_ref = TRUE)


# Write to disk -----------------------------------------------------------

out_path <- paste0(path, "/", src)

if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

arrow::write_parquet(outc_fmt, paste0(out_path, "/outc.parquet"))
arrow::write_parquet(dyn_fmt, paste0(out_path, "/dyn.parquet"))
arrow::write_parquet(sta_fmt, paste0(out_path, "/sta.parquet"))
fwrite(attrition, paste0(out_path, "/attrition.csv"))

