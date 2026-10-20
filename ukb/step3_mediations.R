
################ Parallel mediation analysis for UKB stress IB-CF

# Load all datasets one by one:

# CRP
load( "../data/df_CRP_1pairsm_rt_final.Rdata")
load( "../data/df_CRP_2trailm_as_final.Rdata")
load( "../data/df_CRP_3matpat_pal_final.Rdata")
load( "../data/df_CRP_4numem_final.Rdata")
load( "../data/df_CRP_5fluidin_final.Rdata")
load( "../data/df_CRP_6symbds_final.Rdata")
load( "../data/df_CRP_7towerre_final.Rdata")

# IL6
load( "../data/df_IL6_1pairsm_rt_final.Rdata")
load( "../data/df_IL6_2trailm_as_final.Rdata")
load( "../data/df_IL6_3matpat_pal_final.Rdata")
load( "../data/df_IL6_4numem_final.Rdata")
load( "../data/df_IL6_5fluidin_final.Rdata")
load( "../data/df_IL6_6symbds_final.Rdata")
load( "../data/df_IL6_7towerre_final.Rdata")

# IBs
load( "../data/df_IBs_1pairsm_rt_final.Rdata")
load( "../data/df_IBs_2trailm_as_final.Rdata")
load( "../data/df_IBs_3matpat_pal_final.Rdata")
load( "../data/df_IBs_4numem_final.Rdata")
load( "../data/df_IBs_5fluidin_final.Rdata")
load( "../data/df_IBs_6symbds_final.Rdata")
load( "../data/df_IBs_7towerre_final.Rdata")

# CRP z-scores
load( "../data/df_CRP_1_zscore_processing_speed_final.Rdata")
load( "../data/df_CRP_2_zscore_reasoning_final.Rdata")
load( "../data/df_CRP_3_zscore_executive_function_final.Rdata")
load( "../data/df_CRP_4_zscore_memory_final.Rdata")

# IL6 z-scores
load( "../data/df_IL6_1_zscore_processing_speed_final.Rdata")
load( "../data/df_IL6_2_zscore_reasoning_final.Rdata")
load( "../data/df_IL6_3_zscore_executive_function_final.Rdata")
load( "../data/df_IL6_4_zscore_memory_final.Rdata")

# IBs z-scores
load( "../data/df_IBs_1_zscore_processing_speed_final.Rdata")
load( "../data/df_IBs_2_zscore_reasoning_final.Rdata")
load( "../data/df_IBs_3_zscore_executive_function_final.Rdata")
load( "../data/df_IBs_4_zscore_memory_final.Rdata")


# List of stressors:
stressors= c("s_phys_abus_CAT", "s_psy_abus_CAT", "s_neglect", "s_loneliness")


# Libraries
library(mediation)
library(future)
library(future.apply)

# Set up a future plan for parallel processing using the available cores (information gotten from .SLURM file)
plan(multicore, workers = as.numeric(Sys.getenv("SLURM_NTASKS", unset = 1))) # Use the cores defined in the .slurm file


############################ CF-Linear: ----
# Note: I will not use robust standard errors because all models meet assumptions.


############# A) Processing speed: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("########## Processing speed and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_2trailm_as_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_tm_duration_num_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_CRP_2trailm_as_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_2trailm_as_final[[stress]])
  max_stress_value <- max(df_CRP_2trailm_as_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) Processing speed (cf_tm_duration_num_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


########## Create the output table to storage the mediation and moderation results:

### Function for marking p-values based on significance:
# @param pval - the pvalue
# @return - returns a significance marker similar to how R marks significance
mark_pvals <- function(pval){
  if (pval < 0.001) {
    return("***")
  } else if (pval < 0.01) {
    return("**")
  } else if (pval < 0.05) {
    return("*")
  } else if (pval < 0.1) {
    return(".")
  } else {
    return(" ")
  }
}

### Function for making table of summary mediation and moderation results (into csv):
# @param - med.list: the list containing all the summary stats for the MEDIATION
# @param - mod.list: the list containing all the summary stats for the MODeration
# @param - moderator: the type of moderator, i.e., sex or age
# @param - CF: the cognitive function variable
# @param - sum.df: the pre-existing data frame I want to add the summary to
tabulate_results <- function(med.list, mod.list, moderator, CF, sum.df = NULL){
  # Create a list with column names for the table:
  c.n <- c("Moderator", "Stressor", "IB",   "CF",   
           # 1) Mediations: Indirect effect-ACME results:
           "MED: Ind_effect_ACME (Estimate)",   "MED: Ind_effect_ACME (CI Lower)",  "MED: Ind_effect_ACME (CI Upper)",  
           "MED: Ind_effect_ACME (p-value)",    "MED: Ind_effect_ACME (pvalue-marking)", "MED: Proportion_mediated_ACME (avg)",
           # 2) Moderations: Indirect effect-ACME results:
           "Mod: Ind_effect_ACME (Estimate)",   "Mod: Ind_effect_ACME (CI Lower)",  "Mod: Ind_effect_ACME (CI Upper)",  
           "Mod: Ind_effect_ACME (p-value)",    "Mod: Ind_effect_ACME (pvalue-marking)", 
           # 3) Mediations: Direct effect-ADE results:
           "MED: Dir_effect_ADE (Estimate)",    "MED: Dir_effect_ADE (CI Lower)",   "MED: Dir_effect_ADE (CI Upper)",   
           "MED: Dir_effect_ADE (p-value)", "MED: Dir_effect_ADE (pvalue-marking)",
           # 4) Moderations: Direct effect-ADE results:
           "Mod: Dir_effect_ADE (Estimate)", "Mod: Dir_effect_ADE (CI Lower)",  "Mod: Dir_effect_ADE (CI Upper)",   
           "Mod: Dir_effect_ADE (p-value)", "Mod: Dir_effect_ADE (pvalue-marking)")
  
  # Initialize a new dataframe:
  if (is.null(sum.df)){
    sum.df <- data.frame(matrix(ncol = 25, nrow = 0))
    colnames(sum.df) <- c.n # Assign column names
  }
  
  # Iterate through all mediation results stored in med.list:
  for (l in names(med.list)){
    # Extract stressor and IB names from the stored keys:
    stress_and_IB <- strsplit(l, "_", fixed = TRUE)[[1]]
    stressor <- paste(head(stress_and_IB, -1), collapse = "_")
    IB <- tail(stress_and_IB, 1)
    lSum.df <- summary(med.list[[l]]) # Extract summary results from MEDIATION analysis
    lMod.list <- mod.list[[l]] # Extract summary results from MOderation analysis
    
    # Populate the table row by row:
    new.r <- c(
      moderator, stressor, IB, CF, 
      # 1) Mediations: Indirect effect-ACME results:
      lSum.df$d.avg, lSum.df$d.avg.ci[1], lSum.df$d.avg.ci[2], 
      formatC(lSum.df$d.avg.p, format = "e", digits = 15), # p-value with 5 decimal points
      mark_pvals(lSum.df$d.avg.p), lSum.df$n.avg,
      # 2) Moderations: Indirect effect-ACME results:
      lMod.list[[1]]$statistic, lMod.list[[1]]$conf.int[1], lMod.list[[1]]$conf.int[2], 
      formatC(lMod.list[[1]]$p.value, format = "f", digits = 7), 
      mark_pvals(lMod.list[[1]]$p.value), 
      # 3) Mediations: Direct effect-ADE results:
      lSum.df$z.avg, lSum.df$z.avg.ci[1], lSum.df$z.avg.ci[2], 
      formatC(lSum.df$z.avg.p, format = "f", digits = 7), 
      mark_pvals(lSum.df$z.avg.p),
      # 4) Moderations: Direct effect-ADE results:
      lMod.list[[2]]$statistic, lMod.list[[2]]$conf.int[1], lMod.list[[2]]$conf.int[2], 
      formatC(lMod.list[[2]]$p.value, format = "f", digits = 7), 
      mark_pvals(lMod.list[[2]]$p.value)
    )
    
    # Append each row to the summary dataframe:
    nr <- nrow(sum.df)
    sum.df[nr+1,] <- new.r # Add each row below the last existent
  } 
  return(sum.df)
}


## D. Apply the function to create the output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", "Processing speed (cf_tm_duration_num_MERG)")

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("########## Processing speed and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_2trailm_as_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_tm_duration_num_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IL6_2trailm_as_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_2trailm_as_final[[stress]])
  max_stress_value <- max(df_IL6_2trailm_as_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) Processing speed (cf_tm_duration_num_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  "Processing speed (cf_tm_duration_num_MERG)", UKB_IB_age_df)

###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("########## Processing speed and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_2trailm_as_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_tm_duration_num_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IBs_2trailm_as_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_2trailm_as_final[[stress]])
  max_stress_value <- max(df_IBs_2trailm_as_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) Processing speed (cf_tm_duration_num_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  "Processing speed (cf_tm_duration_num_MERG)", UKB_IB_age_df)

## ..................................... ----
############################ ZSCORES (also linear): ----


############# A) cf_zscore_processing_speed: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("########## cf_zscore_processing_speed and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_1_zscore_processing_speed_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_processing_speed ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_CRP_1_zscore_processing_speed_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_1_zscore_processing_speed_final[[stress]])
  max_stress_value <- max(df_CRP_1_zscore_processing_speed_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_processing_speed: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age",  "cf_zscore_processing_speed", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("########## cf_zscore_processing_speed and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_1_zscore_processing_speed_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_processing_speed ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IL6_1_zscore_processing_speed_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_1_zscore_processing_speed_final[[stress]])
  max_stress_value <- max(df_IL6_1_zscore_processing_speed_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_processing_speed: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  "cf_zscore_processing_speed", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("########## cf_zscore_processing_speed and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_1_zscore_processing_speed_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_processing_speed ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IBs_1_zscore_processing_speed_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_1_zscore_processing_speed_final[[stress]])
  max_stress_value <- max(df_IBs_1_zscore_processing_speed_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_processing_speed: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  "cf_zscore_processing_speed", UKB_IB_age_df)



############# B) cf_zscore_reasoning: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("########## cf_zscore_reasoning and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_2_zscore_reasoning_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_reasoning ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_CRP_2_zscore_reasoning_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_2_zscore_reasoning_final[[stress]])
  max_stress_value <- max(df_CRP_2_zscore_reasoning_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_reasoning : Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", "cf_zscore_reasoning ", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("########## cf_zscore_reasoning and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_2_zscore_reasoning_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_reasoning ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IL6_2_zscore_reasoning_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_2_zscore_reasoning_final[[stress]])
  max_stress_value <- max(df_IL6_2_zscore_reasoning_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_reasoning : Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", "cf_zscore_reasoning ", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("########## cf_zscore_reasoning and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_2_zscore_reasoning_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_reasoning ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IBs_2_zscore_reasoning_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_2_zscore_reasoning_final[[stress]])
  max_stress_value <- max(df_IBs_2_zscore_reasoning_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_reasoning : Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", "cf_zscore_reasoning ", UKB_IB_age_df)



############# C) cf_zscore_executive_function: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("########## cf_zscore_executive_function and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_3_zscore_executive_function_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_executive_function ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_CRP_3_zscore_executive_function_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_3_zscore_executive_function_final[[stress]])
  max_stress_value <- max(df_CRP_3_zscore_executive_function_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_executive_function: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", "cf_zscore_executive_function", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("########## cf_zscore_executive_function and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_3_zscore_executive_function_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_executive_function ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IL6_3_zscore_executive_function_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_3_zscore_executive_function_final[[stress]])
  max_stress_value <- max(df_IL6_3_zscore_executive_function_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_executive_function: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", "cf_zscore_executive_function", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("########## cf_zscore_executive_function and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_3_zscore_executive_function_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_executive_function ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IBs_3_zscore_executive_function_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_3_zscore_executive_function_final[[stress]])
  max_stress_value <- max(df_IBs_3_zscore_executive_function_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_executive_function: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age",  "cf_zscore_executive_function", UKB_IB_age_df)


############# D) cf_zscore_memory: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("########## cf_zscore_memory and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_4_zscore_memory_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_memory ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_CRP_4_zscore_memory_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_4_zscore_memory_final[[stress]])
  max_stress_value <- max(df_CRP_4_zscore_memory_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_memory: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age",  "cf_zscore_memory", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("########## cf_zscore_memory and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_4_zscore_memory_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_memory ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IL6_4_zscore_memory_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_4_zscore_memory_final[[stress]])
  max_stress_value <- max(df_IL6_4_zscore_memory_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_memory: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", "cf_zscore_memory", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("########## cf_zscore_memory and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_4_zscore_memory_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_zscore_memory ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = gaussian(link = "identity"),
                 data = df_IBs_4_zscore_memory_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_4_zscore_memory_final[[stress]])
  max_stress_value <- max(df_IBs_4_zscore_memory_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) cf_zscore_memory: Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  "cf_zscore_memory", UKB_IB_age_df)


# ###### ----- Place always at the end: 
#  
# write.csv(UKB_IB_age_df, file =  "../outs/UKB_IB_age_results_zscores_beforepoisson.csv", row.names = FALSE, quote = TRUE)

## ..................................... ----
############################ CF-Poisson: ----

############# C) Immediate Episodic Memory: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("########## Immediate Episodic Memory and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_1pairsm_rt_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_pairs_match_1 ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_CRP_1pairsm_rt_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_1pairsm_rt_final[[stress]])
  max_stress_value <- max(df_CRP_1pairsm_rt_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) Immediate Episodic Memory (cf_pairs_match_1): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  "Immediate Episodic Memory (cf_pairs_match_1)", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("########## Immediate Episodic Memory and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_1pairsm_rt_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_pairs_match_1 ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IL6_1pairsm_rt_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_1pairsm_rt_final[[stress]])
  max_stress_value <- max(df_IL6_1pairsm_rt_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) Immediate Episodic Memory (cf_pairs_match_1): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  "Immediate Episodic Memory (cf_pairs_match_1)", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("########## Immediate Episodic Memory and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_1pairsm_rt_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_pairs_match_1 ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IBs_1pairsm_rt_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_1pairsm_rt_final[[stress]])
  max_stress_value <- max(df_IBs_1pairsm_rt_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1) Immediate Episodic Memory (cf_pairs_match_1): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  "Immediate Episodic Memory (cf_pairs_match_1)", UKB_IB_age_df)

############# D)  Working Memory: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("##########  Working Memory and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_4numem_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_num_memory_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_CRP_4numem_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_4numem_final[[stress]])
  max_stress_value <- max(df_CRP_4numem_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Working Memory (cf_num_memory_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Working Memory (cf_num_memory_MERG)", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("##########  Working Memory and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_4numem_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_num_memory_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IL6_4numem_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_4numem_final[[stress]])
  max_stress_value <- max(df_IL6_4numem_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Working Memory (cf_num_memory_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Working Memory (cf_num_memory_MERG)", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("##########  Working Memory and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_4numem_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_num_memory_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IBs_4numem_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_4numem_final[[stress]])
  max_stress_value <- max(df_IBs_4numem_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Working Memory (cf_num_memory_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Working Memory (cf_num_memory_MERG)", UKB_IB_age_df)



############# E)  Digit Symbol Substitution: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("##########  Digit Symbol Substitution and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_6symbds_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_symb_digit_subst_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_CRP_6symbds_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_6symbds_final[[stress]])
  max_stress_value <- max(df_CRP_6symbds_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Digit Symbol Substitution (cf_symb_digit_subst_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Digit Symbol Substitution (cf_symb_digit_subst_MERG)", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("##########  Digit Symbol Substitution and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_6symbds_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_symb_digit_subst_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IL6_6symbds_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_6symbds_final[[stress]])
  max_stress_value <- max(df_IL6_6symbds_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Digit Symbol Substitution (cf_symb_digit_subst_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Digit Symbol Substitution (cf_symb_digit_subst_MERG)", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("##########  Digit Symbol Substitution and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_6symbds_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_symb_digit_subst_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IBs_6symbds_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_6symbds_final[[stress]])
  max_stress_value <- max(df_IBs_6symbds_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Digit Symbol Substitution (cf_symb_digit_subst_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Digit Symbol Substitution (cf_symb_digit_subst_MERG)", UKB_IB_age_df)


############# F)  Fluid intelligence: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("##########  Fluid intelligence and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_5fluidin_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_fluid_intell_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_CRP_5fluidin_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_5fluidin_final[[stress]])
  max_stress_value <- max(df_CRP_5fluidin_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Fluid intelligence (cf_fluid_intell_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Fluid intelligence (cf_fluid_intell_MERG)", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("##########  Fluid intelligence and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_5fluidin_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_fluid_intell_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IL6_5fluidin_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_5fluidin_final[[stress]])
  max_stress_value <- max(df_IL6_5fluidin_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Fluid intelligence (cf_fluid_intell_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Fluid intelligence (cf_fluid_intell_MERG)", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("##########  Fluid intelligence and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_5fluidin_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_fluid_intell_MERG ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IBs_5fluidin_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_5fluidin_final[[stress]])
  max_stress_value <- max(df_IBs_5fluidin_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Fluid intelligence (cf_fluid_intell_MERG): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Fluid intelligence (cf_fluid_intell_MERG)", UKB_IB_age_df)


############# G)  Matrix pattern completion: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("##########  Matrix pattern completion and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_3matpat_pal_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_matrix_pattern ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_CRP_3matpat_pal_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_3matpat_pal_final[[stress]])
  max_stress_value <- max(df_CRP_3matpat_pal_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Matrix pattern completion (cf_matrix_pattern): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Matrix pattern completion (cf_matrix_pattern)", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("##########  Matrix pattern completion and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_3matpat_pal_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_matrix_pattern ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IL6_3matpat_pal_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_3matpat_pal_final[[stress]])
  max_stress_value <- max(df_IL6_3matpat_pal_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Matrix pattern completion (cf_matrix_pattern): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Matrix pattern completion (cf_matrix_pattern)", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("##########  Matrix pattern completion and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_3matpat_pal_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_matrix_pattern ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IBs_3matpat_pal_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_3matpat_pal_final[[stress]])
  max_stress_value <- max(df_IBs_3matpat_pal_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Matrix pattern completion (cf_matrix_pattern): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Matrix pattern completion (cf_matrix_pattern)", UKB_IB_age_df)



############# G)  Tower Rearranging: --------------------------------------------------

###---###  1) CRP: ----

# List of IBs: 
IBs <- c("CRP") 

# Print a message in the R output:
cat("##########  Tower Rearranging and CRP ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_CRP_7towerre_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_tower_rearranging ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_CRP_7towerre_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_CRP_7towerre_final[[stress]])
  max_stress_value <- max(df_CRP_7towerre_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for CRP and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Tower Rearranging (cf_tower_rearranging): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}

## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Tower Rearranging (cf_tower_rearranging)", UKB_IB_age_df)

###---###  2) IL6: ----

# List of IBs: 
IBs= c("IL6") 

# Print a message in the R output:
cat("##########  Tower Rearranging and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IL6_7towerre_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_tower_rearranging ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IL6_7towerre_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IL6_7towerre_final[[stress]])
  max_stress_value <- max(df_IL6_7towerre_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Tower Rearranging (cf_tower_rearranging): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", 
                                  " Tower Rearranging (cf_tower_rearranging)", UKB_IB_age_df)



###---###  3) Other IBs: ----

# List of IBs: 
IBs= c("ICAM1", "ESEL", "TNF")

# Print a message in the R output:
cat("##########  Tower Rearranging and the other IBs ########### \n")

#### A. Define a function to process the mediations for each stressor and each IB:
process_stressor_ib <- function(stressIBpair) {
  stress <- stressIBpair$stressor
  IB <- stressIBpair$IB
  ### 1) Define the models (with interaction terms for age):
  # Mediator model: Stress vs IB (with age as moderator)
  model_M <- glm(as.formula(paste(IB, "~", stress, "* age + sex + bmi + SES + phys_act + depression")), 
                 family = gaussian(link = "identity"),
                 data = df_IBs_7towerre_final)
  
  # Outcome model: Stress vs CF + IB (with age as a moderator for both):
  model_Y <- glm(as.formula(paste("cf_tower_rearranging ~", stress, "*", IB, "* age + sex + bmi + SES + phys_act + depression")),
                 family = poisson(link = "log"),
                 data = df_IBs_7towerre_final)
  
  ### 2) Calculate min and max of stress variable:
  min_stress_value <- min(df_IBs_7towerre_final[[stress]])
  max_stress_value <- max(df_IBs_7towerre_final[[stress]])
  
  ### 3) Run the mediation analysis: 
  set.seed(12345)
  mediation_result <- mediate(model_M, model_Y,
                              treat = stress,     
                              mediator = IB,     
                              sims = 1000,
                              treat.value = max_stress_value,
                              control.value = min_stress_value)
  
  ### 5) Moderation test comparing age 45 vs 65:
  moderation_result <- test.modmed(mediation_result,
                                   list(age = 45),  
                                   list(age = 65),) 
  
  # Print progress message:
  cat("Mediation for IBs and stressor:", stress, " ... DONE! \n")
  
  ### 5) Return a list containing the stressor name, IB, and both mediation and moderation results
  return(list(stressor = stress, 
              IB = IB, 
              mediation = mediation_result, 
              moderation = moderation_result))
}

### B. Run the mediations using the function created: 
# B.1. Generate all (stressor, IB) combinations:
stress_ib_pairs <- expand.grid(stressor = stressors, IB = IBs, stringsAsFactors = FALSE)

# B.2. Run in parallel using future_lapply
results_list <- future_lapply(
  split(stress_ib_pairs, seq(nrow(stress_ib_pairs))),
  process_stressor_ib,
  future.seed = 12345  # Ensures reproducibility in parallel runs
)

# B.3 Store results in named lists
mediation_results <- list()
moderation_results <- list()
# Use a unique delimiter ("___") to build the key
for (res in results_list) {
  key <- paste(res$stressor, res$IB, sep = "___")  # <-- Triple underscores
  mediation_results[[key]] <- res$mediation
  moderation_results[[key]] <- res$moderation
}

### C. Print all mediation and moderation results - (in R-output file)
for (result in names(mediation_results)) {
  # Split key using the same delimiter ("___")
  parts <- unlist(strsplit(result, "___"))
  stressor <- parts[1]
  IB <- parts[2]
  
  cat("1)  Tower Rearranging (cf_tower_rearranging): Mediation result for:", IB, "and stressor:", stressor, "\n")
  print(summary(mediation_results[[result]]))
  
  cat("\n2) Moderation Test (Age 45 vs 65) for:", IB, "and stressor:", stressor, "\n")
  print(moderation_results[[result]])
}


## D. Apply the function to populate the already created output table:
UKB_IB_age_df <- tabulate_results(mediation_results, moderation_results, "Age", " Tower Rearranging (cf_tower_rearranging)", UKB_IB_age_df)



###### ----- Place always at the end: 

write.csv(UKB_IB_age_df, file =  "../outs/UKB_IB_age_results_zscores_ST_collapsed_last_oct22.csv", row.names = FALSE, quote = TRUE)


