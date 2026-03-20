# Mediations_MIDUS_UKB
**Research manuscript:** *Pathways from chronic stress to cognitive decline: A multi-cohort mediation analysis of inflammation and immune-related gene expression.*

This repository contains all scripts and analysis code used in the mediation project investigating how psychosocial stress impacts cognitive decline through inflammatory and immune-related pathways.

The study uses data from two large population-based cohorts:

- MIDUS (Midlife in the United States): Focuses on both inflammatory biomarkers (IL-6, CRP, TNF-α, ICAM-1, E-Selectin) and CTRA gene expression as potential mediators between chronic stress and cognitive outcomes.

- UK Biobank: Serves as a replication cohort, focusing on inflammatory biomarkers and a range of cognitive measures.

Analyses include:

- Moderated mediation models by age using inflammatory biomarkers as mediators.
 
- Mediation models using immune-related gene expression (CTRA transcripts) as mediators between stress and both cognitive function and inflammation.
 
- Construction of stress composites (e.g., childhood abuse, social isolation) and cognitive composite scores (e.g., memory, executive function).
 
- Dimensionality reduction of gene expression data via clustering and PCA to derive composite CTRA modules.
 

All analyses were conducted in R (version 4.4.2), with multiple imputation for missing data, inverse-normal transformations, and bootstrap-based mediation estimates.
