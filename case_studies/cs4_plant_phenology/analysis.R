################################################################################
####                                                                        ####
####            Case Study 4: Common Milkweed Phenology                    ####
####          Seasonal patterns in Asclepias syriaca life stages           ####
####                                                                        ####
################################################################################

# Description: This script analyzes phenological patterns (emergence, flowering,
# seeding, and senescence) in common milkweed (Asclepias syriaca) using
# iNaturalist observation data.

################################################################################
# Setup
################################################################################

# Load required libraries
library(terra)
library(stringr)
library(mapview)
library(data.table)
library(lubridate)
library(tidyverse)
library(tidyr)
library(ggplot2)
library(httr)
library(xml2)
library(rvest)

################################################################################
# Load and prepare data
################################################################################

# Load phenological stage data (first 200 records from each file)
germ <- fread("data/milkweed_germinating.csv")[1:200, ]
flowering <- fread("data/milkweed_flowering.csv")[1:200, ]
seeds <- fread("data/milkweed_seeds.csv")[1:200, ]
death <- fread("data/milkweed_dying.csv")[1:200, ]

# Calculate coverage (percentage of marked observations)
coverage <- data.frame(
  stage = c("germinating", "flowering", "seeds", "dying"),
  coverage = c(
    length(which(germ$marked == 1)) / 200,
    length(which(flowering$marked == 1)) / 200,
    length(which(seeds$marked == 1)) / 200,
    length(which(death$marked == 1)) / 200
  )
)
print(coverage)

# Filter to marked observations only
germ.marked <- germ[which(germ$marked == 1), ]
flowering.marked <- flowering[which(flowering$marked == 1), ]
seeds.marked <- seeds[which(seeds$marked == 1), ]
death.marked <- death[which(death$marked == 1), ]

# Add phenological stage labels
germ.marked$stage <- "emergence"
flowering.marked$stage <- "flowering"
seeds.marked$stage <- "seeding"
death.marked$stage <- "senescence"

# Combine all data
all <- rbind(germ.marked, flowering.marked, seeds.marked, death.marked)

################################################################################
# Scrape observation dates from iNaturalist
################################################################################

# Function to extract observation date from iNaturalist URL
get_inat_observation_date <- function(url) {
  require(httr)
  require(xml2)
  require(rvest)
  
  HEADERS <- c(
    `User-Agent` = "inat-date-scraper/0.1 (contact: your@email.com)"
  )
  
  tryCatch({
    response <- GET(url, add_headers(.headers = HEADERS), timeout(15))
    stop_for_status(response)
    
    page <- read_html(response)
    date_tag <- html_node(page, "span.date")
    
    if (!is.na(date_tag)) {
      date <- html_attr(date_tag, "title")
      return(date)
    } else {
      warning(paste("No date found for", url))
      return("")
    }
  }, error = function(e) {
    warning(paste("Error fetching", url, ":", e$message))
    return("")
  })
}

# Extract dates for all observations (with 2-second delay to avoid API overload)
for (i in 1:nrow(all)) {
  all$Date[i] <- get_inat_observation_date(all$inat_url[i])
  Sys.sleep(2)  # Rate limiting
  print(paste("Processed", i, "of", nrow(all)))
}

# Convert to day of year
all$DOY <- yday(as.Date(all$Date))

# Filter to common milkweed species only
all <- all[which(all$species == "Asclepias syriaca"), ]

# Check data completeness
missing_dates <- length(which(all$Date == ""))
cat("Observations without date info:", missing_dates, "\n")

################################################################################
# Visualizations
################################################################################

# Define color palette for phenological stages
stage_colors <- c(
  "emergence" = "#6bb46b",
  "flowering" = "#8c517c",
  "seeding" = "#D6C78E",
  "senescence" = "#24363A"
)

# Plot 1: Density plot of day of year by phenological stage
density_plot <- ggplot(all, aes(x = DOY, colour = stage, fill = stage)) +
  geom_density(alpha = 0.5) +
  scale_color_manual(values = stage_colors) +
  scale_fill_manual(values = stage_colors) +
  labs(
    x = "Day of Year",
    y = "Density",
    fill = "Phenological Stage",
    color = "Phenological Stage"
  ) +
  theme_linedraw() +
  theme(
    aspect.ratio = 1,
    legend.position = "none",
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave("outputs/density.pdf", plot = density_plot, width = 4, height = 4)

################################################################################
# Statistical analysis
################################################################################

# ANOVA: Test for differences in day of year across phenological stages
anova_model <- aov(DOY ~ stage, data = all)
cat("\nANOVA Results:\n")
print(summary(anova_model))  # Significant: p < 2e-16

# Tukey HSD post-hoc test for pairwise comparisons
tukey_results <- TukeyHSD(anova_model)
print(tukey_results)

################################################################################
# Plot 2: Boxplot with statistical groupings
################################################################################

boxplot <- ggplot(all, aes(x = DOY, y = stage, colour = stage, fill = stage)) +
  geom_boxplot(alpha = 0.5) +
  scale_color_manual(values = stage_colors) +
  scale_fill_manual(values = stage_colors) +
  scale_y_discrete(limits = c("senescence", "seeding", "flowering", "emergence")) +
  labs(
    x = "Day of Year",
    y = "",
    fill = "Phenological Stage",
    color = "Phenological Stage"
  ) +
  # Add statistical grouping letters (from Tukey HSD)
  annotate("text", x = 340, y = 1, label = "c") +
  annotate("text", x = 345, y = 2, label = "b") +
  annotate("text", x = 255, y = 3, label = "a") +
  annotate("text", x = 275, y = 4, label = "a") +
  theme_linedraw() +
  theme(
    aspect.ratio = 1,
    legend.position = "none",
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank()
  )

ggsave("outputs/boxplot.pdf", plot = boxplot, width = 4, height = 4)

cat("\nAnalysis complete! Outputs saved to outputs/ directory.\n")
