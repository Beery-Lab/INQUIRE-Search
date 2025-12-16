################################################################################
####                                                                        ####
####    Case Study 3: Wildlife Mortality - Urban vs. Rural Patterns       ####
####      Seasonality of bird mortality in Massachusetts                   ####
####                                                                        ####
################################################################################

# Description: This script analyzes seasonal patterns in bird mortality rates
# comparing urban (Boston) and rural (Pioneer Valley) environments in 
# Massachusetts using iNaturalist observations.

################################################################################
# Setup
################################################################################

# Load required libraries
library(tidyverse)
library(ggplot2)
library(ggthemes)
library(rinat)
library(rnaturalearth)
library(sf)
library(ggspatial) 
library(mgcv)
library(scales)
library(patchwork)

################################################################################
# Define functions
################################################################################

# Function to get iNaturalist observation counts by month for a bounding box
get_inat_per_month <- function(bbox) {
  counts <- sapply(1:12, function(m) {
    get_inat_obs(
      bounds     = bbox,
      maxresults = 0,   # Keep payload minimal
      month      = m,
      taxon_name = "Aves",
      meta       = TRUE
    )$meta$found
  })
  
  tibble(month = 1:12, counts = counts)
}

################################################################################
# Load data
################################################################################

# Load mapping data
hypso <- ne_download(scale = 50, type = "HYP_50M_SR_W", category = "raster")

# Get Massachusetts state boundaries
ma <- ne_states(country = "united states of america", returnclass = "sf") |>
  filter(name == "Massachusetts")

################################################################################
# Define study areas
################################################################################

# iNat bounding boxes (format: south lat, west lon, north lat, east lon)
# Eastern study area (Boston - Urban)
bbox_E <- c(42.31, -71.14, 42.38, -71.01)
bb_E <- st_as_sfc(st_bbox(c(
  xmin = bbox_E[2], ymin = bbox_E[1],
  xmax = bbox_E[4], ymax = bbox_E[3]
), crs = 4326))

# Western study area (Pioneer Valley - Rural)
bbox_W <- c(42.31, -72.64, 42.38, -72.51)
bb_W <- st_as_sfc(st_bbox(c(
  xmin = bbox_W[2], ymin = bbox_W[1],
  xmax = bbox_W[4], ymax = bbox_W[3]
), crs = 4326))

################################################################################
# Load mortality data from iNaturalist searches
################################################################################

datE0 <- read_csv("data/search_results_E.csv") %>% 
  mutate(study_area = "East")
datW0 <- read_csv("data/search_results_W.csv") %>% 
  mutate(study_area = "West")

# Filter to marked observations and remove duplicate records
datE <- datE0 %>% 
  filter(marked == 1, month != 0) %>% 
  distinct(species, month, latitude, longitude, .keep_all = TRUE)

datW <- datW0 %>% 
  filter(marked == 1, month != 0) %>% 
  distinct(species, month, latitude, longitude, .keep_all = TRUE)

# Calculate total mortalities per area
tot_mort_E <- nrow(datE)
tot_mort_W <- nrow(datW)

cat("Total mortality observations - Urban:", tot_mort_E, "Rural:", tot_mort_W, "\n")

################################################################################
# Process data: Calculate rates and indices
################################################################################

# Get total monthly iNaturalist bird observations for each study area
obs_count_E <- get_inat_per_month(bbox = bbox_E)
obs_count_W <- get_inat_per_month(bbox = bbox_W)

# Calculate monthly mortality rates
datE_monthly <- datE %>% 
  group_by(month) %>% 
  summarize(n = n(), sr = n_distinct(species)) %>% 
  left_join(obs_count_E, by = "month") %>% 
  mutate(
    mort_rate = n / counts,
    sr_rate = sr / counts,
    study_area = "Urban"
  )

datW_monthly <- datW %>% 
  group_by(month) %>% 
  summarize(n = n(), sr = n_distinct(species)) %>% 
  left_join(obs_count_W, by = "month") %>% 
  mutate(
    mort_rate = n / counts,
    sr_rate = sr / counts,
    study_area = "Rural"
  )

# Combine study areas and calculate relative indices
dat_comb <- datE_monthly %>% 
  bind_rows(datW_monthly) %>% 
  mutate(
    month_num = month,
    month = as.factor(month),
    study_area = as.factor(study_area)
  ) %>% 
  group_by(study_area) %>% 
  mutate(
    mort_idx = log2(mort_rate / mean(mort_rate)),
    sr_per_mort = sr_rate / n,
    sr_idx = log2(sr_per_mort / mean(sr_per_mort)),
    sr_idx2 = log2(sr_rate / mean(sr_rate))
  )

################################################################################
# Visualizations
################################################################################

# Define color palette
pal <- c(Urban = "#0072B2", Rural = "#D55E00")  # Blue and orange

# Plot 1: Study area map
lab_E <- st_centroid(bb_E)
lab_W <- st_centroid(bb_W)

study_area_plot <- ggplot() +
  geom_sf(data = ma, fill = "gray", colour = "black", linewidth = 0.6) +
  geom_sf(data = bb_E, fill = NA, colour = pal[1], linewidth = 1) +
  geom_sf(data = bb_W, fill = NA, colour = pal[2], linewidth = 1) +    
  geom_sf_text(
    data = lab_E, aes(label = "Boston"),
    colour = pal[1], fontface = "bold", size = 5, nudge_y = 0.1, nudge_x = -0.1
  ) +
  geom_sf_text(
    data = lab_W, aes(label = "Pioneer Valley"),
    colour = pal[2], fontface = "bold", size = 5, nudge_y = 0.1
  ) +
  coord_sf(
    xlim = st_bbox(ma)[c("xmin", "xmax")],
    ylim = st_bbox(ma)[c("ymin", "ymax")],
    expand = FALSE
  ) +
  theme_void()

ggsave(study_area_plot, file = "outputs/study_area_fig.png")

# Plot 2: Seasonal mortality patterns
mort_plot <- ggplot(
  dat_comb, 
  aes(x = month_num, y = mort_idx, colour = study_area, group = study_area)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = 1:12, labels = month.abb, expand = c(0.01, 0)) +
  scale_y_continuous(breaks = pretty_breaks(), name = "Mortality Index") +
  scale_colour_manual(values = pal) +
  labs(
    x = NULL,
    colour = "Location",
    title = "Seasonal Patterns in Relative Bird Mortality Rates"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background = element_rect(fill = "white", colour = NA)
  )

ggsave(mort_plot, file = "outputs/mortality_plot.png")

# Plot 3: Combined figure with inset map
study_area_inset <- ggplot() +
  geom_sf(data = ma, fill = "gray", colour = "black", linewidth = 0.6) +
  geom_sf(data = bb_E, fill = NA, colour = pal[1], linewidth = 1) +
  geom_sf(data = bb_W, fill = NA, colour = pal[2], linewidth = 1) +    
  geom_sf_text(
    data = lab_E, aes(label = "Boston"),
    colour = pal[1], fontface = "bold", size = 3, nudge_x = 0.7
  ) +
  geom_sf_text(
    data = lab_W, aes(label = "Pioneer Valley"),
    colour = pal[2], fontface = "bold", size = 3, nudge_y = 0.2, nudge_x = 0.1
  ) +
  coord_sf(
    xlim = st_bbox(ma)[c("xmin", "xmax")],
    ylim = st_bbox(ma)[c("ymin", "ymax")],
    expand = FALSE
  ) +
  theme_void()

mort_plot_inset <- mort_plot +
  inset_element(
    study_area_inset,
    left = 0.1, bottom = 0.58, right = 0.38, top = 0.88,
    align_to = "full"
  )

ggsave(mort_plot_inset, file = "outputs/mortality_plot_w_inset.png", 
       width = 8, height = 6)

################################################################################
# Statistical modeling (GAM)
################################################################################

# Generalized Additive Model with cyclic smooths for seasonal patterns
mod_gam <- gam(
  cbind(n, counts - n) ~ study_area + 
    s(month_num, bs = "cc", k = 6) +                    # Overall seasonal curve
    s(month_num, by = study_area, bs = "cc", k = 6),    # Study area deviation
  family = binomial,
  data = dat_comb,
  method = "REML",
  knots = list(month_num = c(0.5, 12.5))                # Ensure cyclicity
)

cat("\nGAM Model Summary:\n")
print(summary(mod_gam))

# Generate predictions
newdat <- expand.grid(
  month_num = seq(1, 12, length.out = 200),
  study_area = c("Urban", "Rural")
)

# Predict on link scale and back-transform
p <- predict(mod_gam, newdata = newdat, type = "link", se.fit = TRUE)

newdat <- newdat %>% 
  mutate(
    fit_link = p$fit,
    se_link = p$se.fit,
    prob = mod_gam$family$linkinv(fit_link),
    lower = mod_gam$family$linkinv(fit_link - 1.96 * se_link),
    upper = mod_gam$family$linkinv(fit_link + 1.96 * se_link)
  )

# Plot GAM predictions
gam_plot <- ggplot(newdat, aes(month_num, prob, colour = study_area, fill = study_area)) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              alpha = 0.25, linewidth = 0, show.legend = FALSE) +
  geom_line(linewidth = 1) +
  scale_x_continuous(breaks = 1:12, labels = month.abb, expand = c(0.01, 0)) +
  labs(
    x = NULL,
    y = "Mortality probability per iNat observation",
    colour = NULL
  ) +
  coord_cartesian(ylim = c(0, max(newdat$upper) * 1.05)) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

ggsave(gam_plot, file = "outputs/gam_predictions.png", width = 8, height = 6)

cat("\nAnalysis complete! All outputs saved to outputs/ directory.\n")
