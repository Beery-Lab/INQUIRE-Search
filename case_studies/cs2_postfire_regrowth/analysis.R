################################################################################
####                                                                        ####
####        Case Study 2: Post-Fire Vegetation Regrowth                    ####
####     Coniferous vs. deciduous recovery in the High Park Fire           ####
####                                                                        ####
################################################################################

# Description: This script analyzes post-fire vegetation regrowth patterns in
# the High Park Fire (Colorado, 2012) using iNaturalist observations. Compares
# coniferous and deciduous tree/shrub recovery across burn severity gradients.

################################################################################
# Setup
################################################################################

# Load required libraries
library(tidyverse)
library(terra)
library(sf)
library(ggplot2)
library(mapview)
library(tmap)
library(gridExtra)
library(cowplot)

################################################################################
# Load and prepare fire data
################################################################################

# Load MTBS (Monitoring Trends in Burn Severity) fire perimeter data
mtbs <- st_read('gis/mtbs_perimeter_data_firebundle/mtbs_perims_DD.shp')

# Filter to suitable fires: 2011-2017, large fires in AK/CO/WY
fires <- mtbs %>% 
  mutate(
    Ig_year = as.numeric(substr(Ig_Date, start = 1, stop = 4)),
    State = substr(Event_ID, start = 1, stop = 2)
  ) %>% 
  dplyr::filter(
    Ig_year >= 2011, 
    Ig_year <= 2017,
    State %in% c('AK', 'CO', 'WY'),
    BurnBndAc >= as.numeric(quantile(mtbs$BurnBndAc, .3))
  )

cat("Filtered to", nrow(fires), "candidate fires\n")

# Select High Park Fire (2012, Colorado)
highpark <- fires %>% dplyr::filter(Incid_Name == 'HIGH PARK')

# Extract bounding box coordinates
bbox_coords <- st_bbox(highpark)
cat("High Park Fire bounds:\n")
cat("Latitude:", bbox_coords[c('ymin', 'ymax')], "\n")
cat("Longitude:", bbox_coords[c('xmin', 'xmax')], "\n")

################################################################################
# Process burn severity data
################################################################################

# Load burn severity raster for Colorado 2012
# Categories: 1=unburned to low, 2=low, 3=moderate, 4=high, 
#             5=increased greenness, 6=non-mapping area
bs <- rast('gis/composite_data/MTBS_BSmosaics/2012/mtbs_CO_2012.tif')
bs <- crop(bs, vect(st_transform(highpark, st_crs(bs))))

# Remove non-mapping areas (category 6)
bs <- ifel(bs == 6, NA, bs)

# Project to WGS84 for coordinate matching
bs_wgs <- project(bs, 'epsg:4326', method = 'near')

# Transform fire perimeter to match
highpark <- st_transform(highpark, crs = 4326)

# Define color palette for burn severity categories
mtbs_cat_cols <- c('#006400', '#7fffd4', '#ffff00', '#ff0000', '#7fff00')

################################################################################
# Aggregate burn severity to 0.01 degree resolution
################################################################################
# (Matches iNaturalist coordinate precision)

# Create 0.01 x 0.01 degree grid
res_deg <- 0.01
fishnet <- rast(
  xmin = floor(xmin(bs_wgs) * 100) / 100 - 0.005, 
  xmax = ceiling(xmax(bs_wgs) * 100) / 100 + 0.005, 
  ymin = floor(ymin(bs_wgs) * 100) / 100 - 0.005, 
  ymax = ceiling(ymax(bs_wgs) * 100) / 100 + 0.005,
  resolution = res_deg, 
  crs = "EPSG:4326"
)
values(fishnet) <- 1:ncell(fishnet)

# Resample burn severity using mode (better captures landscape patterns)
bsmode <- resample(bs_wgs, fishnet, method = 'mode')

# Create binary burn area mask (1=burned, 0=not burned)
bs_wgs01 <- ifel(bs_wgs > 0, 1, 0)
bs_wgs01 <- ifel(is.na(bs_wgs), 0, 1)

################################################################################
# Create Figure 1: Study area and burn severity maps
################################################################################

# Load base map data
western_states <- st_read('gis/western_states.shp') %>% 
  st_transform(crs = st_crs(highpark))
colorado <- western_states %>% dplyr::filter(NAME %in% 'Colorado')

tmap_mode("plot")

# Define burn severity labels
bs_colors <- c("lightblue", 'yellow', "orange", "red", 'darkblue')
bs_labels <- c(
  "1" = "Unburned to Low",
  "2" = "Low",
  "3" = "Moderate",
  "4" = "High",
  "5" = "Increased Greenness"
)

# Panel A: Western US context map
fig1a <- tm_shape(western_states) +
  tm_polygons(col = "gray90", border.col = "white") +
  tm_shape(colorado) +
  tm_polygons(col = "gray70", border.col = "black", lwd = 1.5) +
  tm_shape(highpark) +
  tm_polygons(col = "red", fill = 'red', fill_alpha = 1) +
  tm_title(text = "A) High Park Fire Location", size = 1.2) +
  tm_layout(frame = TRUE, outer.margins = 0) +
  tm_scalebar(
    position = c("LEFT", "BOTTOM"),
    breaks = c(0, 250, 500),
    text.size = 0.5
  )

# Panel B: 30m resolution burn severity
fig1b <- tm_shape(bs_wgs, bbox = highpark) +
  tm_raster(col.scale = tm_scale_categorical(values = bs_colors, labels = bs_labels)) +
  tm_shape(highpark) +
  tm_borders(col = "black", lwd = 1) +
  tm_title(text = "B) 30m Burn Severity", size = 1.2) +
  tm_layout(frame = TRUE, legend.show = FALSE, outer.margins = 0) +
  tm_scalebar(
    position = c("LEFT", "TOP"),
    breaks = c(0, 1, 2),
    text.size = 0.5
  )

# Panel C: Aggregated burn severity
fig1c <- tm_shape(bsmode, bbox = highpark) +
  tm_raster(col.scale = tm_scale_categorical(values = bs_colors, labels = bs_labels)) +
  tm_shape(highpark) +
  tm_borders(col = "black", lwd = 1) +
  tm_title(text = "C) Aggregated Burn Severity", size = 1.2) +
  tm_layout(frame = TRUE, legend.show = FALSE, outer.margins = 0) +
  tm_scalebar(
    position = c("LEFT", "TOP"),
    breaks = c(0, 1, 2),
    text.size = 0.5
  )

# Standalone legend
legend_fig <- tm_shape(bs_wgs) +
  tm_raster(
    col.scale = tm_scale_categorical(values = bs_colors, labels = bs_labels),
    col.legend = tm_legend(
      title = "Burn Severity", 
      text.size = 1, 
      title.size = 1.2,
      position = tmap::tm_pos_in("left", "center"),
      frame = FALSE, 
      frame.lwd = 0, 
      frame.col = 'white'
    )
  ) +
  tm_layout(legend.only = TRUE, outer.margins = 0)

# Combine into final figure
g1a <- tmap_grob(fig1a)
g1b <- tmap_grob(fig1b)
g1c <- tmap_grob(fig1c)
g_legend <- tmap_grob(legend_fig)

col1 <- arrangeGrob(g1a, g_legend, ncol = 1, heights = c(2, 1),
                    padding = unit(0, "mm"))
col2 <- arrangeGrob(g1b, g1c, ncol = 1, padding = unit(0, "mm"))

grid.arrange(col1, col2, ncol = 2, widths = c(1, 1.2), padding = unit(0, "mm"))

# Save figure (uncomment to export)
# jpeg("outputs/Figure1_Map_burnseverity.jpg", width = 8, height = 6, 
#      units = "in", res = 400)
# grid.arrange(col1, col2, ncol = 2, widths = c(1, 1.2), padding = unit(0, "mm"))
# dev.off()

################################################################################
# Process coniferous tree data
################################################################################
# Search query: "young coniferous trees in burned forest"

conifers <- read_csv('data/HighPark_search_results_coniferous.csv')
conifers$ID <- 1:nrow(conifers)

# Convert to spatial object
conifers <- st_as_sf(conifers, coords = c("longitude", "latitude"), crs = 4326)

# Flag points within burn perimeter
inside <- st_within(conifers, highpark, sparse = FALSE)
conifers$InBurnArea <- as.integer(apply(inside, 1, any))

# Extract burn severity values at photo locations
extracted_bs <- extract(bsmode, vect(conifers), ID = FALSE)
conifers$burnseverity <- extracted_bs$mtbs_CO_2012

# Export processed data (uncomment to save)
# write.csv(
#   cbind(st_drop_geometry(conifers), 
#         longitude = st_coordinates(conifers)[, 1], 
#         latitude = st_coordinates(conifers)[, 2]), 
#   "data/Processed_HighPark_search_results_coniferous.csv", 
#   row.names = FALSE
# )

################################################################################
# Process deciduous tree data
################################################################################
# Search query: "young deciduous trees in burned forest"

decid <- read_csv('data/HighPark_search_results_deciduous.csv')
decid$ID <- 1:nrow(decid)

# Convert to spatial object
decid <- st_as_sf(decid, coords = c("longitude", "latitude"), crs = 4326)

# Extract burn area membership
extracted_burnarea <- extract(bs_wgs01, vect(decid), ID = FALSE)
decid$InBurnArea <- extracted_burnarea$mtbs_CO_2012

# Extract burn severity values
extracted_bs <- extract(bsmode, vect(decid), ID = FALSE)
decid$burnseverity <- extracted_bs$mtbs_CO_2012

# Export processed data (uncomment to save)
# write.csv(
#   cbind(st_drop_geometry(decid),
#         longitude = st_coordinates(decid)[, 1],
#         latitude = st_coordinates(decid)[, 2]),
#   "data/Processed_HighPark_search_results_deciduous.csv", 
#   row.names = FALSE
# )

################################################################################
# Filter and validate observations
################################################################################

# Load manually verified data (after manual classification in Excel)
conifers2 <- read_csv('data/Processed_HighPark_search_results_coniferous.csv')
conifers2 <- st_as_sf(conifers2, coords = c("longitude", "latitude"), crs = 4326)

decid2 <- read_csv('data/Processed_HighPark_search_results_deciduous.csv')
decid2 <- st_as_sf(decid2, coords = c("longitude", "latitude"), crs = 4326)

# Re-validate burn area membership
extracted_burnarea <- extract(bs_wgs01, vect(conifers2), ID = FALSE)
conifers2$InBurnArea2 <- extracted_burnarea$mtbs_CO_2012
conifers2 <- conifers2 %>% 
  mutate(InBurnArea2 = ifelse(is.na(InBurnArea2), 0, InBurnArea2))

cat("\nConifers: Photos outside burn area:", 
    sum(conifers2$InBurnArea2 == 0), "\n")

# Filter to photos within burn perimeter
conifers3 <- conifers2 %>% dplyr::filter(InBurnArea2 %in% 1)

# Keep juniper (coniferous shrub) with coniferous trees
conifers3 <- conifers3 %>% 
  mutate(Coniferous = ifelse(Coniferous == 0 & Notes == 'Juniper', 1, Coniferous)) %>%
  dplyr::filter(Coniferous == 1)

cat("Conifer observations by burn severity:\n")
print(table(conifers3$burnseverity))

# Process deciduous data
decid2 <- decid2 %>% 
  mutate(InBurnArea = ifelse(is.na(InBurnArea), 0, InBurnArea))

cat("\nDeciduous: Photos outside burn area:", 
    sum(decid2$InBurnArea == 0), "\n")

# Filter to photos within burn perimeter
decid3 <- decid2 %>% dplyr::filter(InBurnArea %in% 1)

# Include shrubs with deciduous trees (difficult to distinguish when young)
decid3 <- decid3 %>% 
  mutate(Deciduous = ifelse(Deciduous == 0 & Notes %in% c('Shrub'), 1, Deciduous)) %>%
  dplyr::filter(Deciduous == 1)

cat("Deciduous observations by burn severity:\n")
print(table(decid3$burnseverity))

################################################################################
# Create summary figures
################################################################################

# Prepare data for visualization
conifers4 <- conifers3 %>% 
  mutate(veg_type = "Coniferous") %>% 
  dplyr::select(photo_id, species, ID, burnseverity, veg_type)

decid4 <- decid3 %>% 
  mutate(veg_type = "Deciduous") %>% 
  dplyr::select(photo_id, species, ID, burnseverity, veg_type)

# Combine datasets
combined_df <- bind_rows(conifers4, decid4)

# Add descriptive burn severity labels
bs_df <- data.frame(
  burnseverity = seq(1, 5), 
  burnseverity_category = c('Unburned to low', 'Low', 'Moderate', 'High', 
                            'Increased Greenness')
)
bs_df$burnseverity_category <- factor(
  bs_df$burnseverity_category,
  levels = c('Unburned to low', 'Low', 'Moderate', 'High', 'Increased Greenness')
)

combined_df <- merge(combined_df, bs_df, by = 'burnseverity') %>% as_tibble()

# Calculate percentages for each vegetation type
percent_df <- combined_df %>%
  group_by(veg_type, burnseverity_category) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(veg_type) %>%
  mutate(percent = n / sum(n) * 100)

# Panel A: Count of images by burn severity and vegetation type
fig_nphotos <- ggplot(combined_df, aes(x = burnseverity_category, fill = veg_type)) +
  geom_bar(position = "dodge") +
  labs(
    title = 'A)',
    x = "",
    y = "Number of Images"
  ) +
  scale_fill_manual(values = c("Coniferous" = "#006400", "Deciduous" = "#90EE90")) +
  theme_bw() +
  theme(legend.position = "none", text = element_text(size = 12))

# Panel B: Percentage of images by burn severity and vegetation type
fig_pphotos <- ggplot(percent_df, aes(x = burnseverity_category, y = percent, fill = veg_type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = 'B)',
    x = "Burn Severity",
    y = "Percent of Images",
    fill = "Forest Type"
  ) +
  scale_fill_manual(values = c("Coniferous" = "#006400", "Deciduous" = "#90EE90")) +
  theme_bw() +
  theme(legend.position = "bottom", text = element_text(size = 12))

# Extract legend
get_legend2 <- function(plot, legend = NULL) {
  if (is_ggplot(plot)) {
    gt <- ggplotGrob(plot)
  } else if (is.grob(plot)) {
    gt <- plot
  } else {
    stop("Plot object is neither a ggplot nor a grob.")
  }
  
  pattern <- "guide-box"
  if (!is.null(legend)) {
    pattern <- paste0(pattern, "-", legend)
  }
  
  indices <- grep(pattern, gt$layout$name)
  not_empty <- !vapply(
    gt$grobs[indices], 
    inherits, what = "zeroGrob", 
    FUN.VALUE = logical(1)
  )
  indices <- indices[not_empty]
  
  if (length(indices) > 0) {
    return(gt$grobs[[indices[1]]])
  }
  return(NULL)
}

legend <- get_legend2(fig_pphotos)
fig_pphotos_nolegend <- fig_pphotos + theme(legend.position = "none")

# Combine plots with shared legend
final_plot <- plot_grid(
  fig_nphotos, fig_pphotos_nolegend,
  nrow = 1,
  align = "v"
)

final_with_legend <- plot_grid(final_plot, legend, ncol = 1, rel_heights = c(1, 0.1))

print(final_with_legend)

# Save figure (uncomment to export)
# jpeg('outputs/Figure3_summaryvalues.jpg', width = 7, height = 3, 
#      units = 'in', res = 500)
# print(final_with_legend)
# dev.off()

################################################################################
# Identify mixed forest sites
################################################################################

# Find photos that show both coniferous and deciduous vegetation
mixedforest <- merge(
  conifers3 %>% dplyr::select(photo_id, Coniferous) %>% st_drop_geometry(), 
  decid3 %>% dplyr::select(photo_id, Deciduous, burnseverity) %>% st_drop_geometry(),
  by = 'photo_id'
)

cat("\nMixed forest observations:", nrow(mixedforest), "\n")
cat("Distribution by burn severity:\n")
print(table(mixedforest$burnseverity))

################################################################################
# Summary notes
################################################################################

cat("\n=== Processing Notes ===\n")
cat("Coniferous processing:\n")
cat("  - Juniper (coniferous shrub) grouped with coniferous trees\n")
cat("  - Difficult to distinguish from distance\n\n")

cat("Deciduous processing:\n")
cat("  - Young trees and shrubs grouped together (too similar)\n")
cat("  - Included alders and willows as trees\n")
cat("  - Excluded sagebrush (easier to distinguish)\n")
cat("  - Excluded vines (grape, clematis)\n")

cat("\nAnalysis complete! All outputs saved to outputs/ directory.\n")
