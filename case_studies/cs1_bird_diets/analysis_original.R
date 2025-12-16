################################################################################
####                                                                        ####
####              Case Study 1: Bird Dietary Preferences                   ####
####            Seasonal variation in avian diet composition                ####
####                                                                        ####
################################################################################

# Description: This script analyzes seasonal dietary patterns in five bird 
# species using iNaturalist search results. Compares observed diet composition
# with reference data from SAviTraits database.

# Load required libraries
library(tidyverse)
library(ggplot2)
library(janitor)

# Read in raw exports
# Define species and dietary categories
species <- c("Catharus_minimus", "Melanerpes_carolinus", "Spizelloides_arborea",
             "Synthliboramphus_antiquus", "Turdus_migratorius")
diet <- c("Invertebrate", "Seed", "Fruit", "Vertebrate", "Carrion", "Nectar", "PlantOther")
season <- c("Jun-Aug", "Dec-Feb")

# Initialize data structure
sp.diet <- list()

# Load and combine all diet data across species, diet types, and seasons
for (i in 1:length(species)) {
  for (j in 1:length(diet)) {
    for (k in 1:length(season)) {
      dat <- read.csv(paste0("data/", species[i], "_", diet[j], "_", season[k], ".csv"))
      
      # Limit to 500 records per combination
      if (nrow(dat) > 500) {
        dat <- dat[1:500, ]
      }
      
      # Add metadata columns
      dat$species2 <- species[i]
      dat$diet <- diet[j]
      dat$season <- season[k]
      
      sp.diet <- bind_rows(sp.diet, dat)
    }
  }
}

################################################################################
# 1. Calculate coverage: percentage of positive marks by species
################################################################################
counts <- data.frame(species=character(), ones=integer(), zeroes=integer(), total=integer(), perc_positive=numeric())

for (i in 1:length(species)){
  sp.i <- species[i]
  dat <- sp.diet %>% filter(species2 == sp.i)
  ones <- nrow(dat[dat$marked == 1,])
  zeroes <- nrow(dat[dat$marked == 0,])
  total <- nrow(dat)
  perc_positive <- (ones/total)*100
  counts[i,1] <- sp.i
  counts[i,2] <- ones
  counts[i,3] <- zeroes
  counts[i,4] <- total
  counts[i,5] <- perc_positive
}

p <- ggplot(counts, aes(x=species, y=perc_positive, fill=species)) + 
  geom_bar(stat = "identity", width=0.5) +
  theme(#panel.background = element_rect(fill = 'white', colour = 'grey85'),
    panel.border = element_rect(fill=NA, colour = "white", size=1),
    axis.line = element_line(color = 'black', size=1.5),
    plot.title = element_text(size=15, vjust=2, family="sans"),
    axis.text.x = element_blank(),
    #axis.text.x = element_text(colour='black',size=22),
    axis.text.y = element_text(colour='black',size=15),
    axis.title.x = element_text(colour='black',size=15),
    axis.title.y = element_text(colour='black',size=15),
    #axis.title.x = element_blank(),
    #axis.title.y = element_blank(),
    axis.ticks = element_line(color = 'black', size=1.5),
    axis.ticks.length=unit(0.3,"cm"),
    legend.position="right",
    legend.text=element_text(size=15),
    legend.title=element_blank(),
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA))
  
ggsave(p, file="./graphics/perc-positive.png", bg="transparent", width=8, height=6, dpi=600)


p <- ggplot(counts, aes(x=species, y=ones, fill=species)) + 
  geom_bar(stat = "identity", width=0.5) +
  theme(#panel.background = element_rect(fill = 'white', colour = 'grey85'),
    panel.border = element_rect(fill=NA, colour = "white", size=1),
    axis.line = element_line(color = 'black', size=1.5),
    plot.title = element_text(size=15, vjust=2, family="sans"),
    axis.text.x = element_blank(),
    #axis.text.x = element_text(colour='black',size=22),
    axis.text.y = element_text(colour='black',size=15),
    axis.title.x = element_text(colour='black',size=15),
    axis.title.y = element_text(colour='black',size=15),
    #axis.title.x = element_blank(),
    #axis.title.y = element_blank(),
    axis.ticks = element_line(color = 'black', size=1.5),
    axis.ticks.length=unit(0.3,"cm"),
    legend.position="right",
    legend.text=element_text(size=15),
    legend.title=element_blank(),
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA))

ggsave(p, file="./graphics/marked-positive.png", bg="transparent", width=8, height=6, dpi=600)

#get count of positive marks, per season
counts.sum <- data.frame(species=character(), season = character(), ones=integer(), zeroes=integer(), total=integer(), perc_positive=numeric())
counts.win <- data.frame(species=character(), season = character(), ones=integer(), zeroes=integer(), total=integer(), perc_positive=numeric())

for (i in 1:length(species)){
  sp.i <- species[i]
  dat <- sp.diet %>% filter(species2 == sp.i)
  dat.sum <- dat %>% filter(season == "Jun-Aug")
  ones <- nrow(dat.sum[dat.sum$marked == 1,])
  zeroes <- nrow(dat.sum[dat.sum$marked == 0,])
  total <- nrow(dat.sum)
  perc_positive <- (ones/total)*100
  counts.sum[i,1] <- sp.i
  counts.sum[i,2] <- "Jun-Aug"
  counts.sum[i,3] <- ones
  counts.sum[i,4] <- zeroes
  counts.sum[i,5] <- total
  counts.sum[i,6] <- perc_positive
}

for (i in 1:length(species)){
  sp.i <- species[i]
  dat <- sp.diet %>% filter(species2 == sp.i)
  dat.win <- dat %>% filter(season == "Dec-Feb")
  ones <- nrow(dat.win[dat.win$marked == 1,])
  zeroes <- nrow(dat.win[dat.win$marked == 0,])
  total <- nrow(dat.win)
  perc_positive <- (ones/total)*100
  counts.win[i,1] <- sp.i
  counts.win[i,2] <- "Dec-Feb"
  counts.win[i,3] <- ones
  counts.win[i,4] <- zeroes
  counts.win[i,5] <- total
  counts.win[i,6] <- perc_positive
}

counts.all <- counts.sum %>%
  bind_rows(counts.win)

p <- ggplot(counts.all, aes(x=species, y=ones, fill=season)) + 
  geom_bar(stat = "identity", width=0.5, position=position_dodge(width=0.9)) +
  theme(#panel.background = element_rect(fill = 'white', colour = 'grey85'),
    panel.border = element_rect(fill=NA, colour = "white", size=1),
    axis.line = element_line(color = 'black', size=1.5),
    plot.title = element_text(size=15, vjust=2, family="sans"),
    #axis.text.x = element_blank(),
    axis.text.x = element_text(colour='black',size=15, angle=90),
    axis.text.y = element_text(colour='black',size=15),
    axis.title.x = element_text(colour='black',size=15),
    axis.title.y = element_text(colour='black',size=15),
    #axis.title.x = element_blank(),
    #axis.title.y = element_blank(),
    axis.ticks = element_line(color = 'black', size=1.5),
    axis.ticks.length=unit(0.3,"cm"),
    legend.position="right",
    legend.text=element_text(size=15),
    legend.title=element_blank(),
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA))

ggsave(p, file="./graphics/marked-positive-byseason.png", bg="transparent", width=8, height=6, dpi=600)






#get percent diet for species that have some data 
#(only three: Melanerpes carolinus, Spizelloides arborea, Turdus migratorius)
species.sub <- c("Melanerpes_carolinus","Spizelloides_arborea",
             "Turdus_migratorius")
diet <- c("Invertebrate","Seed","Fruit","Vertebrate","Carrion","Nectar","PlantOther")
sp.diet.new <- list()
perc.diet <- data.frame(species=character(), season=character(), diet=character(), perc=numeric())

for (i in 1:length(species.sub)){
  sp.i <- species.sub[i]
  dat <- sp.diet %>% filter(species2 == sp.i)

  for (k in 1:length(season)){
    seas <- season[k]
    dat.s <- dat %>% filter(season == seas)
    dat.s <- dat.s %>% filter(marked == 1)
    for (j in 1:length(diet)){
      diet.j <- nrow(dat.s[dat.s$diet == diet[j],])
      perc.j <- (diet.j/nrow(dat.s))*100
      perc.diet[1,1] <- sp.i
      perc.diet[1,2] <- seas
      perc.diet[1,3] <- diet[j]
      perc.diet[1,4] <- perc.j
      sp.diet.new <- bind_rows(sp.diet.new,perc.diet)
    }
  }
}


#read in SaviTraits
require(janitor)
raw_trait_df <- read.csv('./data/SAviTraits_1-0_1.csv') %>% 
  #Trimming out unnecessary variables
  dplyr::select(Species_Scientific_Name, Diet_Sub_Cat, 
                Jan, Feb, Mar, Apr, May, Jun, 
                Jul, Aug, Sep, Oct, Nov, Dec)

#Collapses the vertebrate diet groups into a single vertebrate diet class to reduce the number of traits analysed and because most of the diet proportions lie in the "unknown vertebrate category anyway
full_trait_df <- raw_trait_df %>% 
  
  #Preparing the data for distance matrix and manipulation by reformatting to make diets columns and months rows
  pivot_longer(cols=c(Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec),
               names_to = "Month",
               values_to = "Proportion") %>% 
  pivot_wider(names_from = "Diet_Sub_Cat",
              values_from = "Proportion") %>% 
  
  #Collapsing the vertebrate diet estimates
  mutate(Vertebrate = Ectotherms + Endotherms + Fish + Unknown) %>% 
  dplyr::select(-c(Ectotherms, Endotherms, Fish, Unknown)) %>% 
  
  #Changing species names to match the naming in our range files
  mutate(Species = str_replace(Species_Scientific_Name, " ", "_"),
         id_uniq = paste(Species, Month, sep = "_")) %>% 
  
  #Standardising naming format
  clean_names() 

# select the five species
species2 <- c("Catharus_minimus","Melanerpes_carolinus","Spizelloides_arborea",
             "Synthliboramphus_antiquus","Turdus_migratorius")

savitraits.sub <- full_trait_df %>% 
  filter(species %in% species2)

#prep summer and winter only
months <- c("Jan","Feb","Jun","Jul","Aug","Dec")

savitraits.sub <- savitraits.sub %>%
  filter(month %in% months)

#average over summer and winter months
savi.sub.means <- data.frame(species=character(), season=character(), 
                             Invertebrate=character(),Carrion=character(),Fruit=character(),Nectar=character(),
                             Seed=character(),PlantOther=character(),Vertebrate=character())
savi.sub.means.all <- list()
month.sum <- c("Jun","Jul","Aug")
month.win <- c("Jan","Feb","Dec")

for (i in 1:length(species2)){
  sp.i <- species2[i]
  savi.i <- savitraits.sub %>% filter(species == sp.i)
  savi.sum <- savi.i %>% filter(month %in% month.sum)
  savi.win <- savi.i %>% filter(month %in% month.win)
  mean.sum <- apply(savi.sum[, 3:9],2,mean)
  mean.win <- apply(savi.win[, 3:9],2,mean)
  savi.sub.means[1,1] <- sp.i
  savi.sub.means[1,2] <- "Jun-Aug"
  savi.sub.means[2,1] <- sp.i
  savi.sub.means[2,2] <- "Dec-Feb"
  savi.sub.means[1,3:9] <- mean.sum
  savi.sub.means[2,3:9] <- mean.win
  savi.sub.means.all <- bind_rows(savi.sub.means.all,savi.sub.means)
}

savi.sub.all <- savi.sub.means.all %>%
  select(species, season, Invertebrate, Seed, Fruit, Vertebrate, Carrion, Nectar, PlantOther) %>%
  pivot_longer(cols=c(Invertebrate, Seed, Fruit, Vertebrate, Carrion, Nectar, PlantOther),
               names_to = "diet",
               values_to = "perc_savi") %>%
  mutate(perc_savi = as.numeric(perc_savi))

sp.diet.new <- sp.diet.new %>%
  mutate(perc_inat = perc) %>%
  select(species, season, diet, perc_inat)

#merge both datasets
species.sub <- c("Melanerpes_carolinus","Spizelloides_arborea",
                 "Turdus_migratorius")
savi.sub.3 <- savi.sub.all %>%
  filter(species %in% species.sub)

all.data <- savi.sub.3 %>% 
  full_join(sp.diet.new, by = c("species", "season", "diet"))

all.data.long <- all.data %>%
  pivot_longer(cols=c(perc_savi, perc_inat),
               names_to = "data_type",
               values_to = "perc")

#plot
#summer
all.data.sum <- all.data.long %>%
  filter(season == "Jun-Aug")

p <- ggplot(all.data.sum, aes(x=diet, y=perc, fill=data_type)) + 
  geom_bar(stat = "identity", width=0.5, position=position_dodge(width=0.9)) + 
  facet_grid(~species) +
  theme(#panel.background = element_rect(fill = 'white', colour = 'grey85'),
    panel.border = element_rect(fill=NA, colour = "white", size=1),
    axis.line = element_line(color = 'black', size=1.5),
    plot.title = element_text(size=15, vjust=2, family="sans"),
    #axis.text.x = element_blank(),
    axis.text.x = element_text(colour='black',size=15, angle=90),
    axis.text.y = element_text(colour='black',size=15),
    axis.title.x = element_text(colour='black',size=15),
    axis.title.y = element_text(colour='black',size=15),
    #axis.title.x = element_blank(),
    #axis.title.y = element_blank(),
    axis.ticks = element_line(color = 'black', size=1.5),
    axis.ticks.length=unit(0.3,"cm"),
    legend.position="right",
    legend.text=element_text(size=15),
    legend.title=element_blank(),
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA))

ggsave(p, file="./graphics/diet-comparison-summer.png", bg="transparent", width=8, height=6, dpi=600)

#winter
all.data.win <- all.data.long %>%
  filter(season == "Dec-Feb")

p <- ggplot(all.data.win, aes(x=diet, y=perc, fill=data_type)) + 
  geom_bar(stat = "identity", width=0.5, position=position_dodge(width=0.9)) + 
  facet_grid(~species) +
  theme(#panel.background = element_rect(fill = 'white', colour = 'grey85'),
    panel.border = element_rect(fill=NA, colour = "white", size=1),
    axis.line = element_line(color = 'black', size=1.5),
    plot.title = element_text(size=15, vjust=2, family="sans"),
    #axis.text.x = element_blank(),
    axis.text.x = element_text(colour='black',size=15, angle=90),
    axis.text.y = element_text(colour='black',size=15),
    axis.title.x = element_text(colour='black',size=15),
    axis.title.y = element_text(colour='black',size=15),
    #axis.title.x = element_blank(),
    #axis.title.y = element_blank(),
    axis.ticks = element_line(color = 'black', size=1.5),
    axis.ticks.length=unit(0.3,"cm"),
    legend.position="right",
    legend.text=element_text(size=15),
    legend.title=element_blank(),
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA))

ggsave(p, file="./graphics/diet-comparison-winter.png", bg="transparent", width=8, height=6, dpi=600)

