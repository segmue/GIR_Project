library(dplyr)
library(sf)
library(ggplot2)

acled_eu <- read.csv("data/Europe-Central-Asia_2018-2024_Nov22.csv")
acled_af <- read.csv("data/Africa_1997-2024_Nov29.csv")
acled_me <- read.csv("data/MiddleEast_2015-2024_Nov29.csv")
acled_sa <- read.csv("data/LatinAmerica_2018-2024_Dec06.csv")
acled_asia <- read.csv("data/Asia-Pacific_2018-2024_Dec06.csv")
acled_usa <- read.csv("data/USA_Canada_2020_2024_Dec06.csv")

library(dplyr)

combine_data <- function(df_list) {
  # Apply column selection to each data frame in the list
  dfs <- lapply(df_list, function(df) select(df, event_id_cnty, longitude, latitude))
  
  # Combine all data frames into one
  combined_df <- bind_rows(dfs)
  
  return(combined_df)
}

df_list <- list(acled_eu, acled_af, acled_me, acled_sa, acled_asia, acled_usa)
combined_acled <- combine_data(df_list)


# Replace empty strings with NA only in character columns
preprocess_data <- function(data) {
  data <- st_as_sf(data, coords = c("longitude","latitude"))
  st_crs(data) <- 4326
  data <- data[!st_is_empty(data), ]
  
  return(data)
}

## Weiter im Takt: ----------

check_location_in_notes <- function(data) {
  # Identify rows where 'location' is in 'notes'
  matches <- data[apply(data, 1, function(row) grepl(row["location"], row["notes"])), ]
  
  # Identify rows where 'location' is not in 'notes'
  non_matches <- data[!apply(data, 1, function(row) grepl(row["location"], row["notes"])), ]
  
  # Print counts
  cat("Matches:", nrow(matches), "\n")
  cat("Non-Matches:", nrow(non_matches), "\n")
  
  return(non_matches)
}

unmatched_locs <- check_location_in_notes(acled_eu)


## Visualizations -----------
filter_data <- function(data){
  data %>% filter(event_type %in% c("Battles", "Explosions/Remote violence", "Violence against civilians"))
}

library(sf)
library(ggplot2)
library(ggspatial)

plot_points <- function(sf_df, alpha = 0.1, size = 0.5, reproject = TRUE, crs = 3857, 
                        basemap = "cartolight", limit_x = NULL, limit_y = NULL, zoom = 4, title = title) {
  # Extract coordinates in the source CRS (WGS 84 assumed)
  coords <- st_coordinates(sf_df)
  sf_df$lon <- coords[, 1]
  sf_df$lat <- coords[, 2]
  
  # Apply limits if specified
  if (!is.null(limit_x)) {
    sf_df <- sf_df[sf_df$lon >= limit_x[1] & sf_df$lon <= limit_x[2], ]
  }
  if (!is.null(limit_y)) {
    sf_df <- sf_df[sf_df$lat >= limit_y[1] & sf_df$lat <= limit_y[2], ]
  }
  
  # Reproject if required
  if (reproject) {
    sf_df <- st_transform(sf_df, crs = crs)
  }
  
  # Extract coordinates again after reprojection
  coords <- st_coordinates(sf_df)
  
  # Create the plot
  ggplot() +
    #annotation_map_tile(zoom = zoom, type = basemap) +  # Add basemap
    geom_point(aes(x = coords[,1], y = coords[,2]), alpha = alpha, size = size, color = "darkred") +
    theme_minimal() +
    labs(title = title,
         x = "",
         y = "")+
    coord_map("mollweide")  # Apply Mollweide projection
}

# 
# acled_af_filtered <- preprocess_data(acled_af) %>% filter_data()
# m1 <- plot_points(acled_af_filtered,alpha = 0.06, size = 0.3, title = "")
# ggsave("plots/acled_af_filtered.png", m1, width = 10, height = 8, dpi = 600)
# 
# acled_af_unfiltered <- preprocess_data(acled_af)
# m1 <- plot_points(acled_af_unfiltered,alpha = 0.06, size = 0.3, title = "")
# ggsave("plots/acled_af_unfiltered.png", m1, width = 10, height = 8, dpi = 600)
# 
# 
# acled_me_filtered <- preprocess_data(acled_me) %>% filter_data()
# m1 <- plot_points(acled_me_filtered,alpha = 0.02, size = 0.3, limit_x = c(30,62), zoom = 5, title = "")
# ggplot2::ggsave("plots/acled_me_filtered.png", m1,width = 10, height = 8, dpi = 600)
# 
# acled_me_unfiltered <- preprocess_data(acled_me)
# m1 <- plot_points(acled_me_unfiltered,alpha = 0.02, size = 0.3, limit_x = c(30,62), zoom = 5, title = "")
# ggplot2::ggsave("plots/acled_me_unfiltered.png", m1,width = 10, height = 8, dpi = 600)
# 
# 
# acled_eu_filtered <- preprocess_data(acled_eu) %>% filter_data()
# m1 <- plot_points(acled_eu_filtered,alpha = 0.02, size = 0.3, limit_x = c(-10,60), limit_y = c(30,62), title= "ACLED Data Europe", zoom = 4);m1
# ggplot2::ggsave("plots/acled_eu_filtered.png", m1, width = 10, height = 8, dpi = 600)
# 
# acled_eu_unfiltered <- preprocess_data(acled_eu)
# m1 <- plot_points(acled_eu_unfiltered,alpha = 0.02, size = 0.3, limit_x = c(-10,60), limit_y = c(30,62), title= "ACLED Data Europe", zoom = 5);m1
# ggplot2::ggsave("plots/acled_eu_unfiltered.png", m1,width = 10, height = 8, dpi = 600)
# 
# m1 <- plot_points(preprocess_data(acled_sa),alpha = 0.02, size = 0.3, title= "", zoom = 4);m1
# ggplot2::ggsave("plots/acled_sa_unfiltered.png", m1,width = 10, height = 8, dpi = 600)
# 
# m1 <- plot_points(preprocess_data(acled_asia),alpha = 0.02, size = 0.3, title= "", zoom = 4, limit_x = c(60,150), limit_y = c(-45,45));m1
# ggplot2::ggsave("plots/acled_asia_unfiltered.png", m1,width = 10, height = 8, dpi = 600)
# 

library(ggplot2)
library(sf)

# Function to plot points on a map with Mollweide projection
plot_points_mollweide <- function(sf_df, alpha = 0.5, size = 1, title = NULL, zoom = 4) {
  # Ensure that the CRS is WGS84 (EPSG:4326)
  if (st_crs(sf_df) != 4326) {
    sf_df <- st_transform(sf_df, crs = 4326)
  }
  
  # Extract coordinates after transformation (if necessary)
  coords <- st_coordinates(sf_df)
  sf_df$lon <- coords[, 1]
  sf_df$lat <- coords[, 2]
  
  # Create the static map with points
  ggplot() +
    geom_point(data = sf_df, aes(x = lon, y = lat), alpha = alpha, size = size, color = "darkred") + 
    coord_map("mollweide") +  # Apply Mollweide projection
    theme_minimal() + 
    labs(title = title, x = "", y = "") +
    theme(
      plot.title = element_text(hjust = 0.5),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )
}

# Example usage
# Assuming you have a sf dataframe 'sf_df' with points in WGS84
# plot_points_mollweide(sf_df, title = "Mollweide Projection Map")

# ALL Data:
combined_acled2 <- preprocess_data(combined_acled)
m1 <- plot_points_mollweide(combined_acled2,alpha = 0.02, size = 0.3, title= "", zoom = 4);m1
ggplot2::ggsave("plots/test.png", m1,width = 10, height = 8, dpi = 600)

