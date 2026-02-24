/**
 * @file types.ts
 * This file contains all the TypeScript type definitions used throughout the application.
 * Defining these interfaces ensures type safety and improves code readability.
 */

/**
 * Represents a single business lead.
 * This is the core data structure for the application.
 */
export interface Business {
  id: string; // Unique identifier for the business
  name: string; // Name of the business
  address: string; // Full physical address
  type: string; // Business category or type (e.g., "Restaurant", "Plumber")
  phone?: string; // Contact phone number
  rating?: number; // Star rating (e.g., 4.5)
  reviews?: number; // Number of reviews
  website?: string; // Official website URL
  latitude?: number; // Geographic latitude
  longitude?: number; // Geographic longitude
  scrapedData?: ScrapedData; // Contact info scraped from the website
  isScraping?: boolean; // Flag to indicate if scraping is in progress for this lead
  scrapeError?: string; // Stores an error message if scraping fails
}

/**
 * Represents the data extracted from a business's website.
 */
export interface ScrapedData {
  emails: string[]; // Array of found email addresses
  phones: string[]; // Array of found phone numbers
  socials: string[]; // Array of found social media profile URLs
}

/**
 * Represents a geographic coordinate pair.
 */
export interface LatLng {
  latitude: number;
  longitude: number;
}

/**
 * Represents an item in the user's search history.
 */
export interface SearchHistoryItem {
  id: string; // Unique identifier for the history entry
  query: string; // The search query string used
  timestamp: number; // The time the search was performed
  resultCount: number; // The number of results found for the query
}
