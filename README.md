# Lead Finder Pro

Lead Finder Pro is a powerful web application designed to find and qualify potential business leads. It leverages the Google Maps API via Gemini to search for businesses based on user queries, displays them on an interactive map, and provides tools to scrape contact information from their websites.

## ✨ Features

-   **Business Search**: Users can search for any type of business (e.g., "plumbers in new york") and get a detailed list of results.
-   **Interactive Map View**: All search results with valid coordinates are plotted on an interactive Leaflet map using OpenStreetMap tiles.
    -   **Dynamic Markers**: Each business has a corresponding marker on the map.
    -   **Map-List Sync**: Clicking a result in the list pans the map to its marker and highlights it. Clicking a marker on the map highlights the corresponding result in the list.
    -   **Auto-Fit**: The map automatically adjusts its zoom and center to display all markers after a search.
-   **Contact Scraping**: For each business with a website, users can click a "Scrape" button to extract contact details like emails, phone numbers, and social media links using a Gemini function.
-   **Batch Scraping**: A "Scrape All" feature allows users to automatically scrape contact information for all businesses in the current results list.
-   **Geocoding Fallback**: If a business from the initial search is missing coordinates, the application automatically uses the Gemini API to geocode its address, ensuring maximum visibility on the map.
-   **Data Export**: All gathered lead data, including scraped information, can be easily exported to a CSV file.
-   **Search History**: The application keeps a history of recent searches, allowing users to quickly re-run a previous query.
-   **Responsive Design**: The interface is built with Tailwind CSS for a clean, modern, and responsive user experience.

## 🚀 Technology Stack

-   **Frontend Framework**: React
-   **Language**: TypeScript
-   **AI/API**: Google Gemini API for searching, scraping, and geocoding.
-   **Mapping**: Leaflet.js with OpenStreetMap tiles.
-   **Styling**: Tailwind CSS
-   **Geolocation**: Browser Geolocation API to enhance local searches.

## 📁 File Structure

```
.
├── App.tsx                 # Main application component, manages state and logic.
├── components/
│   ├── icons.tsx           # SVG icon components used throughout the app.
│   └── ResultCard.tsx      # Component to display a single business lead.
├── services/
│   └── geminiService.ts    # Service module for all interactions with the Gemini API.
├── types.ts                # TypeScript type definitions for the application's data structures.
├── index.html              # The main HTML file, entry point of the app.
├── index.tsx               # Renders the main React App component into the DOM.
└── metadata.json           # Application metadata.
```

---
