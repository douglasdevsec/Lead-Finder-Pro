/**
 * @file App.tsx
 * This is the root component of the Lead Finder Pro application.
 * It manages the main application state, handles API calls, and orchestrates
 * the interaction between all other components like the search bar, map, and results list.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Business, LatLng, SearchHistoryItem } from './types';
import { searchLeads, scrapeContacts, geocodeAddresses, normalizeQuery, NormalizedQuery } from './services/geminiService';
import { ResultCard } from './components/ResultCard';
import { LogoIcon, ExportIcon, SearchIcon, LoadingSpinner } from './components/icons';

// Constant for pagination: number of results to show per "Load More" click.
const RESULTS_PER_PAGE = 10;
// Constant for the search bar's placeholder text.
const PLACEHOLDER_TEXT = 'Get your next client. By Douglas P';

/**
 * The header component for the application.
 * @param {{ onExport: () => void }} props - Props containing the export callback.
 */
const Header: React.FC<{ onExport: () => void }> = ({ onExport }) => (
    <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
            <div className="flex items-center">
                <LogoIcon className="h-8 w-8 text-indigo-600" />
                <h1 className="text-2xl font-bold text-gray-800 ml-3">Lead Finder Pro</h1>
            </div>
            <div className="flex items-center space-x-6">
                <button 
                    onClick={onExport}
                    className="flex items-center bg-indigo-600 text-white font-semibold px-4 py-2 rounded-md hover:bg-indigo-700 transition">
                    <ExportIcon className="w-5 h-5 mr-2" />
                    Export All
                </button>
            </div>
        </div>
    </header>
);

/**
 * The search bar component with input field, filters, and submit button.
 */
const RESULT_COUNT_OPTIONS = [25, 50, 100];

const SearchBar: React.FC<{ 
    query: string;
    setQuery: (q: string) => void;
    onSubmit: () => void;
    isLoading: boolean;
    resultCount: number;
    setResultCount: (count: number) => void;
}> = ({ query, setQuery, onSubmit, isLoading, resultCount, setResultCount }) => {
    
    // Determine if the placeholder is currently active for styling and logic.
    const isPlaceholderActive = query === PLACEHOLDER_TEXT;

    // Handles form submission to trigger a search.
    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        // Prevent submitting if the query is empty or is still the placeholder.
        if (query.trim() && !isPlaceholderActive) {
            onSubmit();
        }
    };

    // When the user clicks on the input, clear the placeholder text.
    const handleFocus = () => {
        if (isPlaceholderActive) {
            setQuery('');
        }
    };

    // When the user clicks away, if the input is empty, restore the placeholder.
    const handleBlur = () => {
        if (query.trim() === '') {
            setQuery(PLACEHOLDER_TEXT);
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
            <h2 className="text-lg font-semibold text-gray-700 mb-4">Google Maps Search</h2>
            <form onSubmit={handleSubmit} className="flex items-center space-x-4">
                <input
                    type="text"
                    value={query}
                    onFocus={handleFocus}
                    onBlur={handleBlur}
                    onChange={(e) => setQuery(e.target.value)}
                    // Conditionally apply text color: gray for placeholder, black for user input.
                    className={`flex-grow w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 ${isPlaceholderActive ? 'text-gray-400' : 'text-gray-900'}`}
                />
                <select
                    value={resultCount}
                    onChange={(e) => setResultCount(Number(e.target.value))}
                    className="px-4 py-2 border border-gray-300 rounded-md bg-white focus:ring-indigo-500 focus:border-indigo-500"
                >
                    {RESULT_COUNT_OPTIONS.map(opt => (
                        <option key={opt} value={opt}>{opt} resultados</option>
                    ))}
                </select>
                <button type="submit" disabled={isLoading || isPlaceholderActive} className="flex items-center justify-center bg-blue-600 text-white font-semibold px-6 py-2 rounded-md hover:bg-blue-700 transition disabled:bg-blue-300 w-32">
                    {isLoading ? <LoadingSpinner className="w-5 h-5" /> : <><SearchIcon className="w-5 h-5 mr-2" /> Search</>}
                </button>
            </form>
        </div>
    );
};

/**
 * Props for the MapView component.
 */
interface MapViewProps {
    leads: Business[];
    userLocation: LatLng | null;
    selectedBusinessId: string | null;
    onMarkerClick: (id: string) => void;
    isLoading: boolean;
}

/**
 * Component to display leads on an interactive Leaflet map.
 */
const MapView: React.FC<MapViewProps> = ({ leads, userLocation, selectedBusinessId, onMarkerClick, isLoading }) => {
    const mapContainerRef = useRef<HTMLDivElement>(null); // Ref to the map container div.
    const mapRef = useRef<any>(null); // Ref to the Leaflet map instance.
    const markersRef = useRef<Map<string, any>>(new Map()); // Ref to a map of Leaflet marker instances, keyed by business ID.

    // Effect to initialize the map instance. Runs only once.
    useEffect(() => {
        if (mapContainerRef.current && !mapRef.current) {
            const L = (window as any).L;
            if (!L) {
                console.error("Leaflet is not loaded");
                return;
            }
            // Set initial map coordinates to user's location or a default.
            const initialCoords: [number, number] = userLocation 
                ? [userLocation.latitude, userLocation.longitude] 
                : [34.0522, -118.2437]; // Default to Los Angeles
            
            const map = L.map(mapContainerRef.current).setView(initialCoords, 10);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);
            mapRef.current = map;
        }
    }, [userLocation]);

    // Effect to update map markers whenever the list of leads changes.
    useEffect(() => {
        const L = (window as any).L;
        if (!mapRef.current || !L) return;

        // Clear existing markers from the map.
        markersRef.current.forEach(marker => marker.remove());
        markersRef.current.clear();

        const validLeads = leads.filter(lead => lead.latitude != null && lead.longitude != null);

        if (validLeads.length > 0) {
            const markerGroup = L.featureGroup();
            
            validLeads.forEach(lead => {
                // SECURITY FIX: Create popup content programmatically to prevent XSS.
                // Do not build HTML strings with untrusted data.
                const popupNode = document.createElement('div');
                popupNode.style.fontFamily = "'Inter', sans-serif";

                const nameNode = document.createElement('div');
                nameNode.style.fontWeight = '700';
                nameNode.style.fontSize = '1.1rem';
                nameNode.style.marginBottom = '4px';
                nameNode.textContent = lead.name; // Use textContent to safely render text.

                const addressNode = document.createElement('div');
                addressNode.style.fontSize = '0.9rem';
                addressNode.style.color = '#4B5563';
                addressNode.textContent = lead.address; // Use textContent to safely render text.
                
                popupNode.appendChild(nameNode);
                popupNode.appendChild(addressNode);

                if (lead.website) {
                    const websiteLink = document.createElement('a');
                    websiteLink.href = lead.website;
                    websiteLink.target = '_blank';
                    websiteLink.rel = 'noopener noreferrer'; // Security for target=_blank
                    websiteLink.textContent = 'Website';
                    websiteLink.style.fontSize = '0.9rem';
                    websiteLink.style.color = '#2563EB';
                    websiteLink.style.textDecoration = 'underline';
                    websiteLink.style.marginTop = '4px';
                    websiteLink.style.display = 'block';
                    popupNode.appendChild(websiteLink);
                }

                const marker = L.marker([lead.latitude!, lead.longitude!]).bindPopup(popupNode);
                
                // Add click listener to sync with the results list.
                marker.on('click', () => {
                    onMarkerClick(lead.id);
                });

                marker.addTo(markerGroup);
                markersRef.current.set(lead.id, marker); // Store marker instance.
            });
            
            markerGroup.addTo(mapRef.current);
            // Adjust map view to fit all markers.
            if (markerGroup.getBounds().isValid()) {
                mapRef.current.fitBounds(markerGroup.getBounds().pad(0.1));
            }
        } else if (userLocation) {
            // If no leads, center on user's location.
            mapRef.current.setView([userLocation.latitude, userLocation.longitude], 10);
        }

    }, [leads, userLocation, onMarkerClick]);

    // Effect to handle marker highlighting when a business is selected from the list.
    useEffect(() => {
        const L = (window as any).L;
        if (!L || !mapRef.current) return;
        
        // Define default and highlighted marker icons.
        const defaultIcon = new L.Icon.Default();
        const highlightedIcon = new L.Icon.Default({
             iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-violet.png',
             shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
             iconSize: [25, 41],
             iconAnchor: [12, 41],
             popupAnchor: [1, -34],
             shadowSize: [41, 41]
        });

        // Reset all markers to default icon, then set the selected one to highlighted.
        markersRef.current.forEach((marker, id) => {
            if (id === selectedBusinessId) {
                marker.setIcon(highlightedIcon);
                if(marker.setZIndexOffset) marker.setZIndexOffset(1000); // Bring to front
            } else {
                marker.setIcon(defaultIcon);
                 if(marker.setZIndexOffset) marker.setZIndexOffset(0);
            }
        });

        // If a business is selected, pan the map to it and open its popup.
        if (selectedBusinessId) {
            const selectedMarker = markersRef.current.get(selectedBusinessId);
            if (selectedMarker) {
                mapRef.current.flyTo(selectedMarker.getLatLng(), 15, { duration: 0.5 });
                selectedMarker.openPopup();
            }
        }

    }, [selectedBusinessId]);

    return (
        <div className="relative h-96 rounded-md overflow-hidden">
             <div ref={mapContainerRef} className="bg-gray-200 h-full w-full" />
             {/* Show a message if there are no results to display on the map */}
             {!isLoading && leads.length === 0 && (
                 <div className="absolute inset-0 flex items-center justify-center bg-gray-200 bg-opacity-80">
                    <p className="text-gray-600 font-semibold text-lg">No results to show on the map</p>
                </div>
             )}
        </div>
    );
};

/**
 * The main application component.
 */
const App: React.FC = () => {
    // --- STATE MANAGEMENT ---
    const [leads, setLeads] = useState<Business[]>([]); // Holds the list of business results.
    const [isLoading, setIsLoading] = useState(false); // Tracks loading state for search.
    const [error, setError] = useState<string | null>(null); // Stores any error messages.
    const [activeTab, setActiveTab] = useState('current'); // Manages which tab is active ('current' or 'history').
    const [userLocation, setUserLocation] = useState<LatLng | null>(null); // User's geolocation.
    const [searchQuery, setSearchQuery] = useState(PLACEHOLDER_TEXT); // The current value of the search input.
    const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([]); // List of past searches.
    const [isScrapingAll, setIsScrapingAll] = useState(false); // Tracks loading state for "Scrape All".
    const [visibleLeadsCount, setVisibleLeadsCount] = useState(RESULTS_PER_PAGE); // For "Load More" pagination.
    const [selectedBusinessId, setSelectedBusinessId] = useState<string | null>(null); // ID of the business selected in the list/map.
    const [resultCount, setResultCount] = useState(25); // Número de resultados a solicitar a la API.
    const [queryInterpretation, setQueryInterpretation] = useState<NormalizedQuery | null>(null); // Resultado de la interpretación inteligente de la consulta.
    const lastSearchTimeRef = useRef<number>(0); // Referencia para debounce: evita múltiples llamadas API accidentales.

    // --- EFFECTS ---
    // Effect runs on initial component mount.
    useEffect(() => {
        // Load search history from localStorage.
        try {
            const storedHistory = localStorage.getItem('leadFinderHistory');
            if (storedHistory) {
                setSearchHistory(JSON.parse(storedHistory));
            }
        } catch (e) {
            console.error("Failed to parse search history from localStorage", e);
            setSearchHistory([]);
        }

        // Get user's current geolocation.
        navigator.geolocation.getCurrentPosition(
            (position) => {
                setUserLocation({
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                });
            },
            (err) => {
                console.warn(`ERROR(${err.code}): ${err.message}`);
            },
            { timeout: 10000, maximumAge: 0 }
        );
    }, []);
    
    // --- HELPER FUNCTIONS ---
    // Updates search history state and persists it to localStorage.
    const updateSearchHistory = (newHistory: SearchHistoryItem[]) => {
        setSearchHistory(newHistory);
        localStorage.setItem('leadFinderHistory', JSON.stringify(newHistory));
    };

    // --- CORE LOGIC HANDLERS ---
    // Handles the main search functionality.
    // Incluye protección debounce: mínimo 1 segundo entre búsquedas para no saturar la API.
    const handleSearch = useCallback(async (query: string) => {
        if (!query || query === PLACEHOLDER_TEXT) return;
        const now = Date.now();
        if (now - lastSearchTimeRef.current < 1000) return; // Debounce: 1 segundo mínimo
        lastSearchTimeRef.current = now;
        setIsLoading(true);
        setError(null);
        setLeads([]);
        setSelectedBusinessId(null);
        setVisibleLeadsCount(RESULTS_PER_PAGE);
        setQueryInterpretation(null);
        try {
            // Step 1: Normalizar/corregir la consulta usando Gemini antes de buscar.
            const normalized = await normalizeQuery(query);
            setQueryInterpretation(normalized);
            const effectiveQuery = normalized.correctedQuery;

            // Step 2: Obtener resultados con la consulta normalizada.
            const initialResults = await searchLeads(effectiveQuery, userLocation, resultCount);
            
            // Step 3: Identify leads that need geocoding.
            const toGeocode = initialResults
                .filter(lead => lead.address && (lead.latitude == null || lead.longitude == null))
                .map(lead => ({ id: lead.id, address: lead.address }));

            // Step 4: Batch geocode addresses if necessary.
            if (toGeocode.length > 0) {
                const coordinatesMap = await geocodeAddresses(toGeocode);
                // Step 4: Merge coordinates back into the results.
                const finalResults = initialResults.map(lead => {
                    if (coordinatesMap.has(lead.id)) {
                        const coords = coordinatesMap.get(lead.id)!;
                        return { ...lead, latitude: coords.latitude, longitude: coords.longitude };
                    }
                    return lead;
                });
                setLeads(finalResults);
            } else {
                setLeads(initialResults);
            }
            
            // Step 5: Update search history.
            const newHistoryItem: SearchHistoryItem = { id: `${Date.now()}`, query, timestamp: Date.now(), resultCount: initialResults.length };
            const updatedHistory = [newHistoryItem, ...searchHistory.filter(h => h.query !== query)].slice(0, 20);
            updateSearchHistory(updatedHistory);

        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [userLocation, searchHistory]);

    // Handles scraping a single business's website.
    const handleScrape = useCallback(async (businessId: string, websiteUrl: string) => {
        // Update UI to show scraping is in progress for this specific card.
        setLeads(prevLeads => prevLeads.map(lead =>
            lead.id === businessId ? { ...lead, isScraping: true, scrapeError: undefined } : lead
        ));

        try {
            const scrapedData = await scrapeContacts(websiteUrl);
            // Update the lead with the successfully scraped data.
            setLeads(prevLeads => prevLeads.map(lead =>
                lead.id === businessId ? { ...lead, isScraping: false, scrapedData } : lead
            ));
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Scraping failed.';
            // Update the lead with an error message if scraping fails.
            setLeads(prevLeads => prevLeads.map(lead =>
                lead.id === businessId ? { ...lead, isScraping: false, scrapeError: errorMessage } : lead
            ));
        }
    }, []);
    
    // Handles exporting all current leads to a CSV file.
    const handleExportAll = useCallback(() => {
        if (leads.length === 0) {
            alert("No leads to export.");
            return;
        }

        const headers = ["Name", "Address", "Type", "Phone", "Rating", "Reviews", "Website", "Scraped Emails", "Scraped Phones", "Scraped Socials"];
        
        // SECURITY FIX: Sanitize data for CSV export to prevent Formula Injection.
        const escapeCsvCell = (cellData: any) => {
            let stringData = String(cellData || '');

            // Sanitize against CSV injection
            if (['=', '+', '-', '@'].includes(stringData.charAt(0))) {
                stringData = "'" + stringData;
            }

            // Escape quotes and wrap in quotes if it contains a comma.
            if (stringData.includes(',')) {
                return `"${stringData.replace(/"/g, '""')}"`;
            }
            return stringData;
        };

        const csvRows = leads.map(lead => [
            escapeCsvCell(lead.name), escapeCsvCell(lead.address), escapeCsvCell(lead.type),
            escapeCsvCell(lead.phone), escapeCsvCell(lead.rating), escapeCsvCell(lead.reviews),
            escapeCsvCell(lead.website), escapeCsvCell(lead.scrapedData?.emails.join('; ') || ''),
            escapeCsvCell(lead.scrapedData?.phones.join('; ') || ''), escapeCsvCell(lead.scrapedData?.socials.join('; ') || '')
        ].join(','));
        
        const csvContent = [headers.join(','), ...csvRows].join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.href = url;
        link.setAttribute("download", "leads.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }, [leads]);
    
    // Handles the "Scrape All" button click, iterating through leads.
    const handleScrapeAll = async () => {
        setIsScrapingAll(true);
        for (const lead of leads) {
            // Only scrape if it has a website and hasn't been scraped yet.
            if (lead.website && !lead.scrapedData && !lead.scrapeError) {
                await handleScrape(lead.id, lead.website);
            }
        }
        setIsScrapingAll(false);
    };

    // Increases the number of visible leads for pagination.
    const handleLoadMore = () => {
        setVisibleLeadsCount(prevCount => prevCount + RESULTS_PER_PAGE);
    };
    
    // Reruns a search from the history tab.
    const handleRerunSearch = (query: string) => {
        setActiveTab('current');
        setSearchQuery(query);
        handleSearch(query);
    };

    // Clears the search history.
    const handleClearHistory = () => {
        updateSearchHistory([]);
    };

    // Handles selecting a business card, syncing the map and list.
    const handleSelectBusiness = (id: string) => {
        // Toggle selection off if the same card is clicked again.
        setSelectedBusinessId(prevId => prevId === id ? null : id);
        
        // Auto-scroll logic: if the selected business is outside the visible area, "load more" until it's visible.
        const selectedIndex = leads.findIndex(lead => lead.id === id);
        if (selectedIndex >= visibleLeadsCount) {
             setVisibleLeadsCount(selectedIndex + 1);
        }
    };

    // --- RENDER ---
    return (
        <div className="min-h-screen bg-gray-100">
            <Header onExport={handleExportAll} />
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="space-y-8">
                    <SearchBar query={searchQuery} setQuery={setSearchQuery} onSubmit={() => handleSearch(searchQuery)} isLoading={isLoading} resultCount={resultCount} setResultCount={setResultCount} />
                    
                    {/* Banner de interpretación inteligente: se muestra cuando Gemini corrigió o expandió la consulta */}
                    {queryInterpretation?.wasModified && (
                        <div className="flex items-start justify-between bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-sm text-amber-800">
                            <div className="flex items-center">
                                <span className="text-lg mr-2">🔍</span>
                                <span>{queryInterpretation.explanation}</span>
                            </div>
                            <button
                                onClick={() => setQueryInterpretation(null)}
                                className="ml-4 text-amber-500 hover:text-amber-700 font-bold text-base leading-none"
                                aria-label="Cerrar"
                            >✕</button>
                        </div>
                    )}
                    
                    {/* Tab Navigation */}
                    <div className="border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                            <button onClick={() => setActiveTab('current')} className={`${activeTab === 'current' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}>
                                Current Results
                            </button>
                            <button onClick={() => setActiveTab('history')} className={`${activeTab === 'history' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}>
                                Search History
                            </button>
                        </nav>
                    </div>

                    {/* Conditional rendering based on active tab */}
                    {activeTab === 'current' && (
                        <div className="space-y-8">
                            <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
                                <h2 className="text-lg font-semibold text-gray-700 mb-4">Map View</h2>
                                <MapView 
                                    leads={leads} 
                                    userLocation={userLocation}
                                    selectedBusinessId={selectedBusinessId}
                                    onMarkerClick={handleSelectBusiness}
                                    isLoading={isLoading}
                                />
                            </div>

                            <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
                                <div className="flex justify-between items-center mb-4">
                                    <h2 className="text-lg font-semibold text-gray-700">{leads.length > 0 ? `${leads.length} results` : 'Search Results'}</h2>
                                    {leads.length > 0 && 
                                        <button onClick={handleScrapeAll} disabled={isScrapingAll} className="bg-green-100 text-green-700 font-semibold px-4 py-2 rounded-md hover:bg-green-200 transition disabled:bg-gray-200 disabled:text-gray-500 disabled:cursor-not-allowed flex items-center">
                                            {isScrapingAll && <LoadingSpinner className="w-4 h-4 mr-2" />}
                                            {isScrapingAll ? 'Scraping All...' : 'Scrape All Websites'}
                                        </button>
                                    }
                                </div>
                                {isLoading && <div className="text-center py-10"><LoadingSpinner className="w-8 h-8 mx-auto text-blue-600" /></div>}
                                {error && <div className="text-center py-10 text-red-500">{error}</div>}
                                
                                <div className="space-y-4">
                                    {!isLoading && leads.length === 0 && !error && <div className="text-center py-10 text-gray-500">Search to see results here.</div>}
                                    {leads.slice(0, visibleLeadsCount).map(business => (
                                        <ResultCard 
                                            key={business.id} 
                                            business={business} 
                                            onScrape={handleScrape} 
                                            isSelected={business.id === selectedBusinessId}
                                            onSelect={handleSelectBusiness}
                                        />
                                    ))}
                                </div>

                                {visibleLeadsCount < leads.length && (
                                    <div className="text-center mt-8">
                                        <button onClick={handleLoadMore} className="bg-gray-100 text-gray-700 font-semibold px-6 py-2 rounded-md hover:bg-gray-200 transition">
                                            Load More
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                    
                    {activeTab === 'history' && (
                         <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
                             <div className="flex justify-between items-center mb-4">
                                <h2 className="text-lg font-semibold text-gray-700">Search History</h2>
                                {searchHistory.length > 0 && 
                                    <button onClick={handleClearHistory} className="text-sm text-red-500 hover:underline">Clear History</button>}
                             </div>
                             {searchHistory.length === 0 ? (
                                <p className="text-gray-500 text-center py-8">No search history yet.</p>
                             ) : (
                                 <ul className="divide-y divide-gray-200">
                                     {searchHistory.map(item => (
                                         <li key={item.id} className="py-3 flex justify-between items-center">
                                             <div>
                                                 <p className="font-medium text-gray-800">{item.query}</p>

                                                 <p className="text-sm text-gray-500">
                                                     {new Date(item.timestamp).toLocaleString()} - {item.resultCount} results
                                                 </p>
                                             </div>
                                             <button onClick={() => handleRerunSearch(item.query)} className="bg-gray-100 text-gray-700 font-semibold px-4 py-1.5 rounded-md text-sm hover:bg-gray-200 transition">
                                                Rerun
                                            </button>
                                         </li>
                                     ))}
                                 </ul>
                             )}
                         </div>
                    )}
                </div>
            </main>
        </div>
    );
};

export default App;