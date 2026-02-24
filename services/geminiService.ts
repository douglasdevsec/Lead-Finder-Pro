/**
 * @file geminiService.ts
 * This service module handles all interactions with the Google Gemini API.
 * It includes functions for searching for business leads, scraping websites for contacts,
 * and geocoding addresses that are missing coordinates.
 */

import { GoogleGenAI, Type } from "@google/genai";
import { Business, LatLng, ScrapedData } from '../types';

// Initialize the GoogleGenAI client with the API key from environment variables.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

/**
 * Resultado de la normalización de la consulta del usuario.
 */
export interface NormalizedQuery {
    correctedQuery: string; // La consulta corregida e interpretada lista para buscar
    explanation: string;    // Explicación legible de lo que se interpretó (para mostrar al usuario)
    wasModified: boolean;   // true si la consulta original fue modificada
}

/**
 * Usa Gemini para normalizar e interpretar la consulta de búsqueda del usuario.
 * Corrige errores ortográficos, leet speak, abreviaciones y expande el término
 * al tipo de negocio adecuado.
 * @param {string} rawQuery - La consulta cruda ingresada por el usuario.
 * @returns {Promise<NormalizedQuery>} La consulta interpretada y corregida.
 */
export const normalizeQuery = async (rawQuery: string): Promise<NormalizedQuery> => {
    try {
        const prompt = `You are a smart search query interpreter for a business leads finder app.
Analyze the following user input and determine the correct business search query.

Your tasks:
1. Fix any typos (e.g. "pissa" → "pizza", "resturant" → "restaurant")
2. Convert leet speak or alphanumeric substitutions (e.g. "pizz4" → "pizza", "c0ffee" → "coffee", "b4r" → "bar")
3. Expand short/abbreviated terms to the full business type (e.g. "pizza" → "pizza restaurants", "plumb" → "plumbers", "elec" → "electricians")
4. Interpret the intent to find the right business category
5. Keep any location info the user typed (e.g. "pizza in Miami" → "pizza restaurants in Miami")

User input: "${rawQuery}"

Respond ONLY with a JSON object with exactly these keys:
- "correctedQuery": the clean, expanded, corrected search string ready for Google Maps search
- "explanation": a short 1-sentence human-readable explanation of what you interpreted (in the same language the user wrote)
- "wasModified": boolean, true if you changed anything from the original input

Example: user types "pissa nw york" → { "correctedQuery": "pizza restaurants in New York", "explanation": "Corregido 'pissa' a 'pizza' y expandido a restaurantes de pizza en New York.", "wasModified": true }`;

        const response = await ai.models.generateContent({
            model: 'gemini-2.0-flash',
            contents: prompt,
            config: {
                responseMimeType: 'application/json',
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        correctedQuery: { type: Type.STRING },
                        explanation:    { type: Type.STRING },
                        wasModified:    { type: Type.BOOLEAN },
                    },
                    required: ['correctedQuery', 'explanation', 'wasModified'],
                },
            },
        });

        const data = JSON.parse(response.text);
        return {
            correctedQuery: data.correctedQuery || rawQuery,
            explanation:    data.explanation || '',
            wasModified:    data.wasModified ?? false,
        };

    } catch (error) {
        console.warn('normalizeQuery falló, usando query original:', error);
        // Si falla, retornar la consulta original sin modificar para no bloquear la búsqueda.
        return { correctedQuery: rawQuery, explanation: '', wasModified: false };
    }
};



/**
 * Parses a markdown string response from the Gemini API into an array of Business objects.
 * @param {string} markdown - The markdown string returned by the API.
 * @returns {Business[]} An array of parsed business leads.
 */
function parseLeads(markdown: string): Business[] {
    const leads: Business[] = [];
    // Each business is separated by '---', so we split the string by this delimiter.
    const businesses = markdown.split('---').filter(b => b.trim() !== '');

    businesses.forEach((bizText, index) => {
        const lines = bizText.trim().split('\n');
        const name = lines[0].replace(/\*\*/g, '').trim();
        
        // Create a new Business object with a unique ID.
        const lead: Business = {
            id: `${Date.now()}-${index}`,
            name: name,
            address: '',
            type: '',
        };

        // Iterate over each line to extract details for the business.
        lines.slice(1).forEach(line => {
            const lowerLine = line.toLowerCase();
            if (lowerLine.includes('address:')) {
                lead.address = line.substring(line.indexOf(':') + 1).trim();
            } else if (lowerLine.includes('type:') || lowerLine.includes('category:')) {
                lead.type = line.substring(line.indexOf(':') + 1).trim();
            } else if (lowerLine.includes('phone:')) {
                lead.phone = line.substring(line.indexOf(':') + 1).trim();
            } else if (lowerLine.includes('website:')) {
                lead.website = line.substring(line.indexOf(':') + 1).trim();
            } else if (lowerLine.includes('coordinates:')) {
                 try {
                    const coordsText = line.substring(line.indexOf(':') + 1).trim();
                    const [lat, lng] = coordsText.split(',').map(c => parseFloat(c.trim()));
                    if (!isNaN(lat) && !isNaN(lng)) {
                       lead.latitude = lat;
                       lead.longitude = lng;
                    }
                } catch (e) {
                    console.error("Could not parse coordinates:", e);
                }
            } else if (lowerLine.includes('rating:')) {
                try {
                    const ratingText = line.substring(line.indexOf(':') + 1).trim();
                    const ratingMatch = ratingText.match(/(\d\.\d|\d)/);
                    const reviewsMatch = ratingText.match(/\((\d+)/);
                    if(ratingMatch) lead.rating = parseFloat(ratingMatch[0]);
                    if(reviewsMatch) lead.reviews = parseInt(reviewsMatch[1]);
                } catch(e) {
                    console.error("Could not parse rating:", e);
                }
            }
        });

        // Only add the lead if it has a name and address.
        if (lead.name && lead.address) {
            leads.push(lead);
        }
    });

    return leads;
}


/**
 * Searches for business leads using the Gemini API with Google Maps grounding.
 * @param {string} query - The user's search query (e.g., "restaurants in London").
 * @param {LatLng | null} location - The user's current location to improve search results.
 * @param {number} count - The maximum number of results to request.
 * @returns {Promise<Business[]>} A promise that resolves to an array of business leads.
 */
export const searchLeads = async (query: string, location: LatLng | null, count: number = 25): Promise<Business[]> => {
    try {
        // Construir prompt detallado para la API de Gemini incluyendo el número de resultados deseados.
        const prompt = `Find up to ${count} businesses matching '${query}'. For each business, provide its name, full address, category/type, main phone number, average star rating, number of reviews, official website URL, and geographic coordinates (latitude, longitude). Format each business entry clearly, separated by '---'. Example:\n**Business Name**\n- Address: 123 Main St, City, State, ZIP\n- Category: Category Type\n- Phone: (555) 555-5555\n- Rating: 4.5 (123 reviews)\n- Website: https://example.com\n- Coordinates: 40.7128, -74.0060\n---\n`;

        // Call the Gemini API, enabling the googleMaps tool for grounded results.
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                tools: [{ googleMaps: {} }],
                // If user location is available, provide it to the tool for better local results.
                ...(location && {
                    toolConfig: {
                        retrievalConfig: {
                            latLng: location
                        }
                    }
                })
            },
        });
        
        const markdownResponse = response.text;
        // Parse the markdown response into structured Business objects.
        return parseLeads(markdownResponse);

    } catch (error) {
        console.error("Error searching for leads:", error);
        throw new Error("Failed to fetch leads from Gemini API.");
    }
};


/**
 * Scrapes a website for contact information (emails, phones, social media links).
 * @param {string} websiteUrl - The URL of the website to scrape.
 * @returns {Promise<ScrapedData>} A promise that resolves to the scraped contact data.
 */
export const scrapeContacts = async (websiteUrl: string): Promise<ScrapedData> => {
    if(!websiteUrl || !websiteUrl.startsWith('http')) {
        throw new Error("Invalid or missing website URL.");
    }
    
    try {
        // Prompt for Gemini to analyze a website and extract contact info as a JSON object.
        const prompt = `Analyze the content of the website ${websiteUrl} and extract contact information. I need all unique email addresses, phone numbers, and social media profile links (specifically for Facebook, Instagram, LinkedIn, and Twitter). Provide the output strictly as a JSON object. The JSON object must have three keys: "emails", "phones", and "socials", where each key holds an array of unique strings. If no information is found for a key, provide an empty array.`;
        
        // Call the Gemini API, specifying a JSON response format and schema.
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        emails: { type: Type.ARRAY, items: { type: Type.STRING } },
                        phones: { type: Type.ARRAY, items: { type: Type.STRING } },
                        socials: { type: Type.ARRAY, items: { type: Type.STRING } },
                    },
                },
            },
        });

        const jsonString = response.text;
        const data = JSON.parse(jsonString);

        // Return the parsed data, ensuring arrays exist even if empty.
        return {
            emails: data.emails || [],
            phones: data.phones || [],
            socials: data.socials || [],
        };

    } catch (error) {
        console.error("Error scraping website:", error);
        throw new Error(`Failed to scrape contacts for ${websiteUrl}.`);
    }
};

/**
 * Batch geocodes a list of business addresses to get their latitude and longitude.
 * This is used as a fallback for leads that are missing coordinates from the initial search.
 * @param {{ id: string, address: string }[]} businesses - An array of business objects with ID and address.
 * @returns {Promise<Map<string, LatLng>>} A promise resolving to a Map where keys are business IDs and values are their coordinates.
 */
export const geocodeAddresses = async (businesses: { id: string, address: string }[]): Promise<Map<string, LatLng>> => {
    if (businesses.length === 0) {
        return new Map();
    }
    
    try {
        // Construct a prompt to ask Gemini for coordinates for a list of addresses.
        const prompt = `Provide the geographic coordinates (latitude and longitude) for the following list of businesses.
Input format is "ID: Address".
Your output MUST be a valid JSON object where keys are the business IDs and values are objects with "latitude" and "longitude" properties.
If you cannot find coordinates for a specific business ID, omit it from your JSON response.

Businesses:
${businesses.map(b => `${b.id}: ${b.address}`).join('\n')}
`;
        
        // Call the API, expecting a JSON response.
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
            },
        });

        // Clean up the response string in case it's wrapped in markdown backticks.
        const jsonString = response.text.trim().replace(/^```json|```$/g, '');
        const data = JSON.parse(jsonString);

        // Create a Map to store the results for efficient lookup.
        const coordinatesMap = new Map<string, LatLng>();
        for (const id in data) {
            if (Object.prototype.hasOwnProperty.call(data, id) && data[id].latitude && data[id].longitude) {
                const lat = parseFloat(data[id].latitude);
                const lng = parseFloat(data[id].longitude);
                // Ensure coordinates are valid numbers before adding them to the map.
                if (!isNaN(lat) && !isNaN(lng)) {
                  coordinatesMap.set(id, { latitude: lat, longitude: lng });
                }
            }
        }
        return coordinatesMap;

    } catch (error) {
        console.error("Error batch geocoding addresses:", error);
        return new Map(); // Return an empty map on failure to prevent crashes.
    }
};
