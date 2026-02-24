/**
 * @file index.tsx
 * This is the main entry point for the React application.
 * It imports the main App component and renders it into the DOM.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Find the root HTML element where the React app will be mounted.
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

// Create a React root for the found element.
const root = ReactDOM.createRoot(rootElement);

// Render the main App component within React's StrictMode.
// StrictMode helps with highlighting potential problems in an application.
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
