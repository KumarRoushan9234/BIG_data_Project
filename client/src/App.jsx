import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import ModelDescriptionPage from "./pages/ModelDescriptionPage";
import AboutPage from "./pages/AboutPage";
import DataVisualizationPage from "./pages/DataVisualizationPage";

const App = () => {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/model-description" element={<ModelDescriptionPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/data-visualization" element={<DataVisualizationPage />} />
      </Routes>
    </Router>
  );
};

export default App;
