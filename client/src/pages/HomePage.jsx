import React, { useState } from "react";
import axios from "axios";
import InputForm from "../components/InputForm";
import PredictionResult from "../components/PredictionResult";

const HomePage = () => {
  const [result, setResult] = useState(null);

  const handleSubmit = async (formData) => {
    try {
      const response = await axios.post(
        "http://localhost:8000/predict",
        formData
      );
      setResult(response.data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <div className="flex flex-col items-center p-4">
      <div className="flex space-x-4 mb-6">
        <img
          src="/assets/credit-card-photo.png"
          alt="Credit Card"
          className="w-32 h-32"
        />
        <div className="w-1/2">
          <h2 className="text-xl font-semibold mb-2">
            Credit Card Fraud Detection Model
          </h2>
          <p>
            Our model uses advanced machine learning techniques to detect
            fraudulent credit card transactions.
          </p>
        </div>
      </div>
      <InputForm onSubmit={handleSubmit} />
      {result && <PredictionResult result={result} />}
    </div>
  );
};

export default HomePage;
