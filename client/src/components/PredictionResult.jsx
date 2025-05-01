const PredictionResult = ({ result }) => {
  return (
    <div className="mt-4 p-4 border border-gray-300 rounded-md">
      <h3 className="text-xl font-semibold">Prediction Result</h3>
      <p>
        <strong>Status:</strong> {result.status}
      </p>
      <p>
        <strong>Fraud Probability:</strong> {result.probability}
      </p>
    </div>
  );
};

export default PredictionResult;
