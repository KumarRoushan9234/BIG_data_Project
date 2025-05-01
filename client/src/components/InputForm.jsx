import React, { useState } from "react";

const InputForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    cardNumber: "",
    amount: "",
    transactionType: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({ ...prevData, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col space-y-4">
      <input
        type="text"
        name="cardNumber"
        placeholder="Card Number"
        value={formData.cardNumber}
        onChange={handleChange}
        className="p-2 border border-gray-300"
      />
      <input
        type="text"
        name="amount"
        placeholder="Amount"
        value={formData.amount}
        onChange={handleChange}
        className="p-2 border border-gray-300"
      />
      <input
        type="text"
        name="transactionType"
        placeholder="Transaction Type"
        value={formData.transactionType}
        onChange={handleChange}
        className="p-2 border border-gray-300"
      />
      <button type="submit" className="bg-blue-500 text-white p-2">
        Submit
      </button>
    </form>
  );
};

export default InputForm;
