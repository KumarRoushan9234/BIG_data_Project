import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="bg-gray-800 text-white p-4">
      <Link to="/" className="text-2xl font-bold">
        My App
      </Link>
      <ul className="flex space-x-4 ">
        <li>
          {/* <Link to="/" className="hover:text-blue-400">
            Home
          </Link> */}
        </li>
        <li>
          <Link to="/model-description" className="hover:text-blue-400">
            Model Description
          </Link>
        </li>
        <li>
          <Link to="/about" className="hover:text-blue-400">
            About
          </Link>
        </li>
        <li>
          <Link to="/data-visualization" className="hover:text-blue-400">
            Data Visualization
          </Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
