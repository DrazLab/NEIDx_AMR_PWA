import React from 'react';
import './NotFound.css';

/**
 * This is rendered when a route is not found (404).
 */
export default () =>
  <div className="NotFound">
    <h1>404</h1>
    <h3>Not Found!</h3>
    <a href="/">Go Home/Refresh</a>
  </div>;
