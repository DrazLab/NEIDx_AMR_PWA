import React from 'react';
import { Button, Spinner } from 'react-bootstrap';


export default ({
  isLoading,
  text,
  loadingText,
  className = '',
  disabled = false,
  ...props
}) =>
  <Button
    className={`LoadButton ${className}`}
    disabled={disabled || isLoading}
    {...props}
  >
    {isLoading && <Spinner
      as="span"
      animation="border"
      size="sm"
      role="status"
      aria-hidden="true"
    />}{' '}
    {isLoading ? loadingText : text}
  </Button>;
