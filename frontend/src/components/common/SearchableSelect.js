import React, { useState, useEffect, useRef } from 'react';
import './SearchableSelect.css';

const SearchableSelect = ({ options, value, onChange, placeholder = 'Select...' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const wrapperRef = useRef(null);

  // Find the label for the current value to display initially
  const selectedOption = options.find((opt) => opt.value === value);

  useEffect(() => {
    if (selectedOption) {
      setSearchTerm(selectedOption.label);
    } else {
      setSearchTerm('');
    }
  }, [value, selectedOption]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsOpen(false);
        // Reset search term to selected value if closed without selection
        if (selectedOption) {
          setSearchTerm(selectedOption.label);
        } else {
          setSearchTerm('');
        }
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [selectedOption]);

  const handleInputChange = (e) => {
    setSearchTerm(e.target.value);
    setIsOpen(true);
  };

  const handleOptionClick = (option) => {
    onChange(option.value);
    setSearchTerm(option.label);
    setIsOpen(false);
  };

  const handleInputClick = () => {
    setIsOpen(!isOpen);
  };

  const filteredOptions = options.filter((option) =>
    option.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className={`searchable-select-wrapper${isOpen ? ' is-open' : ''}`} ref={wrapperRef}>
      <div className="searchable-input-container" onClick={handleInputClick}>
        <span className="searchable-lead material-symbols-outlined icon" aria-hidden="true">
          search
        </span>
        <input
          type="text"
          className="searchable-input"
          value={searchTerm}
          onChange={handleInputChange}
          onFocus={(e) => e.target.select()}
          placeholder={placeholder}
        />
        <span className={`arrow-icon icon ${isOpen ? 'open' : ''}`} aria-hidden="true">
          expand_more
        </span>
      </div>
      {isOpen && (
        <ul className="searchable-options-list" role="listbox">
          {filteredOptions.length > 0 ? (
            filteredOptions.map((option) => (
              <li
                key={option.value}
                role="option"
                aria-selected={option.value === value}
                className={`searchable-option ${option.value === value ? 'selected' : ''}`}
                onClick={() => handleOptionClick(option)}
              >
                <span>{option.label}</span>
                {option.value === value && (
                  <span className="icon searchable-check" aria-hidden="true">
                    check
                  </span>
                )}
              </li>
            ))
          ) : (
            <li className="searchable-option no-results">No results found</li>
          )}
        </ul>
      )}
    </div>
  );
};

export default SearchableSelect;
