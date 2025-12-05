import React, { useState, useEffect, useRef } from 'react';
import './SearchableSelect.css';

const SearchableSelect = ({ options, value, onChange, placeholder = "Select..." }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const wrapperRef = useRef(null);

    // Find the label for the current value to display initially
    const selectedOption = options.find(opt => opt.value === value);

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

    const filteredOptions = options.filter(option =>
        option.label.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="searchable-select-wrapper" ref={wrapperRef}>
            <div className="searchable-input-container" onClick={handleInputClick}>
                <input
                    type="text"
                    className="searchable-input"
                    value={searchTerm}
                    onChange={handleInputChange}
                    placeholder={placeholder}
                />
                <span className={`arrow-icon ${isOpen ? 'open' : ''}`}>▼</span>
            </div>
            {isOpen && (
                <ul className="searchable-options-list">
                    {filteredOptions.length > 0 ? (
                        filteredOptions.map((option) => (
                            <li
                                key={option.value}
                                className={`searchable-option ${option.value === value ? 'selected' : ''}`}
                                onClick={() => handleOptionClick(option)}
                            >
                                {option.label}
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
