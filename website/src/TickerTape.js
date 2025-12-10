import React from 'react';
import './App.css';

const TickerTape = ({ data }) => {
    // Duplicate data to create seamless loop
    const loopData = [...data, ...data];

    return (
        <div className="ticker-tape-container">
            <div className="ticker-track">
                {loopData.map((item, index) => (
                    <div key={`${item.symbol}-${index}`} className="ticker-item">
                        <span className="ticker-symbol">{item.symbol.split('.')[0]}</span>
                        <span className="ticker-price">₹{item.price}</span>
                        <span className={`ticker-change ${item.change >= 0 ? 'pos' : 'neg'}`}>
                            {item.change >= 0 ? '▲' : '▼'} {Math.abs(item.pChange)}%
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default TickerTape;
