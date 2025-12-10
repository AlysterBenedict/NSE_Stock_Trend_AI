import React from 'react';
import './App.css';

const StockGrid = ({ data, onSelectStock }) => {
    return (
        <div className="stock-grid">
            {data.map((stock) => (
                <div
                    key={stock.symbol}
                    className="stock-card-mini"
                    onClick={() => onSelectStock(stock)}
                >
                    <div className="stock-mini-header">
                        <span className="stock-mini-name">{stock.name}</span>
                        <span className={`stock-mini-badge ${stock.pChange >= 0 ? 'green' : 'red'}`}>
                            {stock.pChange >= 0 ? '+' : ''}{stock.pChange}%
                        </span>
                    </div>
                    <div className="stock-mini-price">
                        ₹{stock.price}
                    </div>
                    <div className="stock-mini-change">
                        {stock.change >= 0 ? '+' : ''}{stock.change}
                    </div>
                </div>
            ))}
        </div>
    );
};

export default StockGrid;
