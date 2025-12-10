import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const PositionsDashboard = () => {
    const [positions, setPositions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [sortConfig, setSortConfig] = useState({ key: 'pChange', direction: 'desc' });

    useEffect(() => {
        fetchPositions();
    }, []);

    const fetchPositions = async () => {
        try {
            const response = await axios.get('http://127.0.0.1:5000/get-market-data');
            // Initial sort: High Gainers first
            const sortedData = response.data.sort((a, b) => b.pChange - a.pChange);
            setPositions(sortedData);
            setLoading(false);
        } catch (error) {
            console.error("Error fetching positions:", error);
            setLoading(false);
        }
    };

    const handleSort = (key) => {
        let direction = 'desc';
        if (sortConfig.key === key && sortConfig.direction === 'desc') {
            direction = 'asc';
        }
        setSortConfig({ key, direction });

        const sorted = [...positions].sort((a, b) => {
            if (a[key] < b[key]) return direction === 'asc' ? -1 : 1;
            if (a[key] > b[key]) return direction === 'asc' ? 1 : -1;
            return 0;
        });
        setPositions(sorted);
    };

    const getRankColor = (index) => {
        if (index === 0) return 'gold-rank';
        if (index === 1) return 'silver-rank';
        if (index === 2) return 'bronze-rank';
        return '';
    };

    return (
        <div className="positions-container fade-in">
            <div className="section-header">
                <h2>Market Positions 📊</h2>
                <p className="subtitle">Daily Top Gainers & Losers (Nifty 50)</p>
            </div>

            {loading ? (
                <div className="loading-spinner">Loading Positions...</div>
            ) : (
                <div className="positions-table-wrapper">
                    <table className="positions-table">
                        <thead>
                            <tr>
                                <th onClick={() => handleSort('pChange')}>Rank</th>
                                <th onClick={() => handleSort('name')}>Stock Name</th>
                                <th onClick={() => handleSort('price')}>Current Price</th>
                                <th onClick={() => handleSort('open')}>Open</th>
                                <th onClick={() => handleSort('close')}>Prev Close</th>
                                <th onClick={() => handleSort('pChange')}>Gain/Loss %</th>
                            </tr>
                        </thead>
                        <tbody>
                            {positions.map((stock, index) => (
                                <tr key={stock.symbol} className="position-row">
                                    <td className={`rank-cell ${getRankColor(index)}`}>
                                        {index + 1}
                                    </td>
                                    <td className="stock-name-cell">
                                        {stock.name} <span className="ticker-sub">{stock.symbol.split('.')[0]}</span>
                                    </td>
                                    <td className="price-cell">₹{stock.price}</td>
                                    <td>₹{stock.open}</td>
                                    <td>₹{stock.close}</td>
                                    <td className={`change-cell ${stock.pChange >= 0 ? 'pos' : 'neg'}`}>
                                        {stock.pChange >= 0 ? '▲' : '▼'} {Math.abs(stock.pChange)}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default PositionsDashboard;
