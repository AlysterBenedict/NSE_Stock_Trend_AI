import React from 'react';
import { Icon } from './ui';

const ChartSyncToggle = ({ isSyncing, onToggle, style = {} }) => (
  <button
    type="button"
    onClick={onToggle}
    style={style}
    className={`toggle-chip${isSyncing ? ' is-on' : ''}`}
    title={isSyncing ? 'Unlink Charts' : 'Link Charts'}
    aria-pressed={isSyncing}
  >
    <Icon name={isSyncing ? 'link' : 'link_off'} size={16} />
    {isSyncing ? 'Sync On' : 'Sync Off'}
  </button>
);

export default ChartSyncToggle;
