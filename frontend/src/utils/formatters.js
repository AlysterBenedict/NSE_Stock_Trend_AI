import { addBusinessDays, format } from 'date-fns';

export function getNextBusinessDay() {
  return format(addBusinessDays(new Date(), 1), 'yyyy-MM-dd');
}

export function isValidLineData(point) {
  return (
    typeof point === 'object' &&
    point !== null &&
    point.time &&
    point.value !== null &&
    typeof point.value === 'number' &&
    isFinite(point.value)
  );
}

export function formatINR(value, maximumFractionDigits = 0) {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits,
  }).format(value);
}

export function formatUnixOrIsoDate(timestamp) {
  try {
    if (!timestamp) return 'Unknown Date';
    const date = new Date(typeof timestamp === 'number' ? timestamp * 1000 : timestamp);
    return Number.isNaN(date.getTime()) ? 'Recent' : date.toLocaleString();
  } catch {
    return 'Recent';
  }
}

/** Indian-grouped number with fixed decimals; falls back to the raw value when not numeric. */
export function formatPrice(value, digits = 2) {
  const n = Number(value);
  if (value === null || value === undefined || value === '' || !Number.isFinite(n)) {
    return value ?? '—';
  }
  return n.toLocaleString('en-IN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Number with an explicit sign, e.g. +1.25 / -0.40. */
export function formatSigned(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return value;
  return `${n > 0 ? '+' : ''}${n.toFixed(digits)}`;
}

/** Maps a numeric move onto a semantic tone used by badges and text colours. */
export function toneOf(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return 'neutral';
  return n > 0 ? 'up' : 'down';
}

/** Return true when an Axios or Fetch request was aborted intentionally. */
export function isCancelledError(err) {
  return (
    err?.name === 'CanceledError' ||
    err?.name === 'AbortError' ||
    err?.code === 'ERR_CANCELED'
  );
}
