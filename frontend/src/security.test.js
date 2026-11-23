import { sanitizeWatchlistItems } from './context/WatchlistContext';
import { buildPrintableBriefDocument } from './utils/exportResearchBrief';
import { escapeHtml, isSafeHttpUrl } from './utils/urlSafety';

test('isSafeHttpUrl allows http/https and rejects javascript/data URIs', () => {
  expect(isSafeHttpUrl('https://finance.yahoo.com/news/story')).toBe(true);
  expect(isSafeHttpUrl('http://localhost:3000')).toBe(true);
  expect(isSafeHttpUrl('javascript:alert(1)')).toBe(false);
  expect(isSafeHttpUrl('  JaVaScRiPt:alert(document.cookie)  ')).toBe(false);
  expect(isSafeHttpUrl('data:text/html,<script>alert(1)</script>')).toBe(false);
  expect(isSafeHttpUrl('')).toBe(false);
  expect(isSafeHttpUrl(null)).toBe(false);
});

test('escapeHtml neutralizes HTML metacharacters', () => {
  expect(escapeHtml('<script>"alert"&\'1\'</script>')).toBe(
    '&lt;script&gt;&quot;alert&quot;&amp;&#39;1&#39;&lt;/script&gt;'
  );
});

test('buildPrintableBriefDocument prevents XSS injection via DOM textContent', () => {
  const doc = document.implementation.createHTMLDocument('');
  const xssPayload = '<img src=x onerror=alert(1)><script>window.pwned=true</script>';
  buildPrintableBriefDocument(doc, `Brief — ${xssPayload}`, [
    {
      heading: xssPayload,
      blocks: [
        { type: 'kv', items: [{ label: 'Equity', value: xssPayload }] },
        { type: 'table', headers: ['Factor'], rows: [[xssPayload]] },
      ],
    },
  ]);

  expect(doc.querySelectorAll('script').length).toBe(0);
  expect(doc.querySelectorAll('img').length).toBe(0);
  expect(doc.body.textContent).toContain(xssPayload);
});

test('sanitizeWatchlistItems filters corrupted entries, invalid alerts, and caps length', () => {
  expect(sanitizeWatchlistItems('not-an-array')).toEqual([]);

  const mixed = [
    null,
    { bad: true },
    { name: 'Infosys', symbol: 'INFY.NS', alertAbove: 2000, alertBelow: -50 },
    { name: 'Infosys Dup', symbol: 'INFY.NS', alertAbove: 'NaN' },
    { name: 'TCS', symbol: 'TCS.NS', alertAbove: 'Infinity', alertBelow: 3500 },
  ];
  const cleaned = sanitizeWatchlistItems(mixed);
  expect(cleaned).toHaveLength(2);
  expect(cleaned[0]).toMatchObject({ symbol: 'INFY.NS', alertAbove: 2000, alertBelow: null });
  expect(cleaned[1]).toMatchObject({ symbol: 'TCS.NS', alertAbove: null, alertBelow: 3500 });
});
