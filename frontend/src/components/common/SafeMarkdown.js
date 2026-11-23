import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { isSafeHttpUrl } from '../../utils/urlSafety';

const SafeLink = ({ href, children, ...rest }) => {
  if (!isSafeHttpUrl(href)) {
    return <span>{children}</span>;
  }
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
      {children}
    </a>
  );
};

const SafeMarkdown = ({ children }) => (
  <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    urlTransform={(url) => (isSafeHttpUrl(url) ? url : '')}
    components={{ a: SafeLink }}
  >
    {children}
  </ReactMarkdown>
);

export default SafeMarkdown;
