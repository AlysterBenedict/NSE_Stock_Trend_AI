import React from 'react';
import { toneOf } from '../../utils/formatters';

/** Material Symbols glyph. Decorative by default. */
export const Icon = ({ name, fill = false, size, className = '', title }) => (
  <span
    className={`icon${fill ? ' icon--fill' : ''}${className ? ` ${className}` : ''}`}
    style={size ? { fontSize: size } : undefined}
    aria-hidden={title ? undefined : 'true'}
    title={title}
  >
    {name}
  </span>
);

/** Brand mark: NeuroStock logo image. */
export const BrandMark = ({ size = 36 }) => (
  <img
    src="/Logo.png"
    alt="NeuroStock Logo"
    className="brand-mark"
    width={size}
    height={size}
    style={{ objectFit: 'contain' }}
  />
);

/** Consistent page title block used at the top of every view. */
export const PageHeader = ({ eyebrow, icon, title, description, actions }) => (
  <header className="page-header">
    <div className="page-header__text">
      {eyebrow && (
        <div className="eyebrow">
          {icon && (
            <span className="eyebrow__icon">
              <Icon name={icon} fill size={15} />
            </span>
          )}
          <span>{eyebrow}</span>
        </div>
      )}
      <h1 className="page-header__title">{title}</h1>
      {description && <p className="page-header__description">{description}</p>}
    </div>
    {actions && <div className="page-header__actions">{actions}</div>}
  </header>
);

export const Card = ({ as: Tag = 'section', className = '', children, ...rest }) => (
  <Tag className={`card${className ? ` ${className}` : ''}`} {...rest}>
    {children}
  </Tag>
);

export const CardHeader = ({ icon, iconTone = 'primary', title, subtitle, actions, eyebrow }) => (
  <div className="card-header">
    <div className="card-header__lead">
      {icon && (
        <span className={`icon-tile icon-tile--${iconTone}`}>
          <Icon name={icon} size={18} />
        </span>
      )}
      <div className="card-header__text">
        {eyebrow && <span className="card-header__eyebrow">{eyebrow}</span>}
        <h2 className="card-header__title">{title}</h2>
        {subtitle && <p className="card-header__subtitle">{subtitle}</p>}
      </div>
    </div>
    {actions && <div className="card-header__actions">{actions}</div>}
  </div>
);

export const Badge = ({ tone = 'neutral', icon, children, className = '' }) => (
  <span className={`badge badge--${tone}${className ? ` ${className}` : ''}`}>
    {icon && <Icon name={icon} size={14} />}
    {children}
  </span>
);

/** Percentage / value change pill with a directional glyph. */
export const ChangeBadge = ({ value, suffix = '%', digits = 2 }) => {
  const tone = toneOf(value);
  const n = Number(value);
  const glyph = tone === 'up' ? 'north_east' : tone === 'down' ? 'south_east' : 'horizontal_rule';
  const text = Number.isFinite(n) ? `${n > 0 ? '+' : ''}${n.toFixed(digits)}${suffix}` : `${value}${suffix}`;
  return (
    <Badge tone={tone} icon={glyph}>
      {text}
    </Badge>
  );
};

export const Spinner = ({ size = 'md' }) => <span className={`spinner spinner--${size}`} role="status" aria-label="Loading" />;

export const Loader = ({ label = 'Loading…', className = '' }) => (
  <div className={`loader${className ? ` ${className}` : ''}`}>
    <Spinner />
    <span>{label}</span>
  </div>
);

export const EmptyState = ({ icon = 'insights', title, description, children }) => (
  <div className="empty-state">
    <span className="empty-state__icon">
      <Icon name={icon} size={28} />
    </span>
    {title && <h3 className="empty-state__title">{title}</h3>}
    {description && <p className="empty-state__description">{description}</p>}
    {children}
  </div>
);

export const Callout = ({ tone = 'neutral', icon = 'info', title, children }) => (
  <div className={`callout callout--${tone}`}>
    <Icon name={icon} size={20} className="callout__icon" />
    <div className="callout__body">
      {title && <strong className="callout__title">{title}</strong>}
      <div>{children}</div>
    </div>
  </div>
);

/** Label + value block used in KPI strips. */
export const Stat = ({ label, value, hint, tone, icon, emphasis = false }) => (
  <div className={`stat${tone ? ` stat--${tone}` : ''}${emphasis ? ' stat--emphasis' : ''}`}>
    <span className="stat__label">
      {icon && <Icon name={icon} size={15} />}
      {label}
    </span>
    <span className="stat__value">{value}</span>
    {hint && <span className="stat__hint">{hint}</span>}
  </div>
);

export const Field = ({ label, hint, htmlFor, children }) => (
  <div className="field">
    {(label || hint) && (
      <div className="field__label-row">
        {label && (
          <label className="field__label" htmlFor={htmlFor}>
            {label}
          </label>
        )}
        {hint && <span className="field__hint">{hint}</span>}
      </div>
    )}
    {children}
  </div>
);
