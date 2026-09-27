import type { SVGProps } from "react";

/**
 * One icon family for the whole product: 24px grid, 1.7px stroke, round joins,
 * currentColor. Hand-rolled to avoid an extra runtime dependency.
 */

type IconProps = SVGProps<SVGSVGElement> & {
  size?: number;
};

function Svg({ size = 20, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

/** Send. Upward arrow, no wordmark. */
export function SendIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 19.2V5" />
      <path d="M5.4 11.6 12 5l6.6 6.6" />
    </Svg>
  );
}

/** External link, for "View source". */
export function ExternalIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M13.8 4H20v6.2" />
      <path d="M20 4 11.4 12.6" />
      <path d="M18.4 14.4v4.4A1.8 1.8 0 0 1 16.6 20.6H5.6a1.8 1.8 0 0 1-1.8-1.8V6.8a1.8 1.8 0 0 1 1.8-1.8h4.4" />
    </Svg>
  );
}

/** Rightward arrow, for prompt affordances. */
export function ArrowRightIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M4.5 12h15" />
      <path d="M13.4 6 19.5 12l-6.1 6" />
    </Svg>
  );
}

/** Information / about. */
export function InfoIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="12" cy="12" r="8.6" />
      <path d="M12 11.2v5" />
      <path d="M12 7.9h.01" />
    </Svg>
  );
}

/** Document, for the source citation. */
export function DocumentIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M13.4 3.2H7.6A1.6 1.6 0 0 0 6 4.8v14.4a1.6 1.6 0 0 0 1.6 1.6h8.8a1.6 1.6 0 0 0 1.6-1.6V7.6z" />
      <path d="M13.4 3.2v3.2a1.2 1.2 0 0 0 1.2 1.2h3.2" />
    </Svg>
  );
}

/** Retry, for genuine technical failures only. */
export function RetryIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M20 12a8 8 0 1 1-2.6-5.9" />
      <path d="M20.2 4.4v4.8h-4.8" />
    </Svg>
  );
}

/** Close dialog. */
export function CloseIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M6.6 6.6 17.4 17.4" />
      <path d="M17.4 6.6 6.6 17.4" />
    </Svg>
  );
}

/** Downward chevron, for "jump to latest". */
export function DownIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M6 9.6 12 15.6l6-6" />
    </Svg>
  );
}
