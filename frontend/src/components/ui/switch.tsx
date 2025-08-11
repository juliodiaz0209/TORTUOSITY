"use client";

import * as React from "react";

type SwitchProps = {
  id?: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
  "aria-label"?: string;
};

export function Switch({ id, checked, onCheckedChange, disabled, className = "", ...rest }: SwitchProps) {
  const handleToggle = React.useCallback(() => {
    if (disabled) return;
    onCheckedChange(!checked);
  }, [checked, disabled, onCheckedChange]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>) => {
    if (disabled) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onCheckedChange(!checked);
    }
  };

  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-disabled={disabled}
      onClick={handleToggle}
      onKeyDown={handleKeyDown}
      disabled={disabled}
      className={
        `relative inline-flex h-6 w-11 items-center rounded-full transition-colors ` +
        `${checked ? "bg-[oklch(0.57_0.2_240)]" : "bg-[oklch(0.92_0_0)]"} ` +
        `${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"} ` +
        className
      }
      {...rest}
    >
      <span
        className={
          `inline-block h-5 w-5 transform rounded-full bg-white transition-transform shadow ` +
          `${checked ? "translate-x-5" : "translate-x-1"}`
        }
      />
    </button>
  );
}