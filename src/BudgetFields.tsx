import { useRef } from "react";
import type { Task } from "./types";

type Budget = Pick<Task, "minutes" | "trials">;
export default function BudgetFields({ minutes, trials, disabled, onChange }: Budget & { disabled?: boolean; onChange: (key: keyof Budget, value: number | null) => void }) {
  const previous = useRef({ minutes: 30, trials: 12 });
  return <div className="budget-fields">
    <div className="field-grid">{(["minutes", "trials"] as const).map(key => {
      const value = key === "minutes" ? minutes : trials;
      const other = key === "minutes" ? trials : minutes;
      if (value !== null && Number.isFinite(value)) previous.current[key] = value;
      return <div className="budget-limit" key={key}>
        <label className="check-label"><input type="checkbox" checked={value !== null} required={other === null} disabled={disabled} onChange={event => onChange(key, event.target.checked ? previous.current[key] : null)} /><span>{key === "minutes" ? "Max time" : "Max trials"}</span></label>
        <input type="number" aria-label={key === "minutes" ? "Minutes" : "Max trials"} required={value !== null} disabled={disabled || value === null} min="1" max={key === "minutes" ? 1440 : 100} value={value !== null && Number.isFinite(value) ? value : ""} placeholder={value === null ? "No limit" : key === "minutes" ? "e.g. 30" : "e.g. 12"} onChange={event => onChange(key, event.target.valueAsNumber)} />
        {key === "minutes" && value !== null && <small className="field-note">Minutes</small>}
      </div>;
    })}</div>
    <p className="field-note" role={minutes === null && trials === null ? "alert" : undefined}>{minutes === null && trials === null ? "Enable at least one limit." : "Use either limit, or both. Stops at whichever comes first."}</p>
  </div>;
}
