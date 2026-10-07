import { useState, type FormEvent } from "react";
import BudgetFields from "./BudgetFields";
export type Followup = { message: string; mode: "chat" | "trials"; minutes: number | null; trials: number | null };
export default function FollowupComposer({ disabled, onSend }: { disabled: boolean; onSend: (value: Followup) => Promise<boolean> }) {
  const [message, setMessage] = useState("");
  const [mode, setMode] = useState<Followup["mode"]>("chat");
  const [minutes, setMinutes] = useState<number | null>(10), [trials, setTrials] = useState<number | null>(3);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (disabled || !message.trim()) return;
    const sent = message;
    setMessage("");
    if (!await onSend({ message: sent, mode, minutes, trials })) setMessage(sent);
  };
  return <form className="followup-composer" onSubmit={event => void submit(event)}>
    <textarea rows={2} aria-label="Follow-up message" placeholder={mode === "chat" ? "Ask about this result…" : "What should the next trials try?"} value={message} maxLength={4000} disabled={disabled} onChange={event => setMessage(event.target.value)} onKeyDown={event => {
      if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); }
    }} />
    <div className="followup-toolbar">
      <select aria-label="Follow-up action" value={mode} disabled={disabled} onChange={event => setMode(event.target.value as Followup["mode"])}><option value="chat">Ask agent</option><option value="trials">Run more trials</option></select>
      {mode === "trials" && <BudgetFields minutes={minutes} trials={trials} disabled={disabled} onChange={(key, value) => key === "minutes" ? setMinutes(value) : setTrials(value)} />}
      <button type="submit" className="send-button" aria-label="Send follow-up" disabled={disabled || !message.trim()}>↑</button>
    </div>
  </form>;
}
