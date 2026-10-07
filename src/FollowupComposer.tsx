import { useState, type FormEvent } from "react";
export type Followup = { message: string; mode: "chat" | "trials"; minutes: number; trials: number };
export default function FollowupComposer({ disabled, onSend }: { disabled: boolean; onSend: (value: Followup) => Promise<boolean> }) {
  const [message, setMessage] = useState("");
  const [mode, setMode] = useState<Followup["mode"]>("chat");
  const [minutes, setMinutes] = useState(10), [trials, setTrials] = useState(3);
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
      {mode === "trials" && <div className="followup-budget"><label>Minutes<input type="number" required min="1" max="1440" placeholder="10" value={Number.isFinite(minutes) ? minutes : ""} disabled={disabled} onChange={event => setMinutes(event.target.valueAsNumber)} /></label><label>Trials<input type="number" required min="1" max="100" placeholder="3" value={Number.isFinite(trials) ? trials : ""} disabled={disabled} onChange={event => setTrials(event.target.valueAsNumber)} /></label></div>}
      <button type="submit" className="send-button" aria-label="Send follow-up" disabled={disabled || !message.trim()}>↑</button>
    </div>
  </form>;
}
