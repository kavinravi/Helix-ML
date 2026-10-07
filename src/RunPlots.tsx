import type { Run } from "./types";

type Point = { x: number; y: number; label: string };
function Plot({ title, points, percent = false, baseline, xLabel }: { title: string; points: Point[]; percent?: boolean; baseline?: number; xLabel: string }) {
  if (!points.length) return null;
  const values = [...points.map(point => point.y), ...(baseline == null ? [] : [baseline])];
  const low = Math.min(...values), high = Math.max(...values);
  const padding = (high - low || Math.abs(high) || 1) * .12;
  const min = percent ? Math.max(0, low - padding) : low - padding;
  const max = percent ? Math.min(1, high + padding) : high + padding;
  const first = Math.min(...points.map(point => point.x)), last = Math.max(...points.map(point => point.x));
  const x = (value: number) => first === last ? 302 : 62 + (value - first) / (last - first) * 480;
  const y = (value: number) => 158 - (value - min) / (max - min || 1) * 126;
  const format = (value: number) => percent ? `${(value * 100).toFixed(1)}%` : Number(value.toPrecision(4)).toString();
  return <figure className="metric-plot">
    <figcaption>{title}</figcaption>
    <svg viewBox="0 0 570 200" role="img" aria-label={`${title}. ${points.length} measured points. Values available below.`}>
      {[min, (min + max) / 2, max].map((value, index) => <g key={index}><line className="plot-grid" x1="62" x2="542" y1={y(value)} y2={y(value)} /><text x="52" y={y(value) + 4} textAnchor="end">{format(value)}</text></g>)}
      {baseline != null && <line className="plot-baseline" x1="62" x2="542" y1={y(baseline)} y2={y(baseline)} />}
      <polyline className="plot-line" points={points.map(point => `${x(point.x)},${y(point.y)}`).join(" ")} />
      {points.map(point => <circle className="plot-point" key={point.x} cx={x(point.x)} cy={y(point.y)} r="3"><title>{point.label}: {format(point.y)}</title></circle>)}
      <text x={x(first)} y="178" textAnchor="middle">{first}</text>{last !== first && <text x={x(last)} y="178" textAnchor="middle">{last}</text>}
      <text x="302" y="196" textAnchor="middle">{xLabel}</text>
    </svg>
    <details className="plot-values"><summary>Values{baseline != null && <span>Dashed: baseline {format(baseline)}</span>}</summary><table><thead><tr><th>{xLabel}</th><th>Value</th></tr></thead><tbody>{points.map(point => <tr key={point.x}><td>{point.label}</td><td>{format(point.y)}</td></tr>)}</tbody></table></details>
  </figure>;
}
export default function RunPlots({ run }: { run: Run }) {
  const points = run.trials.flatMap((trial, index) => trial.score != null && Number.isFinite(trial.score) ? [{ x: run.followup ? index : index + 1, y: trial.score, label: trial.phase === "inheritance" ? "Previous best" : `${run.followup ? index : index + 1}. ${trial.name}` }] : []);
  return <div className="run-plots">
    <Plot title={`Validation ${run.task.metric.replaceAll("_", " ")}`} points={points} baseline={run.baseline?.score} percent={run.task.metric === "accuracy"} xLabel="Trial" />
    {(["loss", "accuracy"] as const).map(metric => {
      const history = (run.trainingHistory || []).flatMap(point => point[metric] == null ? [] : [{ x: point.step, y: point[metric]!, label: `Step ${point.step}` }]);
      return history.length > 1 ? <Plot key={metric} title={`Final model · training ${metric}`} points={history} percent={metric === "accuracy"} xLabel="Training step" /> : null;
    })}
  </div>;
}
