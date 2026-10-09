import CATALOG from './metric_catalog.json' with {type:'json'};
export { CATALOG };
export const METRICS = Object.keys(CATALOG);
export const higherIsBetter = metric => CATALOG[metric]?.maximize === true;
export const normalizeMetric = value => {
  if (typeof value !== "string") return value;
  const key=value.trim().toLowerCase().replaceAll("-","_").replaceAll(" ","_").replace(/^neg_/,"");
  return ({roc_auc:"auroc",roc_auc_score:"auroc",f1_score:"f1",precision_score:"precision",recall_score:"recall",accuracy_score:"accuracy",mean_squared_error:"mse",root_mean_squared_error:"rmse",mean_absolute_error:"mae",mean_absolute_percentage_error:"mape",mean_squared_log_error:"msle",root_mean_squared_log_error:"rmsle",brier_score_loss:"brier_score",db:"davies_bouldin",db_index:"davies_bouldin",davies_bouldin_index:"davies_bouldin",silhouette_score:"silhouette",calinski_harabasz_score:"calinski_harabasz",ari:"adjusted_rand",nmi:"normalized_mutual_info"})[key] || (key.endsWith("_score") && CATALOG[key.slice(0,-6)] ? key.slice(0,-6) : key);
};
export const isBetter = (metric, score, best) =>
  best === null || (higherIsBetter(metric) ? score > best : score < best);

export const objectiveLabel = task => (task.metrics?.length > 1 ? `Mean (${task.metrics.join(" + ")})` : task.metric);
