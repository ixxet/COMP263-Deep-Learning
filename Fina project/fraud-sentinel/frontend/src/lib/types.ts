export type RiskBand = 'low' | 'uncertain' | 'high';

export type TransactionInput = {
  Time: number;
  Amount: number;
} & Record<`V${number}`, number>;

export type PredictionResponse = {
  prediction_id: string;
  risk_score: number;
  anomaly_score: number;
  risk_band: RiskBand;
  model_version: string;
  case_id: string | null;
};

export type PredictionHistoryItem = {
  prediction_id: string;
  risk_score: number;
  anomaly_score: number;
  risk_band: RiskBand;
  model_version: string;
  created_at: string | null;
  transaction_time: number | null;
  amount: number | null;
  case_id: string | null;
  case_status: string | null;
};

export type BatchPredictionRow = {
  row_index: number;
  prediction_id: string;
  risk_score: number;
  anomaly_score: number;
  risk_band: RiskBand;
  model_version: string;
  case_id: string | null;
};

export type BatchRejectedRow = {
  row_index: number;
  reason: string;
};

export type BatchPredictionResponse = {
  accepted_rows: number;
  rejected_rows: number;
  prediction_ids: string[];
  case_ids: string[];
  rows: BatchPredictionRow[];
  rejections: BatchRejectedRow[];
};

export type CaseSummary = {
  case_id: string;
  prediction_id: string;
  status: string;
  risk_band: RiskBand;
  risk_score: number;
  anomaly_score: number;
  created_at: string | null;
  brief: string | null;
};

export type CaseDetail = CaseSummary & {
  transaction: Record<string, number>;
  policy_context: Array<Record<string, unknown>>;
  reviews: Array<Record<string, unknown>>;
  audit_events: Array<Record<string, unknown>>;
};
