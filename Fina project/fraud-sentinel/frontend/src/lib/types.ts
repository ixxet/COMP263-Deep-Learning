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

