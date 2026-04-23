import type {
  BatchPredictionResponse,
  CaseDetail,
  CaseSummary,
  PredictionHistoryItem,
  PredictionResponse,
  RiskBand,
  TransactionInput
} from './types';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { 'content-type': 'application/json' }),
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function predict(transaction: TransactionInput): Promise<PredictionResponse> {
  return request('/api/v1/predict', {
    method: 'POST',
    body: JSON.stringify(transaction)
  });
}

export function uploadBatch(file: File): Promise<BatchPredictionResponse> {
  const form = new FormData();
  form.append('file', file);
  return request('/api/v1/predict/batch', {
    method: 'POST',
    body: form
  });
}

export function listPredictions(params: {
  risk_band?: RiskBand;
  has_case?: boolean;
  limit?: number;
} = {}): Promise<PredictionHistoryItem[]> {
  const query = new URLSearchParams();
  if (params.risk_band) query.set('risk_band', params.risk_band);
  if (params.has_case !== undefined) query.set('has_case', String(params.has_case));
  if (params.limit !== undefined) query.set('limit', String(params.limit));
  const suffix = query.toString() ? `?${query.toString()}` : '';
  return request(`/api/v1/predictions${suffix}`);
}

export function listCases(): Promise<CaseSummary[]> {
  return request('/api/v1/cases');
}

export function getCase(caseId: string): Promise<CaseDetail> {
  return request(`/api/v1/cases/${caseId}`);
}

export function reviewCase(
  caseId: string,
  payload: { action: 'approve' | 'escalate' | 'dismiss'; reviewer: string; rationale: string }
): Promise<{ case_id: string; status: string }> {
  return request(`/api/v1/cases/${caseId}/review`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}
