import { env } from '$env/dynamic/private';
import type { RequestHandler } from './$types';

async function proxy({ params, request, url }: Parameters<RequestHandler>[0]) {
  const path = params.path ?? '';
  const apiBaseUrl = env.API_BASE_URL ?? 'http://localhost:8000';
  const target = `${apiBaseUrl.replace(/\/$/, '')}/${path}${url.search}`;
  const headers = new Headers(request.headers);
  headers.delete('host');
  const response = await fetch(target, {
    method: request.method,
    headers,
    body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
    duplex: 'half'
  } as RequestInit & { duplex: 'half' });
  return new Response(response.body, {
    status: response.status,
    headers: response.headers
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const DELETE = proxy;
